import os
import cv2
import random
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler

import visdom

vis = visdom.Visdom()
vis.close(env="main")

from loader import KITTI_Train_Dataset, KITTI_Validation_Dataset, KITTI_Test_Dataset

from augmentations import Compose, RandomRotate, RandomHorizontallyFlip, RandomCrop, get_composed_augmentations
from utils import conv2DBatchNormRelu, residualBlockPSP, pyramidPooling, get_logger
from scheduler import PolynomialLR
from loss import multi_scale_cross_entropy2d, get_loss_function
from optimizer import get_optimizer
from metrics import runningScore, averageMeter

from tensorboardX import SummaryWriter

DATA_DIR = 'data_road'
# TRAIN_ITERS = 300000
TRAIN_ITERS = 1
VAL_INTERVAL = 1000
N_WORKERS = 16
PRINT_INTERVAL = 50

NUM_OF_CLASS = 2
IMAGE_SIZE = (576, 160)
BATCH_SIZE = 4

def value_tracker(value_plot, value, num):
    """ num, loss_value are Tensor """
    vis.line(X=num,
             Y=value,
             win=value_plot,
             update='append')

augmentations = Compose([RandomRotate(10), RandomHorizontallyFlip(10)])
train_dataset = KITTI_Train_Dataset(DATA_DIR, "image", IMAGE_SIZE, is_transform=True, augmentations=augmentations)
trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=8)

for i, data_samples in enumerate(trainloader):
    print("Test", i)
    imgs, labels = data_samples
    imgs = imgs.numpy()[:, ::-1, :, :]
    imgs = np.transpose(imgs, [0, 2, 3, 1])
    f, axarr = plt.subplots(BATCH_SIZE, 2)
    for j in range(BATCH_SIZE):
        axarr[j][0].imshow(imgs[j])
        cv2.imshow("imgs", imgs[j])
        cv2.waitKey(0)
        axarr[j][1].imshow(train_dataset.decode_segmap(labels.numpy()[j]))

    plt.show()
    a = input()
    if a == 'ex':
        break
    else:
        plt.close()

pspnet_specs = {
    "kitti": {"n_classes": 2, "input_size": (160, 576), "block_config": [3, 4, 23, 3]}
}

class PSPNet(nn.Module):
    """
    Pyramid Scene Parsing Network
    URL: https://arxiv.org/abs/1612.01105


    """

    def __init__(self, n_classes=2, block_config=[3, 4, 23, 3], input_size=(160, 576), version="kitti"):
        super(PSPNet, self).__init__()

        self.block_config = (
            pspnet_specs[version]["block_config"] if version is not None else block_config
        )

        self.n_classes = pspnet_specs[version]["n_classes"] if version is not None else n_classes
        self.input_size = pspnet_specs[version]["input_size"] if version is not None else input_size

        # Encoder
        self.convbnrelu1_1 = conv2DBatchNormRelu(
            in_channels=3, k_size=3, n_filters=64, padding=1, stride=2, bias=False
        )
        self.convbnrelu1_2 = conv2DBatchNormRelu(
            in_channels=64, k_size=3, n_filters=64, padding=1, stride=1, bias=False
        )
        self.convbnrelu1_3 = conv2DBatchNormRelu(
            in_channels=64, k_size=3, n_filters=128, padding=1, stride=1, bias=False
        )

        # Vanilla Residual Blocks
        self.res_block2 = residualBlockPSP(self.block_config[0], 128, 64, 256, 1, 1)
        self.res_block3 = residualBlockPSP(self.block_config[1], 256, 128, 512, 2, 1)

        # Dilated Residual Blocks
        self.res_block4 = residualBlockPSP(self.block_config[2], 512, 256, 1024, 1, 2)
        self.res_block5 = residualBlockPSP(self.block_config[3], 1024, 512, 2048, 1, 4)

        # Pyramid Pooling Module
        self.pyramid_pooling = pyramidPooling(2048, [6, 3, 2, 1])

        # Final conv layers
        self.cbr_final = conv2DBatchNormRelu(4096, 512, 3, 1, 1, False)
        self.dropout = nn.Dropout2d(p=0.1, inplace=False)
        self.classification = nn.Conv2d(512, self.n_classes, 1, 1, 0)

        # Auxiliary layers for training
        self.convbnrelu4_aux = conv2DBatchNormRelu(
            in_channels=1024, k_size=3, n_filters=256, padding=1, stride=1, bias=False
        )
        self.aux_cls = nn.Conv2d(256, self.n_classes, 1, 1, 0)

        # Define auxiliary loss function
        self.loss = multi_scale_cross_entropy2d

    def forward(self, x):
        inp_shape = x.shape[2:]

        # H, W -> H/2, W/2
        x = self.convbnrelu1_1(x)
        x = self.convbnrelu1_2(x)
        x = self.convbnrelu1_3(x)

        # H/2, W/2 -> H/4, W/4
        x = F.max_pool2d(x, 3, 2, 1)

        # H/4, W/4 -> H/8, W/8
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)

        # Auxiliary layers for training
        if self.training:
            x_aux = self.convbnrelu4_aux(x)
            x_aux = self.dropout(x_aux)
            x_aux = self.aux_cls(x_aux)

        x = self.res_block5(x)

        x = self.pyramid_pooling(x)

        x = self.cbr_final(x)
        x = self.dropout(x)

        x = self.classification(x)
        x = F.interpolate(x, size=inp_shape, mode="bilinear", align_corners=True)

        if self.training:
            return (x, x_aux)
        else:
            return x

    def tile_predict(self, imgs, include_flip_mode=True):
        """
        Predict by takin overlapping tiles from the image.
        Strides are adaptively computed from the imgs shape
        and input size
        :param imgs: torch.Tensor with shape [N, C, H, W] in BGR format
        :param side: int with side length of model input
        :param n_classes: int with number of classes in seg output.
        """

        side_x, side_y = self.input_size
        n_classes = self.n_classes
        n_samples, c, h, w = imgs.shape
        # n = int(max(h,w) / float(side) + 1)
        n_x = int(h / float(side_x) + 1)
        n_y = int(w / float(side_y) + 1)
        stride_x = (h - side_x) / float(n_x)
        stride_y = (w - side_y) / float(n_y)

        x_ends = [[int(i * stride_x), int(i * stride_x) + side_x] for i in range(n_x + 1)]
        y_ends = [[int(i * stride_y), int(i * stride_y) + side_y] for i in range(n_y + 1)]

        pred = np.zeros([n_samples, n_classes, h, w])
        count = np.zeros([h, w])

        slice_count = 0
        for sx, ex in x_ends:
            for sy, ey in y_ends:
                slice_count += 1

                imgs_slice = imgs[:, :, sx:ex, sy:ey]
                if include_flip_mode:
                    imgs_slice_flip = torch.from_numpy(
                        np.copy(imgs_slice.cpu().numpy()[:, :, :, ::-1])
                    ).float()

                is_model_on_cuda = next(self.parameters()).is_cuda

                inp = Variable(imgs_slice, volatile=True)
                if include_flip_mode:
                    flp = Variable(imgs_slice_flip, volatile=True)

                if is_model_on_cuda:
                    inp = inp.cuda()
                    if include_flip_mode:
                        flp = flp.cuda()

                psub1 = F.softmax(self.forward(inp), dim=1).data.cpu().numpy()
                if include_flip_mode:
                    psub2 = F.softmax(self.forward(flp), dim=1).data.cpu().numpy()
                    psub = (psub1 + psub2[:, :, :, ::-1]) / 2.0
                else:
                    psub = psub1

                pred[:, :, sx:ex, sy:ey] = psub
                count[sx:ex, sy:ey] += 1.0

        score = (pred / count[None, None, ...]).astype(np.float32)
        return score / np.expand_dims(score.sum(axis=1), axis=1)


# Setup seeds
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
np.random.seed(1337)
random.seed(1337)

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup Augmentation
# augmentations = None
# data_aug = get_composed_augmentations(augmentations)

# Setup Dataloader
augmentations = Compose([RandomRotate(10), RandomHorizontallyFlip(10)])
train_dataset = KITTI_Train_Dataset(DATA_DIR, "image", IMAGE_SIZE, is_transform=True, augmentations=augmentations)
validation_dataset = KITTI_Validation_Dataset(DATA_DIR, "image", IMAGE_SIZE, is_transform=True)

trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
validloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)

run_id = random.randint(1, 100000)
logdir = os.path.join("runs", "psp_kitti_road", str(run_id))
writer = SummaryWriter(log_dir=logdir)

print("RUNDIR: {}".format(logdir))

logger = get_logger(logdir)
logger.info("Let the games begin")

# Setup Metrics
running_metrics_val = runningScore(n_classes=NUM_OF_CLASS)

# Setup Model
model = PSPNet(version="kitti").to(device)

a = torch.Tensor(1, 3, 160, 576).to(device)

model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

# Setup optimizer, lr_scheduler and loss function
optimizer_cls = get_optimizer("sgd")

optimizer = optimizer_cls(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
logger.info("Using Optimizer {}".format(optimizer))

# scheduler = get_scheduler(optimizer, cfg["training"])
scheduler = PolynomialLR(optimizer, max_iter=TRAIN_ITERS, decay_iter=1, gamma=0.9, last_epoch=-1)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

loss_fn = get_loss_function("cross_entropy")
logger.info("Using loss {}".format(loss_fn))

start_iter = 0
if "pspnet_kitti_best_model.pkl" is not None:
    if os.path.isfile("pspnet_kitti_best_model.pkl"):
        logger.info(
            "Loading model and optimizer from checkpoint '{}'".format("pspnet_kitti_best_model.pkl")
        )
        checkpoint = torch.load("pspnet_kitti_best_model.pkl")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_iter = checkpoint["epoch"]
        logger.info(
            "Loaded checkpoint '{}' (iter {})".format(
                "pspnet_kitti_best_model.pkl", checkpoint["epoch"]
            )
        )
    else:
        logger.info("No checkpoint found at '{}'".format("pspnet_kitti_best_model.pkl"))

val_loss_meter = averageMeter()
time_meter = averageMeter()

best_iou = -100.0
i = start_iter
flag = True

while i <= TRAIN_ITERS and flag:
    for i, data in (trainloader, 0):
        images, labels = data
        i += 1
        start_ts = time.time()
        scheduler.step()
        model.train()
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = loss_fn(input=outputs, target=labels)

        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - start_ts)

        if (i + 1) % 50 == 0:
            fmt_str = "Iter [{:d}/{:d} Loss: {:.4f} Time/Image: {:.4f}"
            print_str = fmt_str.format(
                i + 1,
                TRAIN_ITERS,
                loss.item(),
                time_meter.avg / BATCH_SIZE,
            )

            print(print_str)
            logger.info(print_str)
            writer.add_scalar("loss/train_loss", loss.item(), i + 1)
            time_meter.reset()

        if (i + 1) % VAL_INTERVAL == 0 or (i + 1) == TRAIN_ITERS:
            model.eval()
            with torch.no_grad():
                for i_val, (images_val, labels_val) in tqdm(enumerate(validloader)):
                    images_val = images_val.to(device)
                    labels_val = labels_val.to(device)

                    outputs = model(images_val)
                    val_loss = loss_fn(input=outputs, target=labels_val)

                    pred = outputs.data.max(1)[1].cpu().numpy()
                    gt = labels_val.data.cpu().numpy()

                    running_metrics_val.update(gt, pred)
                    val_loss_meter.update(val_loss.item())

            writer.add_scalar("loss/val_loss", val_loss_meter.avg, i + 1)
            logger.info("Iter %d Loss: %.4f" % (i + 1, val_loss_meter.avg))

            score, class_iou = running_metrics_val.get_scores()
            for k, v in score.items():
                print(k, v)
                logger.info("{}: {}".format(k, v))
                writer.add_scalar("val_metrics/{}".format(k), v, i + 1)

            for k, v in class_iou.items():
                logger.info("{}: {}".format(k, v))
                writer.add_scalar("val_metrics/cls_{}".format(k), v, i + 1)

            val_loss_meter.reset()
            running_metrics_val.reset()

            if score["Mean IoU : \t"] >= best_iou:
                best_iou = score["Mean IoU : \t"]
                state = {
                    "epoch": i + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_iou": best_iou,
                }
                save_path = os.path.join(
                    writer.file_writer.get_logdir(),
                    "{}_{}_best_model.pkl".format("pspnet", "kitti"),
                )
                torch.save(state, save_path)

        if (i + 1) == TRAIN_ITERS:
            flag = False
            break

cd = 0
import scipy.misc as m

# Just need to do this one time
# caffemodel_dir_path = "PATH_TO_PSPNET_DIR/evaluation/model"
# psp.load_pretrained_model(
#     model_path=os.path.join(caffemodel_dir_path, "pspnet101_cityscapes.caffemodel")
# )

# psp.float()
# psp.cuda(cd)
# psp.eval()

dataset_root_dir = "data_road"
# dst = cl(root=dataset_root_dir)
test_dataset = KITTI_Test_Dataset(DATA_DIR, "image", IMAGE_SIZE, is_transform=True, augmentations=None)


img = cv2.imread(
    os.path.join(
        dataset_root_dir,
        "testing_2", "image_2", "*.png"
    )
)
cv2.imwrite("cropped.png", img)
orig_size = img.shape[:-1]
img = img.transpose(2, 0, 1)
img = img.astype(np.float64)
img -= np.array([123.68, 116.779, 103.939])[:, None, None]
img = np.copy(img[::-1, :, :])
img = torch.from_numpy(img).float()  # convert to torch tensor
img = img.unsqueeze(0)

out = psp.tile_predict(img)
pred = np.argmax(out, axis=1)[0]
decoded = test_dataset.decode_segmap(pred)
cv2.imwrite("cityscapes_sttutgart_tiled.png", decoded)
# m.imsave('cityscapes_sttutgart_tiled.png', pred)

checkpoints_dir_path = "checkpoints"
if not os.path.exists(checkpoints_dir_path):
    os.mkdir(checkpoints_dir_path)
psp = torch.nn.DataParallel(
    psp, device_ids=range(torch.cuda.device_count())
)  # append `module.`
state = {"model_state": psp.state_dict()}
torch.save(state, os.path.join(checkpoints_dir_path, "pspnet_101_cityscapes.pth"))
# torch.save(state, os.path.join(checkpoints_dir_path, "pspnet_50_ade20k.pth"))
# torch.save(state, os.path.join(checkpoints_dir_path, "pspnet_101_pascalvoc.pth"))
print("Output Shape {} \t Input Shape {}".format(out.shape, img.shape))