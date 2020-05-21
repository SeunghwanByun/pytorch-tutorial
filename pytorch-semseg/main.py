import cv2
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from torchsummary import summary

from torch.utils.tensorboard import SummaryWriter

from loader import Train_DataSet, Valid_DataSet, Test_DataSet
from PSPNet import PSPNet
from utils import poly_learning_rate, value_tracker, acc_check

KITTI_IMAGE_SIZE = (576, 160)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)
if device == 'cuda':
    torch.cuda.manual_seed_all(1377)

def main():
    dataset = Train_DataSet("data_road", "image", KITTI_IMAGE_SIZE)
    dataset_loader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True, num_workers=0)

    dataset_valid = Valid_DataSet("data_road", "image", KITTI_IMAGE_SIZE)
    dataset_loader_valid = DataLoader(dataset_valid, batch_size=1, num_workers=0)

    dataset_test = Test_DataSet("data_road", "image", KITTI_IMAGE_SIZE)
    dataset_loader_test = DataLoader(dataset_test, batch_size=1, num_workers=0)


    model = PSPNet().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    
    writer = SummaryWriter()
    
    epochs = 300
    epoch = 0
    while True:
        running_loss = 0.0

        for i, data in enumerate(dataset_loader, 0):
            max_iter = epochs * len(dataset_loader)
            current_iter = epoch * len(dataset_loader) + i + 1
            current_lr = poly_learning_rate(0.01, current_iter, max_iter)

            # for index in range(0, 5):
            optimizer.param_groups[0]['lr'] = current_lr

            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output, main_loss, aux_loss = model(images, labels)
            
            loss = main_loss + 0.4 * aux_loss
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:
                writer.add_scalar("Training Loss", running_loss / 10, epoch * len(dataset_loader) + i)
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 10))

                running_loss = 0.0
                
            if i % 30 == 0:
                grid = torchvision.utils.make_grid(output)
                writer.add_image("Output Image", grid, 0)

        # Check Accuracy
        acc = acc_check(model, dataset_valid, dataset_loader_valid, epoch, save=1)
        writer.add_scalar("Training Accuracy", acc, epoch)
        
        epoch += 1
        
        if acc > 96 or epoch > epochs:
            print("Finished Training..!")
            break
    
    writer.close()

    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(dataset_loader_test):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            name = dataset_test.image_dir[i].split("\\")[-1]

            output_softmax = outputs.detach().cpu().numpy()[0].transpose(1, 2, 0)
            gt = labels.detach().cpu().numpy()[0].transpose(1, 2, 0)

            output_img = np.squeeze(output_softmax)[:, :, 1].reshape(KITTI_IMAGE_SIZE[1], KITTI_IMAGE_SIZE[0])
            segmentation_r = (output_img > 0.8).reshape(KITTI_IMAGE_SIZE[1], KITTI_IMAGE_SIZE[0], 1)
            mask = np.dot(segmentation_r, np.array([[0, 255, 0]]))
            
            cv2.imwrite("data_road/output/{0}".format(name), mask)

            total += labels.size()[2] * labels.size()[3]
            correct += np.all((output_softmax > 0.8) == gt, axis=2).sum()

            correct += (predicted == labels).sum().item()

    print("Accuracy of the network on the %d test images: %d %%" % (dataset_test.__len__(), 100 * correct / total))

    summary(model, input_size=(3, 160, 576))

if __name__ == '__main__':
    main()
