import cv2
import numpy as np

import torch
import torch.nn as nn

import visdom

def conv1x1(in_channels, out_channels, stride, padding):
    model = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding), nn.ReLU())
    return model

def conv3x3(in_channels, out_channels, stride, padding):
    model = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding), nn.ReLU())
    return model

def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr

def acc_check(net, device, test_set, test_set_loader, image_size, epoch, save=1):
    correct = 0
    total = 0

    net.eval()

    with torch.no_grad():
        for i, data in enumerate(test_set_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)

            name = test_set.image_dir[i].split("\\")[-1]
            output_softmax = outputs.cpu().numpy()[0].transpose(1, 2, 0)
            gt = labels.cpu().numpy()[0].transpose(1, 2, 0)

            total += labels.size()[2] * labels.size()[3]
            correct += np.all((output_softmax > 0.8) == gt, axis=2).sum()

            output_img_train = np.squeeze(output_softmax)[:, :, 1].reshape(image_size[1], image_size[0])
            segmentation_r_train = (output_img_train > 0.8).reshape(image_size[1], image_size[0], 1)
            mask_train = np.dot(segmentation_r_train, np.array([[0, 255, 0]]))

            cv2.imwrite("./data_road/output_valid/{0}_{1}".format(epoch, name), mask_train)

    acc = (100 * correct / total)

    print("Accuracy of the network on the %d test images: %d %% dimension" % (test_set.__len__(), acc))
    
    if save:
        torch.save(net.state_dict(), "./model/model_epoch_{}_acc_{}.pth".format(epoch, int(acc)))

    net.train()

    return acc
