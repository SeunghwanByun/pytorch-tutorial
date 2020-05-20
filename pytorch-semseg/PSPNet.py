import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import conv1x1, conv3x3

import ResNet

class BottleNeck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.stride = stride
        self.layer = nn.Sequential(conv1x1(in_channels, mid_channels, self.stride, 0),
                                   conv3x3(mid_channels, mid_channels, 1, 1),
                                   conv1x1(mid_channels, out_channels, 1, 0))

        self.downsample = nn.MaxPool2d(2, stride)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=self.stride, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.conv1(x)

        x = self.layer(x)

        x += residual
        x = self.relu(x)
        return x


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, output_dim, bins):
        super(PyramidPoolingModule, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, output_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

class PSPNet(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), classes=2, dropout=0.1, zoom_factor=8, use_ppm=True,
                 is_training=True, criterion=nn.BCELoss()): #criterion=nn.CrossEntropyLoss(ignore_index=255)):
        super(PSPNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]

        self.criterion = criterion
        self.classes = classes
        self.use_ppm = use_ppm
        self.zoom_factor = zoom_factor
        self.is_training = is_training

        if layers == 50:
            resnet = ResNet.resnet50()
        elif layers == 101:
            resnet = ResNet.resnet101()
        elif layers == 152:
            resnet = ResNet.resnet152()

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048
        if self.use_ppm:
            self.ppm = PyramidPoolingModule(fea_dim, int(fea_dim/len(bins)), bins)
            fea_dim *= 2
        self.cls = nn.Sequential(nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout2d(p=dropout),
                                 nn.Conv2d(512, self.classes, kernel_size=1))
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, self.classes, kernel_size=1)
            )
        # self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
        #                            nn.BatchNorm2d(64), nn.ReLU())
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.layer1 = nn.Sequential(BottleNeck(in_channels=64, mid_channels=64, out_channels=256),
        #                             BottleNeck(in_channels=256, mid_channels=64, out_channels=256),
        #                             BottleNeck(in_channels=256, mid_channels=64, out_channels=256))
        # self.layer2 = nn.Sequential(
        #     BottleNeck(in_channels=256, mid_channels=128, out_channels=512, stride=2),
        #     BottleNeck(in_channels=512, mid_channels=128, out_channels=512),
        #     BottleNeck(in_channels=512, mid_channels=128, out_channels=512),
        #     BottleNeck(in_channels=512, mid_channels=128, out_channels=512))
        # self.layer3 = nn.Sequential(
        #     BottleNeck(in_channels=512, mid_channels=256, out_channels=1024, stride=2),
        #     BottleNeck(in_channels=1024, mid_channels=256, out_channels=1024),
        #     BottleNeck(in_channels=1024, mid_channels=256, out_channels=1024),
        #     BottleNeck(in_channels=1024, mid_channels=256, out_channels=1024),
        #     BottleNeck(in_channels=1024, mid_channels=256, out_channels=1024),
        #     BottleNeck(in_channels=1024, mid_channels=256, out_channels=1024))
        # self.layer4 = nn.Sequential(
        #     BottleNeck(in_channels=1024, mid_channels=512, out_channels=2048, stride=2),
        #     BottleNeck(in_channels=2048, mid_channels=512, out_channels=2048),
        #     BottleNeck(in_channels=2048, mid_channels=512, out_channels=2048))
        #
        # self.encoder_output = nn.Conv2d(2048, 1024, 1, 1, 0)
        #
        # self.ppm = PyramidPoolingModule(1024, int(1024 / len(bins)), bins)
        # self.output = nn.Sequential(nn.Conv2d(1024 * 2, 128, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(128),
        #                             nn.ReLU(), nn.Dropout(p=dropout), nn.Conv2d(128, num_class, kernel_size=1, stride=1, padding=0))
        # if self.is_training is True:
        #     self.aux = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
        #                              nn.BatchNorm2d(64),
        #                              nn.ReLU(),
        #                              nn.Dropout(p=dropout),
        #                              nn.Conv2d(64, self.num_class, kernel_size=1))
        # self.conv_aux = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        # self.softmax = nn.Softmax

    def forward(self, x, y=None):
        # Check Training Images
        cv2.imshow("img", x.cpu().numpy()[0].transpose(1, 2, 0).astype('uint8'))
        cv2.waitKey(30)

        x_size = x.size()
        assert (x_size[2] - 1) % 8 == 7 and (x_size[3] - 1) % 8 == 7
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_temp = self.layer3(x)
        x = self.layer4(x_temp)

        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        x = F.softmax(x, dim=1)

        if self.training:
            aux = self.aux(x_temp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            aux = F.softmax(aux, dim=1)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)

            return x, main_loss, aux_loss # x.max(1)[1]
        else:
            return x

        #
        # h = x_size[2]
        # w = x_size[3]
        #
        # x = self.conv1(x)
        # x = self.maxpool(x)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x_temp = self.layer3(x)
        # x = self.layer4(x_temp)
        # x = self.encoder_output(x)
        # x = self.ppm(x)
        # x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        # x = self.output(x)
        # x = F.softmax(x, dim=1)
        #
        # if self.training:
        #     aux = self.conv_aux(x_temp)
        #     aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
        #     aux = self.aux(aux)
        #     aux = F.softmax(aux, dim=1)
        #
        #     return x, aux
        # else:
        #     return x