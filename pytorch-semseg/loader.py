import os
import cv2
import random
import numpy as np
from glob import glob

import torch
from torch.utils.data import Dataset

from augmentations import Rotation, Scale, Translate, Flip, White_Noise, Gray, Brightness, Contrast, Color, Equalization, Shapness, Power_Law

def process_gt_image(gt_image):
    road_color = np.array([255, 0, 255])
    gt_rd = np.all(gt_image == road_color, axis=2)
    gt_rd = gt_rd.reshape(gt_rd.shape[0], gt_rd.shape[1], 1)

    gt_image = np.concatenate((np.invert(gt_rd), gt_rd), axis=2)

    gt_bg_temp1 = (gt_image[:,:,0] * 255).astype('uint8')

    gt_bg_temp2 = (gt_image[:,:,1] * 255).astype('uint8')

    return gt_image

class Train_DataSet(Dataset):
    def __init__(self, data_folder, data_type, img_size):
        self.data_folder = data_folder
        self.data_types = {"image": "image_2", "lidar": "projection"}
        self.img_size = img_size
        self.image_dir = glob(os.path.join(self.data_folder, "training", self.data_types[data_type], "*.png"))
        self.label_dir = glob(os.path.join(self.data_folder, "training", "gt_image_2", "*_road_*.png"))

    def __getitem__(self, item):
        img = cv2.imread(self.image_dir[item])
        img = np.array(img, dtype=np.uint8)

        lbl = cv2.imread(self.label_dir[item])
        lbl = np.array(lbl, dtype=np.uint8)

        # Step 1
        # Rotation -10~10
        img, lbl = Rotation(img, lbl, 5)
        # Scale 0.9~1.3
        img, lbl = Scale(img, lbl)
        # Translate -50~50
        img, lbl = Translate(img, lbl, 25, 25)
        # Flip
        img, lbl = Flip(img, lbl, 0.5)

        # Step2
        selection_num = random.randint(0, 7)

        if selection_num == 0:
            img = White_Noise(img)
        elif selection_num == 1:
            img = Gray(img)
        elif selection_num == 2:
            img = Brightness(img)
        elif selection_num == 3:
            img = Contrast(img)
        elif selection_num == 4:
            img = Color(img)
        elif selection_num == 5:
            img = Equalization(img)
        elif selection_num == 6:
            img = Shapness(img)
        elif selection_num == 7:
            img = Power_Law(img)

        img = cv2.resize(img, self.img_size)
        lbl = cv2.resize(lbl, self.img_size)

        lbl = process_gt_image(lbl)

        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        lbl = lbl.transpose(2, 0, 1)

        img = torch.tensor(img).float()
        lbl = torch.tensor(lbl).float()

        return img, lbl

    def __len__(self):
        return len(self.image_dir)
        # return 10


class Valid_DataSet(Dataset):
    def __init__(self, data_folder, data_type, img_size):
        self.data_folder = data_folder
        self.data_types = {"image": "image_2", "lidar": "projection"}
        self.img_size = img_size
        self.image_dir = glob(os.path.join(self.data_folder, "validating", self.data_types[data_type], "*.png"))
        self.label_dir = glob(os.path.join(self.data_folder, "validating", "gt_image_2", "*_road_*.png"))

    def __getitem__(self, item):
        img = cv2.imread(self.image_dir[item])
        img = np.array(img, dtype=np.uint8)

        lbl = cv2.imread(self.label_dir[item])
        lbl = np.array(lbl, dtype=np.uint8)

        img = cv2.resize(img, self.img_size)
        lbl = cv2.resize(lbl, self.img_size)

        lbl = process_gt_image(lbl)

        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        lbl = lbl.transpose(2, 0, 1)

        img = torch.tensor(img).float()
        lbl = torch.tensor(lbl).float()

        return img, lbl

    def __len__(self):
        return len(self.image_dir)


class Test_DataSet(Dataset):
    def __init__(self, data_folder, data_type, img_size):
        self.data_folder = data_folder
        self.data_types = {"image": "image_2", "lidar": "projection"}
        self.img_size = img_size
        self.image_dir = glob(os.path.join(self.data_folder, "testing_2", self.data_types[data_type], "*.png"))
        self.label_dir = glob(os.path.join(self.data_folder, "testing_2", "gt_image_2", "*_road_*.png"))

    def __getitem__(self, item):
        img = cv2.imread(self.image_dir[item])
        img = np.array(img, dtype=np.uint8)

        lbl = cv2.imread(self.label_dir[item])
        lbl = np.array(lbl, dtype=np.uint8)

        img = cv2.resize(img, self.img_size)
        lbl = cv2.resize(lbl, self.img_size)

        lbl = process_gt_image(lbl)


        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        lbl = lbl.transpose(2, 0, 1)

        img = torch.tensor(img).float()
        lbl = torch.tensor(lbl).float()

        return img, lbl

    def __len__(self):
        return len(self.image_dir)
