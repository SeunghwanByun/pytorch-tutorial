import os
import re
import cv2
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset

class KITTI_Train_Dataset(Dataset):
    """ KITTI Road Dataset """
    def __init__(self, data_folder, data_type, img_size, is_transform=False, augmentations=None, img_norm=True, test_mode=False):
        """

        :param image_dir:
        :param label_dir:
        :param transform:
        """
        data_types = {"image": "image_2", "lidar": "projection"}
        self.image_dir = glob(os.path.join(data_folder, "training", data_types[data_type], "*.png"))
        self.label_dir = glob(os.path.join(data_folder, "training", "gt_image_2", "*_road_*.png"))
        # self.label_dir = {
        #     re.sub(r'_(lane|road)_', '_', os.path.basename(path)):path
        #     for path in glob(os.path.join(data_folder, "training", "gt_image_2", "*_road_*.png"))}
        self.img_size = img_size
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.test_mode = test_mode
        self.img_norm = img_norm
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        # self.mean = np.array([0.0, 0.0, 0.0])
        self.n_classes = 2


    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        img = cv2.imread(self.image_dir[item])
        img = np.array(img, dtype=np.uint8)

        lbl = cv2.imread(self.label_dir[item])
        lbl = np.array(lbl, dtype=np.uint8)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        img = cv2.resize(img, (self.img_size[0], self.img_size[1])) # uint8 with BGR mode
        img = img.astype(np.float64)
        img -= self.mean

        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def decode_segmap(self, temp, plot=False):
        Road = [0, 0, 255]
        NonRoad = [255, 0, 255]
        Unlabeled = [0, 0, 0]

        label_colours = np.array(
            [
                Road,
                NonRoad,
                Unlabeled
            ]
        )
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()


        print(r.shape)
        print(g.shape)
        print(b.shape)
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[1, 0]
            g[temp == l] = label_colours[1, 1]
            b[temp == l] = label_colours[1, 2]

            rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
            print("rgb", rgb.shape)
            rgb[:, :, 0] = r / 255.0
            rgb[:, :, 1] = g / 255.0
            rgb[:, :, 2] = b / 255.0

            return rgb

class KITTI_Validation_Dataset(Dataset):
    """ KITTI Road Dataset """
    def __init__(self, data_folder, data_type, img_size, is_transform=False, augmentations=None, img_norm=True, test_mode=False):
        """

        :param image_dir:
        :param label_dir:
        :param transform:
        """
        data_types = {"image": "image_2", "lidar": "projection"}
        self.image_dir = glob(os.path.join(data_folder, "validating", data_types[data_type], "*.png"))
        self.label_dir = glob(os.path.join(data_folder, "validating", "gt_image_2", "*_road_*.png"))
        # self.label_dir = {
        #     re.sub(r'_(lane|road)_', '_', os.path.basename(path)):path
        #     for path in glob(os.path.join(data_folder, "training", "gt_image_2", "*_road_*.png"))}
        self.img_size = img_size
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.test_mode = test_mode
        self.img_norm = img_norm
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        # self.mean = np.array([0.0, 0.0, 0.0])
        self.n_classes = 2


    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        img = cv2.imread(self.image_dir[item])
        img = np.array(img, dtype=np.uint8)

        lbl = cv2.imread(self.label_dir[item])
        lbl = np.array(lbl, dtype=np.uint8)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        img = cv2.resize(img, (self.img_size[0], self.img_size[1])) # uint8 with BGR mode
        img = img.astype(np.float64)
        img -= self.mean

        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def decode_segmap(self, temp, plot=False):
        Road = [0, 0, 255]
        NonRoad = [255, 0, 255]
        Unlabeled = [0, 0, 0]

        label_colours = np.array(
            [
                Road,
                NonRoad,
                Unlabeled
            ]
        )
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()


        print(r.shape)
        print(g.shape)
        print(b.shape)
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[1, 0]
            g[temp == l] = label_colours[1, 1]
            b[temp == l] = label_colours[1, 2]

            rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
            print("rgb", rgb.shape)
            rgb[:, :, 0] = r / 255.0
            rgb[:, :, 1] = g / 255.0
            rgb[:, :, 2] = b / 255.0

            return rgb


class KITTI_Test_Dataset(Dataset):
    """ KITTI Road Dataset """
    def __init__(self, data_folder, data_type, img_size, is_transform=False, augmentations=None, img_norm=True, test_mode=False):
        """

        :param image_dir:
        :param label_dir:
        :param transform:
        """
        data_types = {"image": "image_2", "lidar": "projection"}
        self.image_dir = glob(os.path.join(data_folder, "testing_2", data_types[data_type], "*.png"))
        self.label_dir = glob(os.path.join(data_folder, "testing_2", "gt_image_2", "*_road_*.png"))
        # self.label_dir = {
        #     re.sub(r'_(lane|road)_', '_', os.path.basename(path)):path
        #     for path in glob(os.path.join(data_folder, "training", "gt_image_2", "*_road_*.png"))}
        self.img_size = img_size
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.test_mode = test_mode
        self.img_norm = img_norm
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        # self.mean = np.array([0.0, 0.0, 0.0])
        self.n_classes = 2


    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        img = cv2.imread(self.image_dir[item])
        img = np.array(img, dtype=np.uint8)

        lbl = cv2.imread(self.label_dir[item])
        lbl = np.array(lbl, dtype=np.uint8)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        img = cv2.resize(img, (self.img_size[0], self.img_size[1])) # uint8 with BGR mode
        img = img.astype(np.float64)
        img -= self.mean

        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def decode_segmap(self, temp, plot=False):
        Road = [0, 0, 255]
        NonRoad = [255, 0, 255]
        Unlabeled = [0, 0, 0]

        label_colours = np.array(
            [
                Road,
                NonRoad,
                Unlabeled
            ]
        )
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()


        print(r.shape)
        print(g.shape)
        print(b.shape)
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[1, 0]
            g[temp == l] = label_colours[1, 1]
            b[temp == l] = label_colours[1, 2]

            rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
            print("rgb", rgb.shape)
            rgb[:, :, 0] = r / 255.0
            rgb[:, :, 1] = g / 255.0
            rgb[:, :, 2] = b / 255.0

            return rgb