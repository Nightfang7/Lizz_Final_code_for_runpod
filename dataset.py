import torch
# import tensorflow as tf
# from tensorflow.python.platform import build_info as tf_build_info

import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
import shutil
# import seaborn as sns
import pandas as pd
import math
import re
# import cv2
from collections import Counter
import time
import statistics
from itertools import combinations


class SiameseNetworkDataset_1(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self, index):
        img0_tuple = self.imageFolderDataset.imgs[index]

        # We need to approximately 50% of images to be in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # Look untill the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:

            while True:
                # Look untill a different class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        img0 = img0.convert('RGB')
        img1 = img1.convert('RGB')

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32)), img0_tuple[1], img1_tuple[1], img0_tuple[0], img1_tuple[0]

    def __len__(self):
        return len(self.imageFolderDataset.imgs)
    

class DBSCANDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self, index):
        img_tuple = self.imageFolderDataset.imgs[index]

        img = Image.open(img_tuple[0])

        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # Fixed: Use os.path for cross-platform path handling
        img_path = img_tuple[0]
        # Get the parent directory name (class name) in a cross-platform way
        class_name = os.path.basename(os.path.dirname(img_path))
        
        return img, class_name, img_path
    def __len__(self):
        return len(self.imageFolderDataset.imgs)
    

class AutoEncoderDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self, index):
        img_tuple = self.imageFolderDataset.imgs[index]
        img = Image.open(img_tuple[0])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, img_tuple[1], img_tuple[0]
    def __len__(self):
        return len(self.imageFolderDataset.imgs)
    

class PrototypeDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self, index):
        img_tuple = random.choice(self.imageFolderDataset.imgs)

        img = Image.open(img_tuple[0])

        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, img_tuple[1], img_tuple[0]
    def __len__(self):
        return len(self.imageFolderDataset.imgs)