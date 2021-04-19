from __future__ import print_function, division


import skimage as sk
import skimage.io as skio
from scipy.signal import *
import cv2
import scipy
import torch.nn as nn
import imgaug

import imageio
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import os
import torch
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
from skimage.color import rgb2gray
import pandas as pd
from PIL import Image
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# GLOBAL PARAMETERS

meta = torch.load('mean-std.pt')
res = 300


class SkinData(Dataset):
    def __init__(self, root_dir, data, transform=None,  mode='train'):
        self.root_dir = root_dir
        self.data = pd.read_csv(data)
        self.transform = transform
        self.mode = mode

    def __len__(self):
        self.length = len(self.data)
        return self.length

    def __getitem__(self,  idx):
        """
        Obtains an 'image' and 'target' as a tuple
        ##INPUT##
        idx: (int) item id

        ##OUTPUT##
        image: (tensor) image after transformations
        target: (dictionary) contains the targets
            - 'bbox': (list) [xmin, ymin, xmax, ymax]
            - 'labels': (tensor) (7,) One Hot Encoded Vector for the Disease Diagnoses
            - 'area': (float/int) area of bounding box
            - 'id': idx
        """
        datapoint = self.data.iloc[idx]
        image = Image.open(datapoint['path'])
        target = {}
        minx = datapoint['x']
        miny = datapoint['y']
        w = datapoint['w']
        h = datapoint['h']
        maxx = minx + w
        maxy = miny + h

        target['area'] = w * h
        target['labels'] = torch.tensor(datapoint[-7:])
        target['bbox'] = torch.tensor([minx, miny, maxx, maxy])
        target['idx'] = idx

        if self.transform is not None:
            image, target = self.transform((np.array(image), target))

        return image, target


def Resizing(resize=(300, 300)):
    def Resize(it):
        """
        Resizes images according to the resize dimensions
        image: (array) Image
        resize: (tuple of integers) New dimensions
        target: (Dictionary) Containing BBox, Area, Labels and Index

        Returns normalized_image, target
        """
        image, target = it
        resize = res
        new_target = target.copy()
        bbs = BoundingBoxesOnImage([BoundingBox(x1=new_target['bbox'][0].item(), y1=new_target['bbox'][1].item(
        ), x2=new_target['bbox'][2].item(), y2=new_target['bbox'][3].item())], image.shape)
        img = ia.imresize_single_image(np.array(image), resize)
        new_bbs = bbs.on(img)
        bbox = [new_bbs[0].x1, new_bbs[0].y1, new_bbs[0].x2, new_bbs[0].y2]
        new_target['bbox'] = bbox
        new_target['area'] = (new_bbs[0].x2 - new_bbs[0].x1) * \
            (new_bbs[0].y2 - new_bbs[0].y1)
        return img, new_target
    return Resize


def ToTensor(it):
    """
    Wrapper for converting image to tensor

    Returns tensor_image, target
    """
    image, target = it
    return transforms.ToTensor()(image), target


def Normalizer(meta):
    def normalize(it):
        """
        Normalizes images according to the meta['mean'] and meta['std']
        image: (Tensor) Image
        target: (Dictionary) Containing BBox, Area, Labels and Index

        Returns normalized_image, target
        """
        image, target = it
        function = torchvision.transforms.Normalize(
            mean=meta['mean'],
            std=meta['std'],
        )
        img = function(image)
        return img, target
    return normalize
