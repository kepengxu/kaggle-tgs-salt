from torch.utils.data import DataLoader, Dataset, sampler
import os
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import ImageEnhance,Image

import random
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise,Crop)
from utils.RleFunction import run_length_decode
from albumentations.torch import ToTensor
from sklearn.model_selection import StratifiedKFold
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from random import randint
import skimage

from skimage.morphology import disk
import skimage.filters.rank as sfr



from skimage import data, exposure
# image =data.moon()
# result=exposure.is_low_contrast(image)
# print(result)

def enhance(image,factor=1.0):
    if factor==None:
        factor=random.random()*0.8+0.4
    enchence =ImageEnhance.Contrast(image)
    image =enchence.enhance(factor)
    return image

def cropping(image, crop_size, dim1, dim2):
    """crop the image and pad it to in_size
    Args :
        images : numpy array of images
        crop_size(int) : size of cropped image
        dim1(int) : vertical location of crop
        dim2(int) : horizontal location of crop
    Return :
        cropped_img: numpy array of cropped image
    """
    cropped_img = image[dim1:dim1+crop_size, dim2:dim2+crop_size]
    return cropped_img
def add_elastic_transform(image,cropsize, alpha, sigma,pad_size, seed=None):
    """
    Args:
        image : numpy array of image
        alpha : α is a scaling factor
        sigma :  σ is an elasticity coefficient
        random_state = random integer
        Return :
        image : elastically transformed numpy array of image
    """
    image_size = int(image.shape[0])
    image = np.pad(image, pad_size, mode="symmetric")
    if seed is None:
        seed = randint(1, 100)
        random_state = np.random.RandomState(seed)
    else:
        random_state = np.random.RandomState(seed)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    im=map_coordinates(image, indices, order=1).reshape(shape)
    return cropping(im, cropsize, pad_size, pad_size), seed

def ceil_floor_image(image):
    """
    Args:
        image : numpy array of image in datatype int16
    Return :
        image : numpy array of image in datatype uint8 with ceilling(maximum 255) and flooring(minimum 0)
    """
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype("uint8")
    return image

def add_gaussian_noise(image, mean=0, std=1):
    """
    Args:
        image : numpy array of image
        mean : pixel mean of image
        standard deviation : pixel standard deviation of image
    Return :
        image : numpy array of image with gaussian noise added
    """
    gaus_noise = np.random.normal(mean, std, image.shape)
    image = image.astype("int16")
    noise_img = image + gaus_noise
    image = ceil_floor_image(image)
    return noise_img

def limitedEqualize(img_array, limit = 20.0):
    clahe = cv2.createCLAHE(clipLimit = limit, tileGridSize = (8,8))
    return clahe.apply(img_array)

class SIIMDataset(Dataset):
    def __init__(self, df, size, mean, std, phase,mix=True,et=False,adnoise=True,hist=True):
        self.df = df

#         self.df=self.df.query ( " rle_count >= 1",  engine='python').sort_values(by="rle_count", ascending=False)
#         self.df=self.df.query ( " ImageId=='1.2.276.0.7230010.3.1.4.8323329.13056.1517875243.326353' ",  engine='python').sort_values(by="rle_count", ascending=False)
        # self.root = data_folder
        self.size = size
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, size, mean, std)
        self.gb = self.df.groupby('ImageId')
        self.fnames = list(self.gb.groups.keys())
        self.mix=mix
        self.adnoise=adnoise
        self.et=et
        self.hist=hist
        self.tc=1024-self.size

    def __getitem__(self, idx):
        image_id = self.fnames[idx]
        df = self.gb.get_group(image_id)
        imagepath=df['path'].iloc[0]

        annotations = df['EncodedPixels'].tolist()

        image=pydicom.dcmread(imagepath).pixel_array


        image=image.astype('float32')

        mask = np.zeros([1024, 1024])

        if annotations[0] != '-1':

            if self.mix:
                for rle in annotations:
                    mask += run_length_decode(rle)
            else:

                t=random.randint(0,len(annotations)-1)
                mask+= run_length_decode(annotations[t])


        mask = (mask >= 1).astype('float32')  # for overlap cases

        # t3=cv2.resize(image,(512,512)).astype('uint8').copy()
        # t4=cv2.resize(np.array(mask*255,np.uint8),(512,512)).copy()
        # th=np.concatenate((t3,t4),axis=1)
        if self.hist:
            image = skimage.exposure.equalize_adapthist(image/255.0, 20)*255.0
        if self.et and self.phase=='train':
            probablity=random.random()
            if probablity>0.7:
                image,seed=add_elastic_transform(image,1024,75,10,20)
                mask,_=add_elastic_transform(mask,1024,75,10,20,seed)

        if self.phase=='train':
            t=random.randint(0,1)
            if t==1:
                image=np.flip(image,t)
                mask=np.flip(mask,t)
            if self.adnoise:
                gaus_sd, gaus_mean = randint(0, 20)*0.05, 0
                image = add_gaussian_noise(image, gaus_mean, gaus_sd)

        # t1=cv2.resize(image,(512,512)).astype('uint8')
        # t2=cv2.resize(np.array(mask*255,np.uint8),(512,512))
        # tz=np.concatenate((t1,t2),axis=1)
        # cv2.imshow('image',np.concatenate((th,tz),axis=0))
        # cv2.waitKey(10)
        # mask=mask.astype('int')
        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        y1=random.randint(0,self.tc)
        x1=random.randint(0,self.tc)
        image=image[y1:y1+self.size,x1:x1+self.size]
        mask = mask[:,y1:y1 + self.size, x1:x1 + self.size]
        image=torch.unsqueeze(image,0)
        mask=mask.int()
        # print(image.max(), image.min(), mask.max(), mask.min(),image.dtype,mask.dtype,image.shape,mask.shape)

        return image, mask

    def __len__(self):
        return len(self.fnames)


def get_transforms(phase, size, mean, std):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                #                 HorizontalFlip(),
                ShiftScaleRotate(
                    shift_limit=0,  # no resizing
                    scale_limit=0.1,
                    rotate_limit=20,  # rotate
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT
                ),
                #                 GaussNoise(),
            ]
        )
    list_transforms.extend(
        [
            Normalize(mean=mean, std=std, p=1),
            # Resize(size, size),
            ToTensor(),
        ]
    )

    list_trfms = Compose(list_transforms)
    return list_trfms


def provider(
        fold,
        total_folds,
        df_path,
        phase,
        size,
        mean=None,
        std=None,
        batch_size=8,
        num_workers=4,
):
    df = pd.read_csv(df_path)

    df_with_mask = df[df["EncodedPixels"] != " -1"]
    df_with_mask['has_mask'] = 1
    df_without_mask = df[df["EncodedPixels"] != " -1"]
    df_without_mask['has_mask'] = 0
    df_without_mask_sampled = df_without_mask.sample(len(df_with_mask))
    df = pd.concat([df_with_mask, df_without_mask_sampled])
    # NOTE: equal number of positive and negative cases are chosen.

    kfold = StratifiedKFold(total_folds, shuffle=True, random_state=69)
    train_idx, val_idx = list(kfold.split(
        df["ImageId"], df["has_mask"]))[fold]
    train_df, val_df = df.iloc[train_idx],df.iloc[val_idx]
    df = train_df if phase == "train" else val_df
    # NOTE: total_folds=5 -> train/val : 80%/20%
    # df = df.query(" rle_count >= 1 and ImageId=='1.2.276.0.7230010.3.1.4.8323329.5073.1517875186.287848"
    #               ""
    #               ""
    #               "' ",#and ImageId=='1.2.276.0.7230010.3.1.4.8323329.13026.1517875243.170116'
    #               engine='python').sort_values(by="rle_count", ascending=False)
    # df = df.drop_duplicates('EncodedPixels')
    image_dataset = SIIMDataset(df, size, mean, std, phase)

    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return dataloader


def guideFilter(I, p, winSize, eps):
    # I的均值平滑
    mean_I = cv2.blur(I, winSize)

    # p的均值平滑
    mean_p = cv2.blur(p, winSize)

    # I*I和I*p的均值平滑
    mean_II = cv2.blur(I * I, winSize)

    mean_Ip = cv2.blur(I * p, winSize)

    # 方差
    var_I = mean_II - mean_I * mean_I  # 方差公式

    # 协方差
    cov_Ip = mean_Ip - mean_I * mean_p

    a = cov_Ip *(var_I + eps)
    b = mean_p - a * mean_I

    # 对a、b进行均值平滑
    mean_a = cv2.blur(a, winSize)
    mean_b = cv2.blur(b, winSize)

    q = mean_a * I -mean_b

    return q

