from torch.utils.data import DataLoader, Dataset, sampler
import os
import cv2
import numpy as np
import pandas as pd
import pydicom
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from utils.RleFunction import run_length_decode
from albumentations.torch import ToTensor
from sklearn.model_selection import StratifiedKFold



class SIIMDataset(Dataset):
    def __init__(self, df, size, mean, std, phase):
        self.df = df
        # self.root = data_folder
        self.size = size
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, size, mean, std)
        self.gb = self.df.groupby('ImageId')
        self.fnames = list(self.gb.groups.keys())
        # t1=0
        # t2=0
        # print(self.df.shape)
        # for g in self.gb.groups.keys():
        #     a=self.gb.groups[g]
        #     print(len(a))
        #     if len(a)==1:
        #         t1+=1
        #     if len(a)==2:
        #         t2+=1
        # print(t2*2+t1)
        #     # if a.shap
        #     # if self.gb.groups[g][0]!=self.gb.groups[g][1]:
        #     #     # print('hello',self.gb.groups[g][0],self.gb.groups[g][1])

    def __getitem__(self, idx):
        # imagepath=self.df.

        image_id = self.fnames[idx]
        df = self.gb.get_group(image_id)
        imagepath=df['path']
        print(imagepath)
        annotations = df['EncodedPixels'].tolist()
        # TODO:
        # image_path = os.path.join(self.root, image_id + ".png")
        image=pydicom.dcmread(df['path']).pixel_array

        # image = cv2.imread(image_path)
        mask = np.zeros([1024, 1024])
        if annotations[0] != '-1':
            for rle in annotations:
                mask += run_length_decode(rle)
        mask = (mask >= 1).astype('float32')  # for overlap cases
        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        return image, mask

    def __len__(self):
        return len(self.df)


def get_transforms(phase, size, mean, std):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                #                 HorizontalFlip(),
                ShiftScaleRotate(
                    shift_limit=0,  # no resizing
                    scale_limit=0.1,
                    rotate_limit=10,  # rotate
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT
                ),
                #                 GaussNoise(),
            ]
        )
    list_transforms.extend(
        [
            Normalize(mean=mean, std=std, p=1),
            Resize(size, size),
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
    df = df.drop_duplicates('ImageId')
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
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    df = train_df if phase == "train" else val_df
    # NOTE: total_folds=5 -> train/val : 80%/20%

    image_dataset = SIIMDataset(df, size, mean, std, phase)

    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return dataloader