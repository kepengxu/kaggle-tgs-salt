# import os
# import cv2
# import pdb
# import time
# import warnings
# import random
# import numpy as np
# import pandas as pd
# from tqdm import tqdm_notebook as tqdm
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from sklearn.model_selection import StratifiedKFold
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# import torch.optim as optim
# import torch.backends.cudnn as cudnn
# from torch.utils.data import DataLoader, Dataset, sampler
# from matplotlib import pyplot as plt
# from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
# from albumentations.torch import ToTensor
# # import segmentation_models_pytorch as smp
# from utils.RleFunction import run_length_decode
# warnings.filterwarnings("ignore")
# class SIIMDataset(Dataset):
#     def __init__(self, df, data_folder, size, mean, std, phase):
#         self.df = df
#         self.root = data_folder
#         self.size = size
#         self.mean = mean
#         self.std = std
#         self.phase = phase
#         self.transforms = get_transforms(phase, size, mean, std)
#         self.gb = self.df.groupby('ImageId')
#         self.fnames = list(self.gb.groups.keys())
#
#     def __getitem__(self, idx):
#         image_id = self.fnames[idx]
#         df = self.gb.get_group(image_id)
#         annotations = df[' EncodedPixels'].tolist()
#         image_path = os.path.join(self.root, image_id + ".png")
#         image = cv2.imread(image_path)
#         mask = np.zeros([1024, 1024])
#         if annotations[0] != '-1':
#             for rle in annotations:
#                 mask += run_length_decode(rle)
#         mask = (mask >= 1).astype('float32')  # for overlap cases
#         augmented = self.transforms(image=image, mask=mask)
#         image = augmented['image']
#         mask = augmented['mask']
#         return image, mask
#
#     def __len__(self):
#         return len(self.fnames)
#
#
# def get_transforms(phase, size, mean, std):
#     list_transforms = []
#     if phase == "train":
#         list_transforms.extend(
#             [
#                 #                 HorizontalFlip(),
#                 ShiftScaleRotate(
#                     shift_limit=0,  # no resizing
#                     scale_limit=0.1,
#                     rotate_limit=10,  # rotate
#                     p=0.5,
#                     border_mode=cv2.BORDER_CONSTANT
#                 ),
#                 #                 GaussNoise(),
#             ]
#         )
#     list_transforms.extend(
#         [
#             Normalize(mean=mean, std=std, p=1),
#             Resize(size, size),
#             ToTensor(),
#         ]
#     )
#
#     list_trfms = Compose(list_transforms)
#     return list_trfms
#
#
# def provider(
#         fold,
#         total_folds,
#         data_folder,
#         df_path,
#         phase,
#         size,
#         mean=None,
#         std=None,
#         batch_size=8,
#         num_workers=4,
# ):
#     df = pd.read_csv(df_path)
#     df = df.drop_duplicates('ImageId')
#     df_with_mask = df[df[" EncodedPixels"] != " -1"]
#     df_with_mask['has_mask'] = 1
#     df_without_mask = df[df[" EncodedPixels"] != " -1"]
#     df_without_mask['has_mask'] = 0
#     df_without_mask_sampled = df_without_mask.sample(len(df_with_mask))
#     df = pd.concat([df_with_mask, df_without_mask_sampled])
#     # NOTE: equal number of positive and negative cases are chosen.
#
#     kfold = StratifiedKFold(total_folds, shuffle=True, random_state=69)
#     train_idx, val_idx = list(kfold.split(
#         df["ImageId"], df["has_mask"]))[fold]
#     train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
#     df = train_df if phase == "train" else val_df
#     # NOTE: total_folds=5 -> train/val : 80%/20%
#
#     image_dataset = SIIMDataset(df, data_folder, size, mean, std, phase)
#
#     dataloader = DataLoader(
#         image_dataset,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         pin_memory=True,
#         shuffle=True,
#     )
#     return dataloader
# import settings
# data_folder=settings.data_folder
# train_rle_path=settings.train_rle_path
#
#
# dataloader = provider(
#     fold=0,
#     total_folds=5,
#     data_folder=data_folder,
#     df_path=train_rle_path,
#     phase="train",
#     size=512,
#     mean = (0.485, 0.456, 0.406),
#     std = (0.229, 0.224, 0.225),
#     batch_size=16,
#     num_workers=4,
# )
#
# batch = next(iter(dataloader)) # get a batch from the dataloader
# images, masks = batch
# print(images.shape,masks.shape)

# import pydicom
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# path='/home/cooper/PycharmProjects/SIIM/siim-acr-pneumothorax-segmentation/dataset/pneumothorax/dicom-images-test/1.2.276.0.7230010.3.1.2.8323329.580.1517875163.537052/1.2.276.0.7230010.3.1.3.8323329.580.1517875163.537051/1.2.276.0.7230010.3.1.4.8323329.580.1517875163.537053.dcm'
# image=pydicom.dcmread(path)
# ndarray=image.pixel_array
# ndarray=np.array(ndarray,np.uint8)
# plt.imshow(ndarray)
# print(ndarray.shape)
# cv2.imshow('image',ndarray)
# cv2.waitKey(0)
# plt.show()
#
import numpy as np
import pandas as pd
import os
import pydicom
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import settings

# import mask utilities
import sys

# sys.path.insert(0, '../input/siim-acr-pneumothorax-segmentation')
from utils.mask_functions import rle2mask


def extract_dcm_pixel_array(file_path):
    return pydicom.dcmread(file_path).pixel_array


def extract_dcm_metadata(file_path):
    ds = pydicom.dcmread(file_path)
    d = {}
    for elem in ds.iterall():
        if elem.name != 'Pixel Data' and elem.name != "Pixel Spacing":
            d[elem.name.lower().replace(" ", "_").replace("'s", "")] = elem.value
        elif elem.name == "Pixel Spacing":
            d["pixel_spacing_x"] = elem.value[0]
            d["pixel_spacing_y"] = elem.value[1]

    return d


def create_metadataset(df):
    ImageIds = []
    data = []
    all_feats = set()

    for index, row in tqdm(df[["ImageId", "path"]].drop_duplicates().iterrows()):
        path = row["path"]
        ImageId = row["ImageId"]
        feature_dict = extract_dcm_metadata(path)
        data.append(feature_dict)
        ImageIds.append(ImageId)
        feats = set(feature_dict.keys())
        if len(feats - all_feats) > 0:
            all_feats = all_feats.union(feats)

    df_meta = pd.DataFrame(columns=["ImageId"])
    df_meta["ImageId"] = ImageIds

    for feat in sorted(all_feats):
        df_meta[feat] = [d[feat] for d in data]

    df_meta['patient_age'] = df_meta['patient_age'].map(lambda x: int(x))
    return df_meta


DATA_PATH = settings.DATASETDIR
SAMPLE_SUBMISSION = "/home/cooper/PycharmProjects/SIIM/siim-acr-pneumothorax-segmentation/sample_submission.csv"

df_train = pd.DataFrame([(name.replace(".dcm", ""), os.path.join(root, name)) for root, dirs, files in
                         os.walk(DATA_PATH + "/dicom-images-train")
                         for name in files if name.endswith((".dcm"))], columns=['ImageId', 'path'])

df_test = pd.DataFrame([(name.replace(".dcm", ""), os.path.join(root, name)) for root, dirs, files in
                        os.walk(DATA_PATH + "/dicom-images-test")
                        for name in files if name.endswith((".dcm"))], columns=['ImageId', 'path'])



df_sub = pd.read_csv(SAMPLE_SUBMISSION)

df_rle = pd.read_csv(DATA_PATH + "/train-rle.csv")
df_rle = df_rle.rename(columns={' EncodedPixels': 'EncodedPixels'})
df_rle["EncodedPixels"] = df_rle["EncodedPixels"].map(lambda x: x[1:])

df_train = df_train.merge(df_rle, on="ImageId", how="left")

not_pneumothorax_ImageId = set(
    df_train.query("EncodedPixels == '-1' or EncodedPixels.isnull()", engine='python')["ImageId"])
df_train["pneumothorax"] = df_train["ImageId"].map(lambda x: 0 if x in not_pneumothorax_ImageId else 1)

df_train["rle_count"] = df_train["ImageId"].map(df_rle.groupby(["ImageId"]).size())
df_train["rle_count"] = df_train["rle_count"].fillna(-1)

## adding dicom metadata

df_meta = create_metadataset(df_train)
meta_feats = [c for c in df_meta.columns if c != "ImageId"]

df_train = df_train.merge(df_meta, on="ImageId", how='left')
df_test = df_test.merge(create_metadataset(df_test), on="ImageId", how='left')

df_train.to_csv("train.csv", index=False)
df_test.to_csv("test.csv", index=False)

df_sub["entries"] = df_sub["ImageId"].map(df_sub.groupby(['ImageId']).size())

print("train-rle: {}, unique ImageId: {}".format(len(df_rle), len(df_rle["ImageId"].unique())))
print("train: {}, unique ImageId: {}".format(len(df_train), len(df_train["ImageId"].unique())))
print("train ImageId not in rle: {}".format(
    len(df_train.query("EncodedPixels.isnull()", engine='python'))))
print("train ImageId with multiple rle: {}".format(
    len(df_train.query("rle_count > 1", engine='python')["ImageId"].unique())))

print("sample_submission: {}, unique ImageId: {}, ImegeId with multiple entries: {}".format(
    len(df_sub),
    len(df_sub["ImageId"].unique()),
    len(df_sub.query("entries > 1")["ImageId"].unique())
))

print("test: {}, unique ImageId: {}".format(len(df_test), len(df_test["ImageId"].unique())))
print("test ImageId not in sample_submission: {}".format(
    len(df_test[~ df_test["ImageId"].isin(df_sub["ImageId"])])))



# print('fasdfsa')