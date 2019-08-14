# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os


DATA_DIR='fadsfsad'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

TRAIN_IMG_DIR = os.path.join(TRAIN_DIR, 'images')
TRAIN_MASK_DIR =  os.path.join(TRAIN_DIR, 'masks')
TEST_IMG_DIR = os.path.join(TEST_DIR, 'images')

LABEL_FILE = os.path.join(DATA_DIR, 'train.csv')
DEPTHS_FILE = os.path.join(DATA_DIR, 'depths.csv')
META_FILE = os.path.join(DATA_DIR, 'meta.csv')

MODEL_DIR = os.path.join(DATA_DIR, 'models')

ID_COLUMN = 'id'
DEPTH_COLUMN = 'z'
X_COLUMN = 'file_path_image'
Y_COLUMN = 'file_path_mask'

H = W = 128
ORIG_H = ORIG_W = 101


sample_submission_path = '../input/siim-acr-pneumothorax-segmentation/sample_submission.csv'
train_rle_path = '/home/cooper/PycharmProjects/SIIM/siim-acr-pneumothorax-segmentation/dataset/pneumothorax/train-rle.csv'
data_folder = "/home/cooper/PycharmProjects/SIIM/siim-acr-pneumothorax-segmentation/dataset/pneumothorax/dicom-images-train"
test_data_folder = "/home/cooper/PycharmProjects/SIIM/siim-acr-pneumothorax-segmentation/dataset/pneumothorax/dicom-images-test"
DATASETDIR='/home/cooper/PycharmProjects/SIIM/siim-acr-pneumothorax-segmentation/dataset/pneumothorax'
df_path='/home/cooper/PycharmProjects/nni/examples/trials/kaggle-tgs-salt/testdir/train.csv'