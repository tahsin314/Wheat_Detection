import os
import numpy as np 
from sklearn.model_selection import StratifiedKFold
import cv2
import pandas as pd
import torch
from torch import optim
from WheatDataset import WheatDataset
from augmentations.augmix import RandomAugMix
from augmentations.gridmask import GridMask
from augmentations.hair import Hair, AdvancedHairAugmentationAlbumentations
from augmentations.microscope import MicroscopeAlbumentations
from losses.arcface import ArcFaceLoss
from losses.focal import criterion_margin_focal_binary_cross_entropy
from models.EffDet import EffDet
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
n_fold = 5
fold = 0
SEED = 24
batch_size = 8
num_workers = 4
sz = 512
learning_rate = 2e-3
patience = 4
accum_step = 48 // batch_size
opts = ['normal', 'mixup', 'cutmix']
choice_weights = [0.8, 0.1, 0.1]
device = 'cuda:0'
apex = True
pretrained_model = 'tf_efficientdet_d5'
model_name = f'{pretrained_model}_fold_{fold}'
model_dir = 'model_dir'
history_dir = 'history_dir'
load_model = True

if load_model and os.path.exists(os.path.join(history_dir, 'history_{}.csv'.format(model_name))):
    history = pd.read_csv(os.path.join(history_dir, 'history_{}.csv'.format(model_name)))
else:
    history = pd.DataFrame()

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
n_epochs = 40
TTA = 1
balanced_sampler = False
pseudo_lo_thr = 0.1
pseudo_up_thr = 0.7

train_aug = A.Compose(
        [
            A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                           contrast_limit=0.2, p=0.9),
            ],p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            ToTensorV2(p=1.0, always_apply=True),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )
val_aug = A.Compose(
        [
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0, always_apply=True),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )

data_dir = 'data'
image_path = f'{data_dir}/train/'
test_image_path = f'{data_dir}/test'
marking = pd.read_csv(f'{data_dir}/train.csv')
bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
for i, column in enumerate(['x', 'y', 'w', 'h']):
    marking[column] = bboxs[:,i]
marking.drop(columns=['bbox'], inplace=True)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

df_folds = marking[['image_id']].copy()
df_folds.loc[:, 'bbox_count'] = 1
df_folds = df_folds.groupby('image_id').count()
df_folds.loc[:, 'source'] = marking[['image_id', 'source']].groupby('image_id').min()['source']
df_folds.loc[:, 'stratify_group'] = np.char.add(
    df_folds['source'].values.astype(str),
    df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
)
df_folds.loc[:, 'fold'] = 0

for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
    df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number
train_df = df_folds[df_folds['fold'] != fold_number]
val_df = df_folds[df_folds['fold'] == fold_number]
device = "cuda:0"
model = EffDet(pretrained_model).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
train_dataset = WheatDataset(train_df.index.values, markings=marking, dim=1024, transforms=train_aug)
val_dataset = WheatDataset(image_ids=val_df.index.values, markings=marking, dim=1024,            transforms=val_aug,
    phase='val',
)