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
from losses.arcface import ArcFaceLoss
from losses.focal import criterion_margin_focal_binary_cross_entropy
from models.EffDet import EffDet
import albumentations as A
from albumentations.augmentations.transforms import RandomRain, RandomSnow, RandomFog
from albumentations.pytorch.transforms import ToTensorV2
n_fold = 5
fold = 0
SEED = 24
batch_size = 5
num_workers = 4
sz = 512
learning_rate = 1e-3
patience = 4
accum_step = 20 // batch_size
opts = ['normal', 'cutmix', 'mosaic']
choice_weights = [0.5, 0.5, 0.0]
device = 'cuda:0'
apex = False 
pretrained_model = 'tf_efficientdet_d5'
model_name = f'{pretrained_model}_fold_{fold}'
model_dir = 'model_dir'
history_dir = 'history_dir'
load_model = True

os.makedirs(model_dir, exist_ok=True)
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
            A.Resize(height=sz, width=sz, p=1),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), blur_value=7, brightness_coefficient=0.7, rain_type="drizzle", always_apply=False, p=0.1),
            RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=1.5, always_apply=False, p=0.1),
            RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, always_apply=False, p=0.1),
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
            A.Resize(height=sz, width=sz, p=1.0),
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
    df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str))

df_folds.loc[:, 'fold'] = 0

for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
    df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number
train_df = df_folds[df_folds['fold'] != fold_number]
val_df = df_folds[df_folds['fold'] == fold_number]
device = "cuda:0"
model = EffDet(pretrained_model).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lr_reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08)
train_dataset = WheatDataset(train_df.index.values, markings=marking, dim=1024, transforms=train_aug, opts= opts, choice_weights = choice_weights)
val_dataset = WheatDataset(image_ids=val_df.index.values, markings=marking, dim=1024, transforms=val_aug, opts= ['normal'], choice_weights = [1.0],
    phase='val')
