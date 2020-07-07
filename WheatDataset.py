import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import torch.nn.functional as F
import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=str(2**64)
from random import choices
from augmentations.augmentations import *
from config import *
# Any results you write to the current directory are saved as output.
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
from tqdm import tqdm_notebook as tqdm
# from utils import *
import warnings
warnings.filterwarnings('ignore')

TRAIN_ROOT_PATH = 'data/train'

class WheatDataset(Dataset):
    def __init__(self, image_ids, markings=None, dim=256, transforms=None, opts = ['normal', 'cutmix'], choice_weights = [0.5, 0.5],phase='train'):
        super().__init__()
        self.image_ids = image_ids
        self.markings = markings
        self.transforms = transforms
        self.dim = dim
        self.choice = choices(opts, weights=choice_weights)
        
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        if self.choice[0] == 'normal':
            image, boxes = load_image_and_boxes(TRAIN_ROOT_PATH, self.image_ids, idx, self.markings)

        elif self.choice[0] == 'cutmix':
            image, boxes = load_cutmix_image_and_boxes(TRAIN_ROOT_PATH, self.image_ids, idx, self.markings, self.dim)
        
        elif self.choice[0] == 'mosaic':
            image, boxes = load_mosaic_image_and_boxes(TRAIN_ROOT_PATH, self.image_ids, idx, self.markings)
            
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        
        if self.transforms is not None:
            for i in range(10):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    #yxyx: be warning
                    target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  
                    target['labels'] = torch.stack(sample['labels']) # <--- add this!
                    break
        return image, target, image_id

    def __len__(self):
        return len(self.image_ids)
