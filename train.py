import torch
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from config import *
from WheatDataset import WheatDataset
from models.EffDet import get_net

device = "cuda:0"
model = get_net().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
train_dataset = WheatDataset(df_folds.index.values, markings=marking, dim=1024, transforms=train_aug)
def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

for step, (images, targets, image_ids) in enumerate(train_loader):
    images = torch.stack(images)
    images = images.to(device).float()
    batch_size = images.shape[0]
    boxes = [target['boxes'].to(device).float() for target in targets]
    labels = [target['labels'].to(device).float() for target in targets]
    optimizer.zero_grad()
    
    loss, _, _ = model(images, boxes, labels)
    print(loss.mean().data.cpu().numpy())
    loss.backward()