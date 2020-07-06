import time
import gc
import torch
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from apex import amp
from config import *
from utils_train import collate_fn

train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        sampler=SequentialSampler(val_dataset),
        pin_memory=False,
        collate_fn=collate_fn,
    )

if apex:
    amp.initialize(model, optimizer, opt_level='O1')

def train_val(dataloader, train=True):
    t1 = time.time()
    running_loss = 0
    epoch_samples = 0
    if train:
        model.train()
        print("Initiating train phase ...")
    else:
        model.eval()
        print("Initiating val phase ...")
    with torch.set_grad_enabled(train):
        for step, (images, targets, image_ids) in enumerate(dataloader):
            images = torch.stack(images)
            images = images.to(device).float()
            batch_size = images.shape[0]
            boxes = [target['boxes'].to(device).float() for target in targets]
            labels = [target['labels'].to(device).float() for target in targets]
            optimizer.zero_grad()
            
            loss, _, _ = model(images, boxes, labels)
            running_loss += loss.sum().data.cpu().numpy()
            epoch_samples += images.size(0)
            
            if train:
                if apex:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            elapsed = int(time.time() - t1)
            eta = int(elapsed / (step+1) * (len(dataloader)-(step+1)))
            msg = f"Progress: [{step}/{len(dataloader)}] loss: {(running_loss/epoch_samples):.4f} Time: {elapsed}s ETA: {eta} s"
            print(msg, end='\r')
    return running_loss/epoch_samples

for i in range(n_epochs):
    train_loss = train_val(train_loader, True)
    torch.cuda.empty_cache()
    gc.collect()
    print(f'Epoch: {(i+1)/n_epochs} Phase: Train Loss: {train_loss:.4f}')
    val_loss = train_val(val_loader, False)
    print(f'Epoch: {(i+1)/n_epochs} Phase: Val Loss: {val_loss:.4f}')



