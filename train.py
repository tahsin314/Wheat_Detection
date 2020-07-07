import os
import sys
import time
import gc
import torch
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from apex import amp
from config import *
from utils_train import collate_fn

start_epoch = 0
best_val_loss = 1e20

if load_model:
    tmp =  torch.load(f'{model_dir}/{model_name}.pth')
    # torch.save(tmp['model_state_dict'], f'{model_dir}/{model_name}_weight.pth')
    model.load_state_dict(tmp['model_state_dict'])
    optimizer.load_state_dict(tmp['optimizer_state_dict'])
    start_epoch = tmp['Epoch'] + 1
    best_val_loss = tmp['best_loss']
    del tmp
    print('Model loaded')

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

for i in range(start_epoch, n_epochs):
    train_loss = train_val(train_loader, True)
    sys.stdout = open(os.devnull, 'w')
    torch.cuda.empty_cache()
    gc.collect()
    sys.stdout = sys.__stdout__
    print(f'Epoch: [{(i+1):02d}/{n_epochs:02d}] Phase: Train Loss: {train_loss:.4f}')
    val_loss = train_val(val_loader, False)
    lr_reduce_scheduler.step(val_loss)
    print(f'Epoch: [{(i+1):02d}/{n_epochs:02d}] Phase: Val Loss: {val_loss:.4f}')
    if val_loss < best_val_loss:
        print(f"Val loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
        best_val_loss = val_loss
        print('Saving model ...')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict':lr_reduce_scheduler.state_dict(),
            'best_loss': best_val_loss,
            'Epoch': i
        }, f'{model_dir}/{model_name}.pth')
        
        torch.save({
            'model_state_dict': model.model.state_dict(),
        }, f'{model_dir}/{model_name}_weight.pth')



