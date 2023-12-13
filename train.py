from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import torch
import torch.nn as nn
from tqdm import tqdm
import os

from utils.utils import Config, setup_seeds, min_max_normalization
from utils.dataset import KaggleDataset
from models.unet import build_model
from optimizer.loss import surface_dice

def main():
    cfg = Config('configs/train_config.yaml')
    setup_seeds(cfg)

    train_dataset = KaggleDataset(cfg)
    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=cfg.num_workers)
    val_dataset = KaggleDataset(cfg, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=cfg.num_workers)

    model = build_model(cfg)
    model = DataParallel(model)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr,
                                                steps_per_epoch=len(train_dataset), epochs=cfg.epochs+1,
                                                pct_start=0.1)
    
    for epoch in range(cfg.epochs):
        train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, scheduler, epoch, cfg)
        val(model, val_loader, loss_fn, epoch, cfg)
        torch.save(model.state_dict(), os.path.join(cfg.output_path, f'{cfg.model_name}_epoch_{epoch}.pt'))



def train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, scheduler, epoch, cfg):
    model.train()
    iters = tqdm(range(len(train_loader)))
    train_loss = 0
    dice_score = 0
    
    for i, (images, masks) in enumerate(train_loader):
        images = images.cuda()
        masks = masks.cuda()
        images = min_max_normalization(images)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = loss_fn(outputs, masks)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss = (train_loss*i + loss.item())/(i+1)
        dice_score = (dice_score*i + surface_dice(outputs, masks))/(i+1)
        iters.set_description(f"Epoch {epoch+1}/{cfg.epochs}, train loss: {train_loss:.4f}, dice score: {dice_score:.4f}")
        iters.update()
    iters.close()


def val(model, val_loader, loss_fn, epoch, cfg):
    model.eval()
    iters = tqdm(range(len(val_loader)))
    val_loss = 0
    val_dice_score = 0
    
    for i, (images, masks) in enumerate(val_loader):
        images = images.cuda()
        masks = masks.cuda()
        images = min_max_normalization(images)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                outputs = model(images)
                loss = loss_fn(outputs, masks)

        val_loss = (val_loss*i + loss.item())/(i+1)
        val_dice_score = (val_dice_score*i + surface_dice(outputs, masks))/(i+1)
        iters.set_description(f"Epoch {epoch+1}/{cfg.epochs}, val loss: {val_loss:.4f}, val dice score: {val_dice_score:.4f}")
        iters.update()
    iters.close()
    


if __name__ == '__main__':
    main()