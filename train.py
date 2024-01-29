from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import torch.nn.functional as F
from scipy.ndimage import label, sum_labels
from utils.utils import Config, min_max_normalization, setup_seeds,\
                        get_date_time, SurfaceLoss, BCEWithLogitsLossManual,\
                        DiceLoss, norm_with_clip, add_noise
from utils.dataset import KaggleDataset
from models.unet import build_model # we use unet for the model now
from optimizer.loss import surface_dice

def main():
    cfg = Config('configs/train_config.yaml')
    setup_seeds(cfg)
    train_dataset = KaggleDataset(cfg)
    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=cfg.num_workers)
    val_dataset = KaggleDataset(cfg, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=cfg.num_workers)
    
    model = build_model(cfg) # unet
    model = DataParallel(model)

    loss_fn_1 = DiceLoss() 
    loss_fn_2 = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr,
                                                steps_per_epoch=len(train_loader), epochs=cfg.epochs+1,
                                                pct_start=0.1)
    
    date, time = get_date_time()
    
    for epoch in range(cfg.epochs):
        train_one_epoch(model, train_loader, loss_fn_1, loss_fn_1, optimizer, scaler, scheduler, epoch, cfg)
        val(model, val_loader, loss_fn_1,loss_fn_1, epoch, cfg)
        model_dir = os.path.join(cfg.output_path, f'{cfg.model_name}_{date}_{time}_New_Unet')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_save_path = os.path.join(model_dir, f'epoch_{epoch}.pt')
        torch.save(model.state_dict(), model_save_path)



def train_one_epoch(model, train_loader, loss_fn_1 ,loss_fn_2, optimizer, scaler, scheduler, epoch, cfg):
    model.train()
    iters = tqdm(range(len(train_loader)))
    train_loss = 0
    dice_score = 0
    
    for i, (images, masks) in enumerate(train_loader):
        images = images.cuda().to(torch.float32)
        masks = masks.cuda().to(torch.float32)
        images = norm_with_clip(images.reshape(-1,*images.shape[2:])).reshape(images.shape)
        images = add_noise(images,max_randn_rate=0.5,x_already_normed=True)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = 0.0 * loss_fn_1(outputs, masks) + 1 *loss_fn_2(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        train_loss = (train_loss*i + loss.item())/(i+1)
        dice_score = (dice_score*i + surface_dice(outputs, masks))/(i+1)
        iters.set_description(f"Epoch {epoch+1}/{cfg.epochs}, train loss: {train_loss:.4f}, dice score: {dice_score:.4f}")
        iters.update()

    iters.close()


def val(model, val_loader, loss_fn_1,loss_fn_2, epoch, cfg):
    model.eval()
    iters = tqdm(range(len(val_loader)))
    val_loss = 0
    val_dice_score = 0
    
    for i, (images, masks) in enumerate(val_loader):
        images = images.cuda().to(torch.float32)
        masks = masks.cuda().to(torch.float32)
        images = norm_with_clip(images.reshape(-1,*images.shape[2:])).reshape(images.shape)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                outputs = model(images)
                loss = 0.0 * loss_fn_1(outputs, masks) + 1 *loss_fn_2(outputs, masks)
        val_loss = (val_loss*i + loss.item())/(i+1)
        val_dice_score = (val_dice_score*i + surface_dice(outputs, masks))/(i+1)
        iters.set_description(f"Epoch {epoch+1}/{cfg.epochs}, val loss: {val_loss:.4f}, val dice score: {val_dice_score:.4f}")
        iters.update()
    iters.close()
    

if __name__ == '__main__':
    main()