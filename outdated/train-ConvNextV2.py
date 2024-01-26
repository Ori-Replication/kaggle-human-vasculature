from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import torch.nn.functional as F
from scipy.ndimage import label, sum_labels
from utils.utils import Config, min_max_normalization, setup_seeds, get_date_time
from utils.dataset import KaggleDataset
from models.unet import build_model
from optimizer.loss import surface_dice

from transformers import AutoImageProcessor, ConvNextV2Model
from transformers import ConvNextV2Config

class DiceLoss(nn.Module):
    def __init__(self, smooth=0.00001):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + self.smooth)/(inputs.sum() + targets.sum() + self.smooth)  
        
        return 1 - dice

def main():
    cfg = Config('configs/train_config_swin.yaml')
    setup_seeds(cfg)
    train_dataset = KaggleDataset(cfg)
    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=cfg.num_workers)
    val_dataset = KaggleDataset(cfg, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=cfg.num_workers)

    
    # Initializing a ConvNeXTV2 convnextv2-tiny-1k-224 style configuration
    configuration = ConvNextV2Config()
    # Initializing a model (with random weights) from the convnextv2-tiny-1k-224 style configuration
    model = ConvNextV2Model(configuration)
    model = ConvNextV2Model.from_pretrained("facebook/convnextv2-tiny-1k-224")
    model = model.cuda()
    model = DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr,
                                                steps_per_epoch=len(train_loader), epochs=cfg.epochs+1,
                                                pct_start=0.1)
    loss_fn = DiceLoss()
    date, time = get_date_time()
    
    for epoch in range(cfg.epochs):
        train_one_epoch(model, train_loader, loss_fn,loss_fn,  optimizer, scaler, scheduler, epoch, cfg)
        val(model, val_loader, loss_fn,loss_fn, epoch, cfg)
        model_dir = os.path.join(cfg.output_path, f'{cfg.model_name}_{date}_{time}_Dice_With_Surface')
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
        images = images.cuda()
        masks = masks.cuda()
        images = min_max_normalization(images)

        with torch.cuda.amp.autocast():
            outputs = model(images,return_dict = False)
            outputs_pred = outputs[0]
            print(outputs_pred.shape)
            outputs_pred = outputs_pred[:,cfg.in_chans//2,:,:]
            # loss = 0.0 * loss_fn_1(outputs_pred, masks) + 1 *loss_fn_2(outputs_pred, masks)
            loss = loss_fn_2(outputs_pred, masks)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss = (train_loss*i + loss.item())/(i+1)
        dice_score = (dice_score*i + surface_dice(outputs_pred, masks))/(i+1)
        iters.set_description(f"Epoch {epoch+1}/{cfg.epochs}, train loss: {train_loss:.4f}, dice score: {dice_score:.4f}")
        iters.update()
    iters.close()
    
def val(model, val_loader, loss_fn_1,loss_fn_2, epoch, cfg):
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
                outputs = model(images,return_dict = False)
                outputs_pred = outputs
                outputs_pred = 1 - outputs_pred[:,cfg.in_chans//2,:,:]
                # loss = 0.0 * loss_fn_1(outputs_pred, masks) + 1 *loss_fn_2(outputs_pred, masks)
                loss = loss_fn_2(outputs_pred, masks)
        val_loss = (val_loss*i + loss.item())/(i+1)
        val_dice_score = (val_dice_score*i + surface_dice(outputs_pred, masks))/(i+1)
        iters.set_description(f"Epoch {epoch+1}/{cfg.epochs}, val loss: {val_loss:.4f}, val dice score: {val_dice_score:.4f}")
        iters.update()
    iters.close()
    
    
if __name__ == '__main__':
    main()