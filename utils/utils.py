import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Config:
    def __init__(self, cfg_path):
        cfg = OmegaConf.load(cfg_path)
        for k, v in cfg.items():
            setattr(self, k, v)
        
        train_aug_list = [
            A.RandomResizedCrop(cfg.input_size, cfg.input_size, scale=(0.8,1.25)),
            A.ShiftScaleRotate(p=0.75),
            A.OneOf([A.GaussNoise(var_limit=[10, 50]),
                    A.GaussianBlur(),
                    A.MotionBlur()], p=0.4),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            # A.RandomBrightness(limit=0.2, p=0.5), 
            ToTensorV2(transpose_mask=True)]
        
        val_aug_list = [ToTensorV2(transpose_mask=True)]
        
        self.train_aug = A.Compose(train_aug_list)
        self.val_aug = A.Compose(val_aug_list)

def setup_seeds(cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def min_max_normalization(x:torch.Tensor)->torch.Tensor:
    """input.shape=(batch,f1,...)"""
    shape = x.shape
    if x.ndim>2:
        x = x.reshape(x.shape[0],-1)
    
    min_ = x.min(dim=-1,keepdim=True)[0]
    max_ = x.max(dim=-1,keepdim=True)[0]
    if min_.mean() == 0 and max_.mean() == 1:
        return x.reshape(shape)
    
    x = (x-min_) / (max_-min_+1e-9)
    return x.reshape(shape)