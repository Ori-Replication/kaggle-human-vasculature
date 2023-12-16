import random
import numpy as np
import torch
import pandas as pd
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import os

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
            # A.RandomBrightnessContrast(limit=0.2, p=0.5), 
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

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    print(runs)
    runs[1::2] -= runs[::2]# 1,3,5,7...-0,2,4,6...=1,1,1,1...
    print(runs)
    return ' '.join(str(x) for x in runs)
 
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def read_rle_from_path(rle_csv_path, height, width) -> np.ndarray :
    """

    Args:
        rle_csv_path (str): RLE数据的CSV文件路径
        height (_type_): 图片高度
        width (_type_): 图片宽度

    Returns:
        np.ndarray: 解码后的图片数据,为3D数组,shape为(num_images, height, width),值为0或1
    """
    rle_dataframe = pd.read_csv(rle_csv_path)
    decoded_images = []
    for index, row in rle_dataframe.iterrows():
        mask = rle_decode(row['rle'], shape=(height, width))
        decoded_images.append(mask)
    volume_data = np.stack(decoded_images, axis=0)
    return volume_data

def read_images(folder_path):
    """从指定文件夹中读取所有.tif格式的图片并返回3D numpy数组"""
    file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.tif')])
    images = [Image.open(os.path.join(folder_path, name)) for name in file_names]
    stacked_images = np.stack([np.array(image) for image in images], axis=0)
    return stacked_images


def get_date_time():
    import datetime
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.datetime.now().strftime("%H-%M-%S")
    return current_date, current_time
