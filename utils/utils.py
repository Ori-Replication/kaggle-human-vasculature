import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import os
from torch import einsum
from torch import Tensor
from scipy.ndimage import distance_transform_edt as distance
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union
import cv2

class Config:
    def __init__(self, cfg_path):
        cfg = OmegaConf.load(cfg_path)
        for k, v in cfg.items():
            setattr(self, k, v)
        
        p_augm = 0.05
        train_aug_list = [
            A.Rotate(limit=270, p= 0.02),
            A.RandomScale(scale_limit=(0.8,1.25),interpolation=cv2.INTER_CUBIC,p=p_augm),
            A.RandomCrop(cfg.input_size, cfg.input_size,p=1),
            A.RandomGamma(p=p_augm*2/3),
            A.RandomBrightnessContrast(p=p_augm,),
            A.GaussianBlur(p=p_augm),
            A.MotionBlur(p=p_augm),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=p_augm),
            ToTensorV2(transpose_mask=True),
        ]
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

def norm_with_clip(x:torch.Tensor,smooth=1e-5):
    dim=list(range(1,x.ndim))
    mean=x.mean(dim=dim,keepdim=True)
    std=x.std(dim=dim,keepdim=True)
    x=(x-mean)/(std+smooth)
    x[x>5]=(x[x>5]-5)*1e-3 +5
    x[x<-3]=(x[x<-3]+3)*1e-3-3
    return x

def add_noise(x:torch.Tensor,max_randn_rate=0.1,randn_rate=None,x_already_normed=False):
    """input.shape=(batch,f1,f2,...) output's var will be normalizate  """
    ndim=x.ndim-1
    if x_already_normed:
        x_std=torch.ones([x.shape[0]]+[1]*ndim,device=x.device,dtype=x.dtype)
        x_mean=torch.zeros([x.shape[0]]+[1]*ndim,device=x.device,dtype=x.dtype)
    else: 
        dim=list(range(1,x.ndim))
        x_std=x.std(dim=dim,keepdim=True)
        x_mean=x.mean(dim=dim,keepdim=True)
    if randn_rate is None:
        randn_rate=max_randn_rate*np.random.rand()*torch.rand(x_mean.shape,device=x.device,dtype=x.dtype)
    cache=(x_std**2+(x_std*randn_rate)**2)**0.5
    #https://blog.csdn.net/chaosir1991/article/details/106960408
    
    return (x-x_mean+torch.randn(size=x.shape,device=x.device,dtype=x.dtype)*randn_rate*x_std)/(cache+1e-7)

def dice_coef(y_pred:torch.Tensor,y_true:torch.Tensor, thr=0.5, dim=(-1,-2), epsilon=0.001):
    y_pred=y_pred.sigmoid()
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean()
    return dice

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = inputs.sigmoid()   
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

def three_dimension_dice_score(pred:np.ndarray, target:np.ndarray):
    """计算三维Dice分数"""
    intersection = np.sum(pred*target)
    union = np.sum(pred)+ np.sum(target)
    return (2*intersection/union).item() * 100

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
 
# switch between representations
def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
    assert simplex(probs)
 
    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)
 
    return res
 
 
def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, _, _ = probs.shape
    assert simplex(probs)
 
    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)
 
    return res
 
 
def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))
 
    b, w, h = seg.shape  # type: Tuple[int, int, int]
 
    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)
 
    return res
 
 
def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    assert one_hot(torch.Tensor(seg), axis=0)
    C: int = len(seg)
 
    res = np.zeros_like(seg)
    # res = res.astype(np.float64)
    for c in range(C):
        posmask = seg[c].astype(np.bool)
 
        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res
 
 
def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)
 
 
def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])
 
    # Assert utils
 
 
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())
 
 
def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)
 
 
class SurfaceLoss():# TODO:把surfaceLoss用起来，可能在训练过程中前期权重较低，后期提高
    def __init__(self):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = [1]  # 这里忽略背景类  https://github.com/LIVIAETS/surface-loss/issues/3
 
    # probs: bcwh, dist_maps: bcwh
    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)
 
        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)
 
        multiplied = einsum("bcwh,bcwh->bcwh", pc, dc)
 
        loss = multiplied.mean()
 
        return loss
 
class BCEWithLogitsLossManual(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        """
        weight: 给每个类别分配的权重 (如果提供)
        reduction: 指定返回值的形式: 'none', 'mean' 或 'sum' 
        """
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, input_logits, target):
        """
        input_logits: 预测值的logits，未经过sigmoid变换
        target: 真实的标签
        """
        # 使用 sigmoid 函数将 logits 映射到 [0, 1] 区间
        probabilities = torch.sigmoid(input_logits)

        # 计算 log(probabilities) 和 log(1 - probabilities) 并处理数值稳定性问题
        max_val = (-input_logits).clamp(min=0)
        log_probs = input_logits - input_logits * target + max_val + \
            ((-max_val).exp() + (-input_logits - max_val).exp()).log()

        # 计算二元交叉熵损失
        bce_loss = -(target * log_probs + (1 - target) * F.logsigmoid(-input_logits))

        # 如果有权重，则应用它们
        if self.weight is not None:
            bce_loss *= self.weight

        # 根据 reduction 参数进行降维
        if self.reduction == 'mean':
            return torch.mean(bce_loss)
        elif self.reduction == 'sum':
            return torch.sum(bce_loss)
        else: # 'none'
            return bce_loss

def read_kidney_2_label():
    rle_path = './kaggle/input/blood-vessel-segmentation/train_rles.csv'
    df = pd.read_csv(rle_path)
    df = df[df['id'].str.contains('kidney_2')]
    decoded_images = []
    for index, row in df.iterrows():
        mask = rle_decode(row['rle'], shape=(1041, 1511))
        decoded_images.append(mask)
    volume_data = np.stack(decoded_images, axis=0)
    return volume_data
    