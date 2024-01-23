from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import cv2
from glob import glob
import os
import numpy as np


def min_max_normalization(x:torch.Tensor)->torch.Tensor:
    """input.shape=(batch,f1,...)"""
    shape=x.shape
    if x.ndim>2:
        x=x.reshape(x.shape[0],-1)
    
    min_=x.min(dim=-1,keepdim=True)[0]
    max_=x.max(dim=-1,keepdim=True)[0]
    if min_.mean()==0 and max_.mean()==1:
        return x.reshape(shape)
    
    x=(x-min_)/(max_-min_+1e-9)
    return x.reshape(shape)
class LoadData(Dataset):
    def __init__(self,path,is_label,s = "/images/"):
        self.paths = glob(path+f"{s}*.tif")
        self.paths.sort()
        self.bool = (s == "/labels/")
        self.is_label = is_label
        
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self,index):
        img=cv2.imread(self.paths[index],cv2.IMREAD_GRAYSCALE)
        img=torch.from_numpy(img)
        if self.is_label:
            img=(img!=0).to(torch.uint8)*255
        else:
            img=img.to(torch.uint8)
        return img
    
def load_data(path,s,is_label=False):
    loaded_datas = LoadData(path,is_label,s) 
    data_loader = DataLoader(loaded_datas, batch_size=16, num_workers=0)
    data=[]
    for x in tqdm(data_loader):
        data.append(x)
    x = torch.cat(data,dim=0)
    del data
    if not is_label:
        ########################################################################
        TH=x.reshape(-1).numpy()
        index = -int(len(TH) * 1e-3)  # TODO 这是一个可调超参数
        TH:int = np.partition(TH, index)[index]
        x[x>TH]=int(TH)
        ########################################################################
        TH=x.reshape(-1).numpy()
        index = -int(len(TH) * 1e-3)
        TH:int = np.partition(TH, -index)[-index]
        x[x<TH]=int(TH)
        ########################################################################
        x=(min_max_normalization(x.to(torch.float16)[None])[0]*255).to(torch.uint8)
    return x

class KaggleDataset(Dataset):
    def __init__(self, cfg, mode='train'):
        self.mode = mode
        image_chunks = []
        label_chunks = []
        dirs = os.listdir(os.path.join(cfg.root_path, 'train'))
        if mode == 'train':
            self.transform = cfg.train_aug
            for dir in dirs:
                if dir not in cfg.drop:
                    path = os.path.join(cfg.root_path, 'train', dir)
                    print('loading ' + path)
                    image_chunk = load_data(path, "/images/",is_label = False)
                    label_chunk = load_data(path, "/labels/",is_label = True)
                    # augmentation
                    # image_chunks.append(rotate_and_find_max_cuboid(image_chunk,0,30))
                    # image_chunks.append(rotate_and_find_max_cuboid(image_chunk,1,30))
                    # image_chunks.append(rotate_and_find_max_cuboid(image_chunk,2,30))
                    image_chunks.append(image_chunk)
                    image_chunks.append(image_chunk.permute(1, 2, 0))
                    image_chunks.append(image_chunk.permute(2, 0, 1))
                    # label_chunks.append(rotate_and_find_max_cuboid(label_chunks,0,30))
                    # label_chunks.append(rotate_and_find_max_cuboid(label_chunks,1,30))
                    # label_chunks.append(rotate_and_find_max_cuboid(label_chunks,2,30))
                    label_chunks.append(label_chunk)
                    label_chunks.append(label_chunk.permute(1, 2, 0))
                    label_chunks.append(label_chunk.permute(2, 0, 1))

        elif mode == 'val':
            self.transform = cfg.val_aug
            dir = cfg.valid[0]
            path = os.path.join(cfg.root_path, 'train', dir)
            print('loading validate ' + path)
            image_chunk = load_data(path, "/images/",is_label = False)
            label_chunk = load_data(path, "/labels/",is_label = True)
            image_chunks.append(image_chunk)
            label_chunks.append(label_chunk)

        self.image_chunks = image_chunks
        self.label_chunks = label_chunks
        self.input_size = cfg.input_size
        self.in_chans = cfg.in_chans

    def __len__(self):
        len = 0
        for data in self.image_chunks:
            len += data.shape[0] - self.in_chans
        return len


    def __getitem__(self, index):
        i = 0
        for data in self.image_chunks:
            if index > data.shape[0] - self.in_chans:
                index -= data.shape[0] - self.in_chans
                i += 1
            else:
                break
        x = self.image_chunks[i]
        y = self.label_chunks[i]

        x_index = (x.shape[1]-self.input_size)//2#np.random.randint(0,x.shape[1]-self.input_size)
        y_index = (x.shape[2]-self.input_size)//2#np.random.randint(0,x.shape[2]-self.input_size)

        x_slice = x[index:(index + self.in_chans), x_index:(x_index + self.input_size), y_index:(y_index + self.input_size)]
        y_slice = y[index + self.in_chans // 2, x_index:(x_index + self.input_size), y_index:(y_index + self.input_size)]

        pair = self.transform(image = x_slice.numpy().transpose(1,2,0), mask = y_slice.numpy())
        x_slice = pair['image']
        y_slice = pair['mask']>=127
        if self.mode == 'train':
            i = np.random.randint(4)
            x_slice = x_slice.rot90(i,dims=(1,2))
            y_slice = y_slice.rot90(i,dims=(0,1))
            for i in range(3):
                if np.random.randint(2):
                    x_slice = x_slice.flip(dims=(i,))
                    if i >= 1:
                        y_slice = y_slice.flip(dims=(i-1,))
        return x_slice, y_slice

from scipy.ndimage import affine_transform

def create_rotation_matrix(axis, angle):
    # 将角度从度转换为弧度
    theta = np.radians(angle)
    cos, sin = np.cos(theta), np.sin(theta)

    if axis == 0:  # 绕x轴旋转
        return np.array([[1, 0,      0],
                         [0, cos, -sin],
                         [0, sin,  cos]])
    elif axis == 1:  # 绕y轴旋转
        return np.array([[cos, 0, sin],
                         [0,   1,   0],
                         [-sin, 0, cos]])
    elif axis == 2:  # 绕z轴旋转
        return np.array([[cos, -sin, 0],
                         [sin,  cos, 0],
                         [0,     0,  1]])
    else:
        raise ValueError("Axis should be 0, 1, or 2.")

def apply_affine_transform(matrix, axis, angle):
    rot_matrix = create_rotation_matrix(axis, angle)
    # 获取输入矩阵的中心点，用于设置旋转中心
    center = np.array(matrix.shape) // 2
    # 计算仿射变换后的矩阵
    rotated_matrix = affine_transform(matrix, rot_matrix, offset=center-center.dot(rot_matrix))
    return rotated_matrix

def find_max_inscribed_cuboid(matrix):
    # 获取矩阵的三个维度
    depth, height, width = matrix.shape
    
    # 计算每个维度上需要裁减的层数
    d_cut = int(np.ceil(depth * 0.1))
    h_cut = int(np.ceil(height * 0.1))
    w_cut = int(np.ceil(width * 0.1))
    
    # 裁剪矩阵
    inscribed_cuboid = matrix[d_cut:depth - d_cut, h_cut:height - h_cut, w_cut:width - w_cut]
    
    return inscribed_cuboid

    

def rotate_and_find_max_cuboid(matrix, axis, angle):
    matrix = matrix.numpy()
    rotated_matrix = apply_affine_transform(matrix, axis, angle)
    max_cuboid = find_max_inscribed_cuboid(rotated_matrix)
    return torch.tensor(max_cuboid)