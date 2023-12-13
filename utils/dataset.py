from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import cv2
from glob import glob
import os
import numpy as np

class LoadData(Dataset):
    def __init__(self,path,s = "/images/"):
        self.paths = glob(path+f"{s}*.tif")
        self.paths.sort()
        self.bool = (s == "/labels/")
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self,index):
        img = cv2.imread(self.paths[index],cv2.IMREAD_GRAYSCALE)
        img = torch.from_numpy(img)
        if self.bool:
            img = img.to(torch.bool)
        else:
            img = img.to(torch.uint8)
        return img
    
def load_data(path,s):
    loaded_datas = LoadData(path,s)
    data_loader = DataLoader(loaded_datas, batch_size=16, num_workers=0)
    data=[]
    for x in tqdm(data_loader):
        data.append(x)
    return torch.cat(data,dim=0)

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
                    image_chunk = load_data(path, "/images/")
                    label_chunk = load_data(path, "/labels/")
                    # augmentation
                    image_chunks.append(image_chunk)
                    image_chunks.append(image_chunk.permute(1, 2, 0))
                    image_chunks.append(image_chunk.permute(2, 0, 1))
                    label_chunks.append(label_chunk)
                    label_chunks.append(label_chunk.permute(1, 2, 0))
                    label_chunks.append(label_chunk.permute(2, 0, 1))
        elif mode == 'val':
            self.transform = cfg.val_aug
            dir = dirs[0]
            path = os.path.join(cfg.root_path, 'train', dir)
            print('loading validate ' + path)
            image_chunk = load_data(path, "/images/")
            label_chunk = load_data(path, "/labels/")
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
        
        x_index = np.random.randint(0,x.shape[1]-self.input_size)
        y_index = np.random.randint(0,x.shape[2]-self.input_size)

        x_slice = x[index:(index + self.in_chans), x_index:(x_index + self.input_size), y_index:(y_index + self.input_size)].to(torch.float32)
        y_slice = y[index + self.in_chans // 2, x_index:(x_index + self.input_size), y_index:(y_index + self.input_size)].to(torch.float32)

        pair = self.transform(image = x_slice.numpy().transpose(1,2,0), mask = y_slice.numpy())
        x_slice = pair['image']
        y_slice = pair['mask']
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