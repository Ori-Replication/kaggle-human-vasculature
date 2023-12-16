"""
- 2.5d cutting model baseline training
- 洪沐天
- 2023-12-20
v0
epoch:0,loss:0.4211,score:0.1758,lr5.8858e-05: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 842/842 [03:19<00:00,  4.22it/s]
val-->loss:0.1609,score:0.3656: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 143/143 [00:19<00:00,  7.21it/s]
epoch:1,loss:0.0878,score:0.5295,lr5.8782e-05: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 842/842 [03:04<00:00,  4.56it/s]
val-->loss:0.0531,score:0.6054: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 143/143 [00:19<00:00,  7.26it/s]
epoch:2,loss:0.0383,score:0.6355,lr5.4704e-05: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 842/842 [03:07<00:00,  4.49it/s]
val-->loss:0.0289,score:0.6884: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 143/143 [00:19<00:00,  7.46it/s]
epoch:3,loss:0.0232,score:0.6916,lr4.8159e-05: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 842/842 [03:04<00:00,  4.55it/s]
val-->loss:0.0193,score:nan: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 143/143 [00:19<00:00,  7.41it/s]
epoch:4,loss:0.0180,score:0.7133,lr3.9801e-05: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 842/842 [03:06<00:00,  4.52it/s]
val-->loss:0.0145,score:nan: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 143/143 [00:19<00:00,  7.29it/s]
epoch:5,loss:0.0150,score:0.7310,lr3.0465e-05: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 842/842 [03:01<00:00,  4.63it/s]
val-->loss:0.0120,score:nan: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 143/143 [00:19<00:00,  7.30it/s]
epoch:6,loss:0.0131,score:0.7473,lr2.1082e-05: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 842/842 [03:06<00:00,  4.52it/s]
val-->loss:0.0112,score:nan: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 143/143 [00:19<00:00,  7.20it/s]
epoch:7,loss:0.0120,score:0.7565,lr1.2589e-05: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 842/842 [03:05<00:00,  4.55it/s]
val-->loss:0.0097,score:nan: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 143/143 [00:19<00:00,  7.29it/s]
epoch:8,loss:0.0127,score:0.7475,lr5.8354e-06: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 842/842 [03:07<00:00,  4.49it/s]
val-->loss:0.0103,score:nan: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 143/143 [00:19<00:00,  7.20it/s]
epoch:9,loss:0.0133,score:0.7581,lr1.4946e-06: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 842/842 [03:06<00:00,  4.52it/s]
val-->loss:0.0119,score:nan: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 143/143 [00:19<00:00,  7.49it/s]
"""

# !mkdir -p .cache/torch/hub/checkpoints/
# !cp kaggle/input/se-net-pretrained-imagenet-weights/* .cache/torch/hub/checkpoints/
import torch as tc 
import torch.nn as nn  
import numpy as np
from tqdm import tqdm

import os,sys,cv2
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
import albumentations as A
# !python -m pip install --no-index --find-links=/kaggle/input/pip-download-for-segmentation-models-pytorch segmentation-models-pytorch
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
from glob import glob


class CFG:
    # ============== pred target =============
    target_size = 1

    # ============== model CFG =============
    model_name = 'Unet'
    backbone = 'se_resnet50'

    in_chans = 5 # 65
    # ============== training CFG =============
    image_size = 256
    input_size=256
    drop_egde_pixel = 0
    tile_size = image_size
    stride = tile_size // 2
    assert stride>drop_egde_pixel

    train_batch_size = 16 # 32
    valid_batch_size = train_batch_size * 2

    epochs = 10
    lr = 6e-5

    # ============== fold =============
    valid_id = 1


    # ============== augmentation =============
    train_aug_list = [
        A.RandomResizedCrop(
            input_size, input_size, scale=(0.8,1.25)),
        A.ShiftScaleRotate(p=0.75),
        A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                ], p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        # A.RandomBrightness(limit=0.2, p=0.5), 
        ToTensorV2(transpose_mask=True),
    ]
    train_aug = A.Compose(train_aug_list)
    valid_aug_list = [
        ToTensorV2(transpose_mask=True),
    ]
    valid_aug = A.Compose(valid_aug_list)
    
    
class CustomModel(nn.Module):
    def __init__(self, CFG, weight=None):
        super().__init__()
        self.CFG = CFG
        self.encoder = smp.Unet(
            encoder_name=CFG.backbone, 
            encoder_weights=weight,
            in_channels=CFG.in_chans,
            classes=CFG.target_size,
            activation=None,
        )

    def forward(self, image):
        output = self.encoder(image)
        # output = output.squeeze(-1)
        return output[:,0]#.sigmoid()


def build_model(weight="imagenet"):
    from dotenv import load_dotenv
    load_dotenv()

    print('model_name', CFG.model_name)
    print('backbone', CFG.backbone)

    model = CustomModel(CFG, weight)

    return model.cuda()

def min_max_normalization(x:tc.Tensor)->tc.Tensor:
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

class Data_loader(Dataset):
    def __init__(self,path,s="/images/"):
        self.paths=glob(path+f"{s}*.tif")
        self.paths.sort()
        self.bool=s=="/labels/"
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self,index):
        img=cv2.imread(self.paths[index],cv2.IMREAD_GRAYSCALE)
        img=tc.from_numpy(img)
        if self.bool:
            img=img.to(tc.bool)
        else:
            img=img.to(tc.uint8)
        return img

def load_data(path,s):
    data_loader=Data_loader(path,s)
    data_loader=DataLoader(data_loader, batch_size=16, num_workers=2)
    data=[]
    for x in tqdm(data_loader):
        data.append(x)
    return tc.cat(data,dim=0)

def surface_dice(pred:tc.Tensor,target:tc.Tensor):
    if tc.any(pred<0) or tc.any(pred>1):
        pred=pred.sigmoid()
    pred=pred.reshape(-1)>0.5
    target=target.reshape(-1)
    intersection=tc.sum(pred*target)
    union=tc.sum(pred+target)
    return (2*intersection/union).cpu().item()
    
class Kaggld_Dataset(Dataset):
    def __init__(self,x:list,y:list,arg=False):
        super(Dataset,self).__init__()
        self.x=x#list[(C,H,W),...]
        self.y=y#list[(C,H,W),...]
        self.image_size=CFG.image_size
        self.in_chans=CFG.in_chans
        self.arg=arg
        if arg:
            self.transform=CFG.train_aug
        else: 
            self.transform=CFG.valid_aug

    def __len__(self) -> int:
        return sum([y.shape[0]-self.in_chans for y in self.y])
    
    def __getitem__(self,index):
        i=0
        for x in self.x:
            if index>x.shape[0]-self.in_chans:
                index-=x.shape[0]-self.in_chans
                i+=1
            else:
                break
        x=self.x[i]
        y=self.y[i]
        
        x_index=np.random.randint(0,x.shape[1]-self.image_size)
        y_index=np.random.randint(0,x.shape[2]-self.image_size)

        x=x[index:index+self.in_chans,x_index:x_index+self.image_size,y_index:y_index+self.image_size].to(tc.float32)
        y=y[index+self.in_chans//2,x_index:x_index+self.image_size,y_index:y_index+self.image_size].to(tc.float32)

        data = self.transform(image=x.numpy().transpose(1,2,0), mask=y.numpy())
        x = data['image']
        y = data['mask']
        if self.arg:
            i=np.random.randint(4)
            x=x.rot90(i,dims=(1,2))
            y=y.rot90(i,dims=(0,1))
            for i in range(3):
                if np.random.randint(2):
                    x=x.flip(dims=(i,))
                    if i>=1:
                        y=y.flip(dims=(i-1,))
        return x,y


train_x=[]
train_y=[]
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = current_path+"/kaggle/input/blood-vessel-segmentation/"
paths=glob(root_path+"train/*")
paths.sort()
print(paths)
for i,path in enumerate(paths):#Because the memory is not enough, so I don't use some data.
    if path=="/public/sist/home/hongmt2022/MyWorks/kaggle-bv/kaggle/input/blood-vessel-segmentation/train/kidney_3_dense" or path == '/public/sist/home/hongmt2022/MyWorks/kaggle-bv/kaggle/input/blood-vessel-segmentation/train/kidney_1_voi':
        print('pass the' + path)
        continue
    print('loading' + path)
    x=load_data(path,"/images/")
    print(x.shape)
    y=load_data(path,"/labels/")
    train_x.append(x)
    train_y.append(y)

    #(C,H,W)

    #aug
    train_x.append(x.permute(1,2,0))
    train_y.append(y.permute(1,2,0))
    train_x.append(x.permute(2,0,1))
    train_y.append(y.permute(2,0,1))

val_x=load_data(paths[0],"/images/")
val_y=load_data(paths[0],"/labels/")


train_dataset=Kaggld_Dataset(train_x,train_y,arg=True)
train_dataset = DataLoader(train_dataset, batch_size=16, num_workers=2, shuffle=True, pin_memory=True)
val_dataset=Kaggld_Dataset([val_x],[val_y])
val_dataset = DataLoader(val_dataset, batch_size=16, num_workers=2, shuffle=False, pin_memory=True)

model=build_model()
model=DataParallel(model)

loss_fn=nn.DiceLoss()
optimizer=tc.optim.AdamW(model.parameters(),lr=CFG.lr)
scaler=tc.cuda.amp.GradScaler()
scheduler = tc.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.lr,
                                                steps_per_epoch=len(train_dataset), epochs=CFG.epochs+1,
                                                pct_start=0.1,)
for epoch in range(CFG.epochs):
    time=tqdm(range(len(train_dataset)))
    losss=0
    scores=0
    for i,(x,y) in enumerate(train_dataset):
        x=x.cuda()
        y=y.cuda()
        x=min_max_normalization(x)

        with autocast():
            pred=model(x)
            loss=loss_fn(pred,y)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        score=surface_dice(pred.detach(),y)
        losss=(losss*i+loss.item())/(i+1)
        scores=(scores*i+score)/(i+1)
        time.set_description(f"epoch:{epoch},loss:{losss:.4f},score:{scores:.4f},lr{optimizer.param_groups[0]['lr']:.4e}")
        time.update()
        del loss,pred
    time.close()
    val_losss=0
    val_scores=0
    time=tqdm(range(len(val_dataset)))
    for i,(x,y) in enumerate(val_dataset):
        x=x.cuda()
        y=y.cuda()
        x=min_max_normalization(x)

        with autocast():
            with tc.no_grad():
                pred=model(x)
                loss=loss_fn(pred,y)
        score=surface_dice(pred.detach(),y)
        if not isinstance(score,float):
            score=0
        val_losss=(val_losss*i+loss.item())/(i+1)
        val_scores=(val_scores*i+score)/(i+1)
        time.set_description(f"val-->loss:{val_losss:.4f},score:{val_scores:.4f}")
        time.update()

    time.close()
    tc.save(model.state_dict(),f"./{CFG.backbone}_{epoch}_loss{losss:.2f}_score{scores:.2f}_val_loss{val_losss:.2f}_val_score{val_scores:.2f}.pt")

time.close()