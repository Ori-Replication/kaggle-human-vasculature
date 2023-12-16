import torch as tc 
import torch.nn as nn  
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast
import cv2
import os,sys
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import segmentation_models_pytorch as smp
#import cc3d
from torch.utils.data import Dataset, DataLoader

class CFG:
# ============== model CFG =============
    model_name = 'Unet'
    #backbone = 'efficientnet-b0'
    backbone = 'se_resnet50'

    in_chans = 5 # 65
    # ============== training CFG =============
    image_size = 256
    input_size=256
    tile_size = image_size
    stride = tile_size // 2
    drop_egde_pixel=0
    
    target_size = 1
    # ============== fold =============
    valid_id = 1
    batch=64
    th_percentile = 0.002#0.005
    model_path=["./models/baseline-v0-2023-12-20/se_resnet50_0_loss0.42_score0.18_val_loss0.16_val_score0.37.pt"]
    
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


def build_model(weight=None):
    from dotenv import load_dotenv
    load_dotenv()

    print('model_name', CFG.model_name)
    print('backbone', CFG.backbone)

    model = CustomModel(CFG, weight)

    return model


def rle_encode(mask):
    pixel = mask.flatten()
    pixel = np.concatenate([[0], pixel, [0]])
    run = np.where(pixel[1:] != pixel[:-1])[0] + 1
    run[1::2] -= run[::2]
    rle = ' '.join(str(r) for r in run)
    if rle == '':
        rle = '1 0'
    return rle

def load_img(paths):
    output = []
    for path in paths:
        if path is None:
            output.append(None)
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = img.astype('float32') # original is uint16
        output.append(img)
    shape=[x.shape for x in output if x is not None][0]
    for i in range(len(output)):
        if output[i] is None:
            output[i] = tc.randn(shape)
    output=np.stack(output, axis=0)
    return tc.from_numpy(output)

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

class Pipeline_Dataset(Dataset):
    def __init__(self,x,path,labels=None):
        self.img_paths  = glob(path+"/images/*")
        self.img_paths.sort()
        #assert int(self.img_paths[-1].split("/")[-1][:-4])+1==len(x)
        self.debug=labels
        self.in_chan = 5
        z=tc.zeros(self.in_chan//2,*x.shape[1:],dtype=x.dtype)
        self.x=tc.cat((z,x,z),dim=0)
        self.labels=labels
        
    def __len__(self):
        return self.x.shape[0]-4
    
    def __getitem__(self, index):
        x  = self.x[index:index+self.in_chan]
        if self.labels is not None :
            label=self.labels[index]
        else:
            label=tc.zeros_like(x[0])
        #Normalization
        id=self.img_paths[index].split("/")[-3:]
        id.pop(1)
        id="_".join(id)
        #return img,tc.from_numpy(mask),id
        return x,label,id[:-4]

def add_edge(x:tc.Tensor,edge:int):
    #x=(C,H,W)
    #output=(C,H+2*edge,W+2*edge)
    x=tc.cat([x,tc.ones([x.shape[0],edge,x.shape[2]],dtype=x.dtype,device=x.device)*128],dim=1)
    x=tc.cat([x,tc.ones([x.shape[0],x.shape[1],edge],dtype=x.dtype,device=x.device)*128],dim=2)
    x=tc.cat([tc.ones([x.shape[0],edge,x.shape[2]],dtype=x.dtype,device=x.device)*128,x],dim=1)
    x=tc.cat([tc.ones([x.shape[0],x.shape[1],edge],dtype=x.dtype,device=x.device)*128,x],dim=2)
    return x

def TTA(x:tc.Tensor,model:nn.Module,batch=CFG.batch):
    x=x.to(tc.float32)
    x=min_max_normalization(x)
    #x.shape=(batch,c,h,w)
    if CFG.input_size!=CFG.image_size:
        x=nn.functional.interpolate(x,size=(CFG.input_size,CFG.input_size),mode='bilinear',align_corners=True)
    
    shape=x.shape
    x=[x,*[tc.rot90(x,k=i,dims=(-2,-1)) for i in range(1,4)]]
    x=tc.cat(x,dim=0)
    with autocast():
        with tc.no_grad():
            #x=[model((x[i*batch:(i+1)*batch],print(x[i*batch:(i+1)*batch].shape))[0]) for i in range(x.shape[0]//batch+1)]
            x=[model(x[i*batch:(i+1)*batch]) for i in range(x.shape[0]//batch+1)]
            # batch=64,64...48
            x=tc.cat(x,dim=0)
    x=x.sigmoid()
    x=x.reshape(4,shape[0],*shape[2:])
    x=[tc.rot90(x[i],k=-i,dims=(-2,-1)) for i in range(4)]
    x=tc.stack(x,dim=0).mean(0)
    
    if CFG.input_size!=CFG.image_size:
        x=nn.functional.interpolate(x[None],size=(CFG.image_size,CFG.image_size),mode='bilinear',align_corners=True)[0]
    return x

"""
def TTA(x:tc.Tensor, model:nn.Module, batch=CFG.batch):
    x = x.to(tc.float32)
    x = min_max_normalization(x)  # 假设提前已定义
    # x.shape=(batch,c,h,w)
    
    # 如果input_size和image_size不同，则执行新的padding策略
    if CFG.input_size != CFG.image_size:
        pad_height = CFG.input_size - x.size(2)
        pad_width = CFG.input_size - x.size(3)
        # 在底部和右侧进行零填充
        padding = [pad_width, 0, pad_height, 0] # 右, 左, 底, 上
        x = nn.functional.pad(x, padding, 'constant', 0)
    
    shape = x.shape
    x = [x, *[tc.rot90(x, k=i, dims=(-2, -1)) for i in range(1, 4)]]
    x = tc.cat(x, dim=0)
    with autocast():
        with tc.no_grad():
            x=[model(x[i*batch:(i+1)*batch]) for i in range(x.shape[0]//batch+1)]
            x = tc.cat(x, dim=0)
    x = x.sigmoid()
    x = x.reshape(4, shape[0], *shape[2:])
    x = [tc.rot90(x[i], k=-i, dims=(-2, -1)) for i in range(4)]
    x = tc.stack(x, dim=0).mean(0)
    
    if CFG.input_size != CFG.image_size:
        # 如果需要，剪裁返回到原始图像大小
        x = x[:, pad_height:, :CFG.image_size]  # 裁剪掉底部和右侧填充的部分
    
    return x
"""

from torch.nn import DataParallel

model=build_model()
model = DataParallel(model)
model.load_state_dict(tc.load(CFG.model_path[0],"cpu"))
model.eval()
model.cuda()


"""
1. `get_output`函数定义包含一个可选的`debug`参数，默认为`False`。

2. 当`debug`是`True`时，`paths`变量被直接赋值为一个具体路径列表。如果`debug`为`False`，则使用`glob.glob`函数检索相应文件夹中所有匹配指定模式的文件，但实际的`glob`模块导入语句缺失。

3. `debug_count`用来限制调试模式下迭代的次数。

4. 开始迭代处理每个路径下的数据。依据路径，它加载数据、创建数据集(`Pipeline_Dataset`)和相应的`DataLoader`，这个数据加载器用于批量读取数据并可以设置为随机顺序(shuffle)。

5. 在对每一批图像迭代过程中，它将图像`img`和标签`label`发送至CUDA设备（假设设备使用了NVIDIA GPU）。

6. 函数`add_edge`可能被用来对图像更新边缘，确切的效果和实现细节未特定。

7. 计算x和y方向上平铺的大小与步幅，以确定图像切分的方式。`CFG.tile_size`和`CFG.stride`似乎从某全局配置对象中获取（尚未在代码中定义）。

8. 使用初始化为零的`mask_pred`和`mask_count`来累积所有平铺图像的预测和计数。

9. 通过迭代平铺的每个子部分('chip')，并应用某种形式的测试时间增强（TTA），可能通过将图像多次传递给模型并稍微更改输入数据的方法来进行。

10. 根据需要剪切掉预测瓷砖的边缘(`drop_egde_pixel`)，因此只有中心区域被用于最终的预测累加。

11. 对每个瓷砖位置的预测进行求和，并更新计数器，然后将累加的结果除以计数器，得到最终的平均化预测掩膜。

12. 恢复原始目标掩膜的尺寸，去除添加的边缘。

13. 输出是一个元组列表，包含预测掩膜和相应ID。

14. 如果处于调试模式，代码会显示图像和预测掩膜，并且当可视化足够数量的样本后停止。
"""
def get_output(debug=False):
    outputs=[]
    if debug:
        paths=["./kaggle/input/blood-vessel-segmentation/train/kidney_2"]
    else:
        paths=["./kaggle/input/blood-vessel-segmentation/train/kidney_2"]
    debug_count=0
    for path in paths:
        x=load_data(path,"/images/")
        dataset=Pipeline_Dataset(x,path,None)
        print(dataset)
        dataloader=DataLoader(dataset,batch_size=1,shuffle=debug,num_workers=2)
        for img,label,id in tqdm(dataloader):
            #print(label.shape)
            #img=(C,H,W)

            img=img.to("cuda:0")
            label=label.to("cuda:0")

            img=add_edge(img[0],CFG.tile_size//2)[None]
            label=add_edge(label,CFG.tile_size//2)
            x1_list = np.arange(0, label.shape[-2]-CFG.tile_size+1, CFG.stride)
            y1_list = np.arange(0, label.shape[-1]-CFG.tile_size+1, CFG.stride)

            mask_pred = tc.zeros_like(label,dtype=tc.float32,device=label.device)
            mask_count = tc.zeros_like(label,dtype=tc.float32,device=label.device)

            indexs=[]
            chip=[]
            for y1 in y1_list:
                for x1 in x1_list:
                    x2 = x1 + CFG.tile_size
                    y2 = y1 + CFG.tile_size
                    indexs.append([x1+CFG.drop_egde_pixel,x2-CFG.drop_egde_pixel,
                                   y1+CFG.drop_egde_pixel,y2-CFG.drop_egde_pixel])
                    chip.append(img[...,x1:x2,y1:y2])

            y_preds = TTA(tc.cat(chip),model)
            if CFG.drop_egde_pixel:
                y_preds=y_preds[...,CFG.drop_egde_pixel:-CFG.drop_egde_pixel,
                                    CFG.drop_egde_pixel:-CFG.drop_egde_pixel]
            for i,(x1,x2,y1,y2) in enumerate(indexs):
                mask_pred[...,x1:x2, y1:y2] += y_preds[i]
                mask_count[...,x1:x2, y1:y2] += 1

            mask_pred /= mask_count

            #Rrecover
            mask_pred=mask_pred[...,CFG.tile_size//2:-CFG.tile_size//2,CFG.tile_size//2:-CFG.tile_size//2]
            label=label[...,CFG.tile_size//2:-CFG.tile_size//2,CFG.tile_size//2:-CFG.tile_size//2]

            outputs.append(((mask_pred*255).to(tc.uint8).cpu().numpy()[0],id))
            if debug:
                debug_count+=1
                plt.subplot(121)
                plt.imshow(img[0,2].cpu().detach().numpy())
                plt.subplot(122)
                plt.imshow(mask_pred[0].cpu().detach().numpy())
                plt.show()
                if debug_count>6:
                    break
    return outputs
    
    
import datetime
 
# 获取当前日期和时间
current_date = datetime.datetime.now().strftime("%Y-%m-%d")
current_time = datetime.datetime.now().strftime("%H-%M-%S")
 
print("当前日期为：", current_date)
print("当前时间为：", current_time)

is_submit=len(glob("./kaggle/input/blood-vessel-segmentation/test/kidney_5/images/*.tif"))!=3
outputs=get_output(False) # 如果文件夹中的文件数为3，则is_submit = False，启动debug？哦我草我懂了，这是应付kaggle那个机制

TH = [output.flatten() for output,id in outputs] 
TH = np.concatenate(TH)
index = -int(len(TH) * CFG.th_percentile)
TH:int = np.partition(TH, index)[index]
print(TH)
submission_df=[]
for mask_pred,id in outputs:
    if not is_submit:
        plt.subplot(121)
        plt.imshow(mask_pred)
        plt.subplot(122)
        plt.imshow(mask_pred>TH)
        plt.show()
    mask_pred=mask_pred>TH
    rle = rle_encode(mask_pred)
    
    submission_df.append(
        pd.DataFrame(data={
            'id'  : id,
            'rle' : rle,
        })
    )

submission_df =pd.concat(submission_df)
submission_df.to_csv('./data/predictions/prediction' + current_date + current_time +'.csv', index=False)
print(submission_df.head(6))


