# Code For CS182: Deep learning for blood vessel segmentation
## Introduction
This is the code for shanghaitech CS182 Project from group 1. 
Team member: 洪沐天，卫正轩，葛俊辉

In this project, we used various models and different methods to segment vessels from 3D CT image.

# Startup
### Download data from kaggle
First, you should download the dataset from [SenNet + HOA - Hacking the Human Vasculature in 3D | Kaggle](https://www.kaggle.com/competitions/blood-vessel-segmentation/data) (43.15G)
You should put it in
```
kaggle/input/
```
The structure should be like:
```
├─kaggle
│  ├─input
│  │  ├─blood-vessel-segmentation
│  │  │  ├─test
│  │  │  │  ├─kidney_5
│  │  │  │  │  └─images
│  │  │  │  └─kidney_6
│  │  │  │      └─images
│  │  │  └─train
│  │  │      ├─kidney_1_dense
│  │  │      │  ├─images
│  │  │      │  └─labels
│  │  │      ├─kidney_1_voi
│  │  │      │  ├─images
│  │  │      │  └─labels
│  │  │      ├─kidney_2
│  │  │      │  ├─images
│  │  │      │  └─labels
│  │  │      ├─kidney_3_dense
│  │  │      │  └─labels
│  │  │      └─kidney_3_sparse
│  │  │          ├─images
│  │  │          └─labels
```
### Set up the environment
You should init a conda environment with python 3.10.x .Both Windows and Linux shoud be OK.
```command
conda create -n kaggle-bv python=3.10
conda activate kaggle-bv
```

Then install the required packages.
```command
pip install -r requirements.txt
```

### Run the train
The Train requires an GPU with >= 8G VRAM and a RAM >= 20G (Because the images in 3D is large)
```
python train.py
```
The train.py Uses a Unet with Resnet-50 as backbone, which is pretrained on ImageNet. So it requires Internet for you to download It. You may need to start an agent to download it.
If you are running on HPC, you should download it into `.cache/hub/checkpoints/se_resnext50_32x4d-a260b3a4.pth` 

It runs about 3 hours on school P40 HPC. The data loading takes a long time(about 20 minutes) so please be patient.
The config for this training is
```
configs/traim_config.yaml
```

The Swin Transformers model can be trained with 
```
python train-hf.py
```
It requires package huggingface, and it also requires the pretrained model from hugging face.
[microsoft/swin-base-simmim-window6-192 · Hugging Face](https://huggingface.co/microsoft/swin-base-simmim-window6-192)
```
microsoft/swin-base-simmim-window6-192
```
The config for this training is
```
configs/train_config_swin.yaml
```
### Run the inference
The model trained with the code will be saved under 
```
kaggle/working/checkpoints/{cfg.model_name}_{date}_{time}_New_Unet
```
or
```
kaggle/working/checkpoints/{cfg.model_name}_{date}_{time}_Swin
```
You can copy the latest checkpoint to do the infer.

For the Unet model, you should frist modify the path in 
```
inference-xy-yz-zx.py
```
Example
```python
class CFG:
	...
    model_path=["./kaggle/working/checkpoints/Unet_2023-12-18_00-37-18_epoch_21.pt"]
    test_data_path="./kaggle/input/blood-vessel-segmentation/train/kidney_2"


    kaggle = False
```
You only need to modify the model_path.

For the Swin model
you can download an example checkpoint from
(you may need to connect to shanghaitech VPN if you are not in school)
```
传输链接: (内网) https://send.deemos.com/download/aa1be2b09705cbed/#3lct-HUDdfFx-tNRBzA2iw 或打开 send.deemos.com 使用传输口令: puvJ1D 提取.
```

you should frist modify the path in 
```
inference-swin.py
```
Example
```python
class CFG:
	...
    model_path=["/public/sist/home/hongmt2022/MyWorks/kaggle-bv/models/Swin_No_Voi/epoch_19.pt"]
    test_data_path="./kaggle/input/blood-vessel-segmentation/train/kidney_2"


    kaggle = False
```
### Run the score compute
The inference will generate a csv file under
```
/data/predictions/
```
This is the RLE encode for the label 3d-image. (RLE encode Is a way to reduce the size of the label, we can decode it to generate 3D label images)

You can copy the path, and visulize it in the 
```
visulize.py
```
(It requires vtk package)
you can download an example predction csv file from
(you may need to connect to shanghaitech VPN if you are not in school)
```
传输链接: (内网) https://send.deemos.com/download/1d1b3c86ed429752/#C1sjNNbyLa9S0dNwr-7R4A 或打开 send.deemos.com 使用传输口令: QC4eaA 提取.
```
Example:
```python
if __name__ == "__main__":
    rle_csv_path = './data/predictions/prediction-dice-bce2023-12-31-14-37-31.csv'
    main(rle_csv_path)
```

And get an Interactable 3D model of the segmentation.

With the path, you can also Compute the Dice Score and Surface Dice Score.
```
python compute_score.py
```
you should modify this line:
```python
pred_path = 'data/predictions/prediction2023-12-1216-52-47.csv'
```

Warning: Compute the Surface Dice Score requires Really Large RAM. We don't know how much it takes, but we can't do it on a school AI cluster, but we can do it on school P40 cluster.