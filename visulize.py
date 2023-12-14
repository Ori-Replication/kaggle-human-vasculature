import os
import vtk
from PIL import Image
import numpy as np
from utils.utils import rle_decode, read_rle_from_path
def read_images(folder_path):
    """从指定文件夹中读取所有.tif格式的图片并返回3D numpy数组"""
    file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.tif')])
    images = [Image.open(os.path.join(folder_path, name)) for name in file_names]
    stacked_images = np.stack([np.array(image) for image in images], axis=0)
    return stacked_images

def visualize_volumes(volume_data):
    """根据提供的三维体数据创建VTK可视化"""
    
    # print(volume_data[300])
    # data = volume_data[300]
    # print(data.shape)
    # print(data.max())
    # print(data.min())
    # print(data[300,100])
    dataImporter = vtk.vtkImageImport()
    data_string = volume_data.tobytes()  # 使用tostring而不是tobytes是为了与旧版本的VTK兼容
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(1)

    # 设置数据维度信息
    z_dim, y_dim, x_dim = volume_data.shape
    dataImporter.SetDataExtent(0, x_dim-1, 0, y_dim-1, 0, z_dim-1)
    dataImporter.SetWholeExtent(0, x_dim-1, 0, y_dim-1, 0, z_dim-1)
    
    # set the 'spacing' between voxels, if necessary
    # dataImporter.SetDataSpacing(spacingX, spacingY, spacingZ)

    # Thresholding: 基于像素值阈值转换体数据为二值数据
    threshold = vtk.vtkImageThreshold ()
    threshold.SetInputConnection(dataImporter.GetOutputPort())
    threshold.ThresholdByLower(0)  # assumming 0 as the air/background value
    threshold.ReplaceInOn()
    threshold.SetInValue(0)       # this sets all values below 1 to 0, effectively binarizing
    threshold.ReplaceOutOn()
    threshold.SetOutValue(1)
    threshold.Update()

    volumeMapper = vtk.vtkSmartVolumeMapper()
    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.ShadeOn()
    volumeProperty.SetInterpolationTypeToLinear()

    volumeColor = vtk.vtkColorTransferFunction()
    # 根据实际需求调整颜色映射
    volumeColor.AddRGBPoint(0, 0.0, 0.0, 0.0) # 假设0是背景值，映射为黑色
    volumeColor.AddRGBPoint(255, 1.0, 1.0, 1.0) # 假设255是最高的兴趣值，映射为白色

    volumeScalarOpacity = vtk.vtkPiecewiseFunction()
    # 根据实际需求调整不透明度映射
    volumeScalarOpacity.AddPoint(0, 0.0) # 黑色，完全透明
    volumeScalarOpacity.AddPoint(255, 1.0) # 白色，完全不透明
    
    volumeProperty.SetColor(volumeColor)
    volumeProperty.SetScalarOpacity(volumeScalarOpacity)
    
    volumeActor = vtk.vtkVolume()
    volumeActor.SetMapper(volumeMapper)
    volumeActor.SetProperty(volumeProperty)

    # 创建渲染器、渲染窗口和交互式渲染窗口
    renderer = vtk.vtkRenderer()
    renderWin = vtk.vtkRenderWindow()
    renderWin.AddRenderer(renderer)

    renderInteractor = vtk.vtkRenderWindowInteractor()
    renderInteractor.SetRenderWindow(renderWin)

    # 添加Actor到渲染器并设置背景颜色
    renderer.AddVolume(volumeActor)
    renderer.SetBackground(0.0, 0.0, 0.0)

    # 设置初始相机位置
    renderer.ResetCamera()

    # 开始交互
    renderWin.Render()
    renderInteractor.Initialize()
    renderInteractor.Start()

def main(folder_path):
    volume_data = read_images(folder_path)
    print('read images done')
    print(volume_data[300:301])
    data = volume_data[300:600]

    print(volume_data.shape)
    print(data.max())
    print(data.min())
    visualize_volumes(volume_data)
    print('visualize done')

# if __name__ == "__main__":
#     folder_path = 'E:/CSworks/kaggle_blood_vessel/kaggle/input/blood-vessel-segmentation/train/kidney_2/labels'  # Replace with your folder path
#     main(folder_path)


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time

# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
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





def main(rle_csv_path):

    height, width = 1041, 1511  
    # 将2D图像堆叠成3D数据体
    volume_data = read_rle_from_path(rle_csv_path, height, width)
    volume_data = volume_data * 255
    print(volume_data.shape)
    # 可视化体数据
    visualize_volumes(volume_data)
    print('Visualize done')

if __name__ == "__main__":
    #'./data/predictions/prediction2023-12-13-22-29-30.csv'
    rle_csv_path = './data/predictions/prediction2023-12-13-22-29-30.csv'
    main(rle_csv_path)