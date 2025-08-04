
'''
   下载相应数据集，但是我没用这个，这个下载下来是做过预处理的2D模态图像，我用的是.ni 3D模态图像
'''
# import kagglehub
# Download latest version
# path = kagglehub.dataset_download("awsaf49/brats2020-training-data")
#
# print("Path to dataset files:", path)


"""
   这一段也是由于我之前不知道，用kagglehub下载的，所以想要输出出来看看是什么结构
"""
# import h5py
# path = 'E:/Workspace/PycharmProjects/MRI/datasets/awsaf49/brats2020-training-data/versions/3/BraTS2020_training_data/content/data/volume_1_slice_0.h5'
# with h5py.File(path, 'r') as f:
#     print("Keys:", list(f.keys()))
#     for k in f.keys():
#         print(f[k].shape)


"""
    看一下fdata（读取了模态照片然后转化成numpy数组之后的样子）
"""

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# 标准化
# def zscore(img):
#     mask = img > 0
#     mean = img[mask].mean()
#     std = img[mask].std()
#     img[mask] = (img[mask] - mean) / (std + 1e-8)
#     return img
#
# def check_image(img_data,i,slice_idx,show=True):
#     # 查看形状
#     print("图像 shape:", img_data.shape)  # 比如 (240, 240, 155)
#
#     # 随便挑一个切片（Z轴方向第 80 层）
#     plt.imshow(img_data[:, :, slice_idx], cmap='gray')
#     plt.title(f"{i}Flair before {slice_idx} ")
#     plt.axis('off')
#     plt.show()



"""
    看一下标准化前后的图像区别
"""
    # if show:
    #     # 标准化前
    #     plt.hist(img_data[img_data !=0].ravel(), bins=100)
    #     plt.title(f"{i}before")
    #     plt.show()
    #     # print(f"before:{np.min(img_data[:,:,slice])}")
    #
    #
    #     img_z = zscore(img_data.copy())
    #
    #     # 标准化后
    #     plt.hist(img_z[img_z !=0].ravel(), bins=100)
    #     plt.title(f"{i}after")
    #     plt.show()
    #
    #     plt.imshow(img_z[:, :, slice_idx], cmap='gray')
    #     plt.title(f"{i}Flair after {slice_idx} ")
    #     plt.axis('off')
    #     plt.show()

"""
    这个是看不同案例中的相同模态在处理前后等的区别
"""
# for i in range(1,2):
#     slice_idx = 76
#     # 假设你有一个路径：某个模态，比如 flair
#     img_path = f'datasets/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_00{i}/BraTS20_Training_00{i}_flair.nii'
#
#     # 加载 nii 文件
#     nifti_img = nib.load(img_path)
#     img_data = nifti_img.get_fdata()  # 转成 numpy 的 float 数组
#     check_image(img_data,i,slice_idx)


"""
    这个是看相同案例中的不同模态在处理前后等的区别
"""
# MODALITIES = ['flair', 't1', 't1ce', 't2']
# i = 1
# slice_idx = 80
# for modality in MODALITIES:
#     # 假设你有一个路径：某个模态，比如 flair
#     img_path = f'datasets/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_00{i}/BraTS20_Training_00{i}_{modality}.nii'
#
#     # 加载 nii 文件
#     nifti_img = nib.load(img_path)
#     img_data = nifti_img.get_fdata()  # 转成 numpy 的 float 数组
#     check_image(img_data,i,slice_idx)


"""
    再看看不同层数的切片
"""

# for slice_idx in range (75,80):
#     i = 1
#     modality = 'seg'
#     # 假设你有一个路径：某个模态，比如 flair
#     img_path = f'datasets/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_00{i}/BraTS20_Training_00{i}_{modality}.nii'
#
#     # 加载 nii 文件
#     nifti_img = nib.load(img_path)
#     img_data = nifti_img.get_fdata()  # 转成 numpy 的 float 数组
#     check_image(img_data,i,slice_idx,show=False)


"""
输出模型结构
"""
import torch
from torch.utils.tensorboard import SummaryWriter
from model import UNet3D
writer = SummaryWriter("runs/unet3d")  # 可以自定义路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet3D(in_channels=4, out_channels=4).to(device)
# 假设你已经定义了 model，并且有一个样本输入 x
sample_input = torch.randn(1, 4, 128, 128, 128).to(device)  # 注意通道数要和模型输入一致
writer.add_graph(model, sample_input)
writer.close()
