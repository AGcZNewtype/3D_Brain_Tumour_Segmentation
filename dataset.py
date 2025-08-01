import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import random

MODALITIES = ['flair', 't1', 't1ce', 't2']
"""
    每个案例文件夹中都有五个文件，除去标签文件，其他的四个均为不同处理结果的MRI模态图像，所以其实总共的案例个数是总共训练集数量//4
        _flair.nii	FLAIR模态
        _t1.nii	    T1模态
        _t1ce.nii	T1对比增强模态
        _t2.nii	    T2模态
        _seg.nii	标签（分割）
"""

class BraTSDataset(Dataset):
    def __init__(self, root_dir, patch_size=(128, 128, 128), mode='train', sample_num=4, min_tumor_voxels=100, augment=False):
        """
        root_dir: 数据根目录
        patch_size: 裁剪patch大小
        mode: 'train'或'test'，后续可以针对测试实现不同策略
        sample_num: 每个病人采样patch数量
        min_tumor_voxels: 采样patch时肿瘤体素最小数量阈值
        augment: 是否启用简单数据增强
        """
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.sample_num = sample_num
        self.mode = mode
        self.min_tumor_voxels = min_tumor_voxels
        self.augment = augment

        self.patient_dirs = sorted([
            os.path.join(root_dir, d) for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) #判断每个案例文件夹名称是否正确，正确就添加到全部列表中用于选择
        ])
        # print(len(self.patient_dirs),type(self.patient_dirs))

    def __len__(self):
        return len(self.patient_dirs) * self.sample_num #完整数据集长度，就是数据集文件夹中的有效案例数，并且每个案例中有四个

    def __getitem__(self, idx):
        patient_idx = idx // self.sample_num
        patient_dir = self.patient_dirs[patient_idx]   #由于上述在数据集描述中提到的问题，所以实际上病人（案例）的数量是总共数据量/4

        #处理四个模态的图像
        images = []     #创建图像数组
        for modality in MODALITIES: #四个模态图像
            img_path = os.path.join(patient_dir, f"{os.path.basename(patient_dir)}_{modality}.nii") #读取每个模态图像
            try:
                img = nib.load(img_path).get_fdata() #使用nibabel读取图像并转换成numpy的浮点数组
            except Exception as e:
                print(f"Warning: failed to load {img_path}: {e}")
                img = np.zeros(self.patch_size)  # 失败时用0填充避免崩溃
            img = self.zscore_normalize(img) #进行标准化
            images.append(img) #把四个模态图像的数组，加入数组中

        image_np = np.stack(images, axis=0)  # [4, H, W, D] #np.stack()有点像list.append(list()),相当于往np数组里加入一个相同长度的数组，并增加一个维度，这里面的4就是代表四个模态的维度

        #处理标签图像
        seg_path = os.path.join(patient_dir, f"{os.path.basename(patient_dir)}_seg.nii") #获取标签
        try:
            seg = nib.load(seg_path).get_fdata()#将标签数据转化为numpy浮点数组
        except Exception as e:
            print(f"Warning: failed to load {seg_path}: {e}")
            seg = np.zeros(self.patch_size)

        label_np = seg.astype(np.uint8)      # [H, W, D] 长度，宽度，深度(切片高度),把标签里的数据转换成unit8整数类型(原本可能是float之类的)，方便后续训练和loss计算。

        # 随机裁剪patch，重点保证肿瘤区域覆盖
        image_np, label_np = self.random_crop(image_np, label_np)#因为默认的图像太大了，训练起来很麻烦，所以切成128的三维图像，然后尽可能保留肿瘤区域

        # 简单数据增强示例，随机翻转
        if self.augment and self.mode == 'train':
            if random.random() > 0.5:
                image_np = np.flip(image_np, axis=2)  # 随机左右翻转
                label_np = np.flip(label_np, axis=1)

        return torch.tensor(image_np.copy(), dtype=torch.float32), torch.tensor(label_np.copy(), dtype=torch.long)

    def zscore_normalize(self, img): #zero score标准差，就是把非零区域的像素值标准化为“均值为 0、标准差为 1”的分布，方便模型更快收敛。
        mask = img > 0      #使用np数组(img)进行筛选，这里img>0相当于是遍历img中的数字，然后和0做对比，比0大的在mask中就是true，小的就是false
        if np.any(mask):    #只要mask中含有ture的就进行标准差操作
            mean = img[mask].mean()     #通过mask筛选非0区域，并计算均值和标准差
            std = img[mask].std()
            img_copy = img.copy()        # 避免修改原数组，用copy复制出来一份，然后修改复制的图像数组，这样原图像不会有问题
            img_copy[mask] = (img_copy[mask] - mean) / (std + 1e-8)  #然后做标准差，把非0的部分减去平均值（这样数据就变成以0为中心然后左右分布），再除以标准差就能让数据分布更紧凑一点
        return img_copy

    def random_crop(self, image, label):
        c, h, w, d = image.shape   #读取每个维度图片的尺寸-c：维度，h：高度，w：宽度，d：切片层数
        ph, pw, pd = self.patch_size  #裁剪成128*128*128

        valid_h = max(h - ph, 1)#如果图像三维比128大，就进行切片处理，然后这个就是不越界的区域，简单理解就是从240长度里可以切掉112就是128
        valid_w = max(w - pw, 1)
        valid_d = max(d - pd, 1)
        # print(image.shape)
        # print(valid_h, valid_w, valid_d)

        for _ in range(10):
            hh = random.randint(0, valid_h)#范围是从0到112，选个随机值
            ww = random.randint(0, valid_w)#0-112
            dd = random.randint(0, valid_d)#0-27
            """
            切除标签中的区域，从随机点开始128的区域，举个例子，深度一共是155，那我能切除的区域就是从0-27(超过27就没有意义了)
            然后我从0-27随机选择，比如我选择20，那我选择出的区域就是[20,148]，如果肿瘤标签确实在这个范围内，那我就可以保留
            相当于我把没用的一些图像给移除掉了，下面这一段就是这个作用
            """
            patch_label = label[hh:hh+ph, ww:ww+pw, dd:dd+pd]
            if np.sum(patch_label > 0) > self.min_tumor_voxels:#如果在我选择的这个切片内，肿瘤样本总和在100个以上，我就可以保留这个切片
                break
            #不过只有10次机会，所以如果没能完全满足最小肿瘤数量的话也得继续

        image_patch = image[:, hh:hh+ph, ww:ww+pw, dd:dd+pd]
        label_patch = label[hh:hh+ph, ww:ww+pw, dd:dd+pd]
        return image_patch, label_patch


if __name__ == "__main__":
    ds = BraTSDataset("E:/Workspace/PycharmProjects/MRI/datasets/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData", augment=True)
    img, mask = ds[0]
    print(img.shape)   # torch.Size([4, 128, 128, 128])
    print(mask.shape)  # torch.Size([128, 128, 128])