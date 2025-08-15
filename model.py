import torch
import torch.nn as nn


"""
    使用三维U-Net模型进行训练和测试
"""
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()  #调用nn.Module的构造函数
        #定义模型每层的结构块。使用nn.sequential顺序容器把下面的层按顺序“串起来”，forward里就可以一次性调用
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            #3D卷积层，核大小是3x3x3，步长1，至于后面的padding是因为为了保证可以卷积计算，然后将特征图像大小缩小
            nn.BatchNorm3d(out_channels),
            #使用3D 批归一化：对每个通道在 (N, D, H, W) 维上的统计量做标准化，稳定分布、加速收敛、允许更大学习率、缓解梯度消失/爆炸。
            nn.ReLU(inplace=True),
            #激活函数：ReLU(x)=max(0,x)，增加非线性表达能力。inplace=True 原地修改，省显存(仅限不经过多次反向传播的张量)
            nn.Conv3d(out_channels, out_channels, 3, padding=1),#这里就不再进行缩小了
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x) #将数据x传入layers，自动通过顺序容器经过上述定义的网络结构

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #定义下采样需要的一些结构
        self.enc1 = DoubleConv(in_channels, 32)#通过上面定义的卷积块对通道增加。
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.pool = nn.MaxPool3d(2)
        #定义3D最大池化层
        self.up1 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        # 定义反卷积方法，其中in_channel是输入的通道维度尺寸，out_channel是输出的通道维度尺寸，这里我们需要把吃花钱的图像也输入进来，所以这里直接减半
        self.dec1 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec2 = DoubleConv(64, 32)
        self.final = nn.Conv3d(32, out_channels, 1)#通过卷积将特征图通道转换为分类的4个相似矩阵

    def forward(self, x):
        e1 = self.enc1(x)#第一次卷积，从多模态维度转变为特征的通道(4-32)
        e2 = self.enc2(self.pool(e1))#池化缩小尺寸并进行第1次下采样
        e3 = self.enc3(self.pool(e2))#池化缩小尺寸并进行第2次下采样
        d1 = self.up1(e3)#进行反卷积上采样
        d1 = self.dec1(torch.cat([d1, e2], dim=1))#合并下采样和上采样图像，并进行反卷积
        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        out = self.final(d2)#将特征图映射为分类相似矩阵
        return out
