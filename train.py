import torch
from torch.utils.data import DataLoader
from dataset import BraTSDataset
from model import UNet3D
from loss import DiceLoss
from utils import set_seed, dice_score, TensorboardLogger, save_checkpoint
import os
from tqdm import tqdm
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# 设置随机种子和设备
set_seed(2024)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 构造 3D 数据集（修改为你实际数据路径）
data_root = "E:/Workspace/PycharmProjects/MRI/datasets/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
train_dataset = BraTSDataset(root_dir=data_root, patch_size=(128, 128, 128), mode='train', sample_num=4,max_data=50)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# 初始化模型
model = UNet3D(in_channels=4, out_channels=4).to(device)
criterion = DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
logger = TensorboardLogger(log_dir="runs/tumor_seg_experiment")


global_step = 0
for epoch in range(1, 11):
    model.train()
    running_loss = 0

    for i, (images, labels) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        images, labels = images.to(device), labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 记录训练损失
        running_loss += loss.item()
        logger.log_loss(loss.item(), global_step)

        # 每N步记录Dice（例如每20步）
        if global_step % 20 == 0:
            model.eval()
            with torch.no_grad():
                dices = dice_score(outputs, labels)
                logger.log_dice(dices, global_step)
            model.train()

        global_step += 1

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch}/100] - Loss: {avg_loss:.4f}")

    # 保存模型
    if epoch % 10 == 0:
        save_checkpoint(model, optimizer, epoch, path=f"checkpoint_epoch{epoch}.pth")

logger.close()
