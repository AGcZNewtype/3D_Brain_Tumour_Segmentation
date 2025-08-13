# utils.py
import torch
import numpy as np
import os
import random
import nibabel as nib
from torch.utils.tensorboard import SummaryWriter


def save_nifti(output_tensor, reference_path, save_path):
    """
    保存分割结果为 NIfTI 文件
    output_tensor: shape (H, W, D)
    """
    output_np = output_tensor.cpu().numpy().astype(np.uint8)
    reference_img = nib.load(reference_path)
    new_img = nib.Nifti1Image(output_np, affine=reference_img.affine, header=reference_img.header)
    nib.save(new_img, save_path)


def set_seed(seed=42):
    """
    设置随机种子，确保结果可复现
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, path="runs\\unet3\\dcheckpoint.pth"):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, path)


def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']


class TensorboardLogger:
    """
    TensorBoard日志管理类，自动记录loss和dice
    """

    def __init__(self, log_dir="logs"):
        self.writer = SummaryWriter(log_dir)

    def log_loss(self, loss, step):
        self.writer.add_scalar("Loss/train", loss, step)

    def log_dice(self, dice_scores, step):
        for i, dice in enumerate(dice_scores):
            self.writer.add_scalar(f"Dice/class_{i}", dice, step)

    def close(self):
        self.writer.close()
