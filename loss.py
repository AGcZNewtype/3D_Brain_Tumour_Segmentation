import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        targets = nn.functional.one_hot(targets, num_classes=inputs.shape[1])
        targets = targets.permute(0, 4, 1, 2, 3).float()

        intersection = torch.sum(inputs * targets)
        union = torch.sum(inputs + targets)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice
