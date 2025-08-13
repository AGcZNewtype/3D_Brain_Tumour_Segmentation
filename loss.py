import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        assert targets.dtype == torch.long, f"Targets must be long, got {targets.dtype}"
        inputs = torch.softmax(inputs, dim=1)
        targets = nn.functional.one_hot(targets, num_classes=inputs.shape[1])
        targets = targets.permute(0, 4, 1, 2, 3).float()

        intersection = torch.sum(inputs * targets)
        union = torch.sum(inputs + targets)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


def dice_score(preds, targets, num_classes=4):
    """
    计算每个类别的 Dice Score
    preds: 模型输出 logits，shape: (B, C, H, W, D)
    targets: 真实标签，shape: (B, H, W, D)
    """
    preds = torch.argmax(torch.softmax(preds, dim=1), dim=1)
    preds = preds.view(-1)
    targets = targets.view(-1)

    scores = []
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = targets == cls
        intersection = (pred_inds & target_inds).float().sum()
        union = pred_inds.float().sum() + target_inds.float().sum()
        dice = (2. * intersection) / (union + 1e-5)
        scores.append(dice.item())
    return scores
