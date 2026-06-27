import torch
import torch.nn as nn
import cv2
import numpy as np
from torch.nn import functional as F

# =========================================================================
# [新增 & 强烈推荐] 终极提分组合：Dice Loss + Focal Loss
# 完美解决了 Tensor 与 Scalar 相加导致的维度广播问题，
# 并且引入了 Focal Loss 极大增强对困难样本（小病灶/边界模糊区域）的挖掘能力。
# =========================================================================
class Improved_Dice_Focal_Loss(nn.Module):
    #原版def __init__(self, alpha=0.25, gamma=2.0, smooth=1.0):
    #新版
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1e-5):
        super(Improved_Dice_Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # 1. 尺寸对齐 (防止网络输出与 Mask 尺寸不匹配)
        if y_true.shape[2:] != y_pred.shape[2:]:
            y_true = F.interpolate(y_true.float(), size=y_pred.shape[2:], mode='nearest')
            
        y_true = y_true.float()
        
        # 2. 计算 Dice Loss (按 Batch 和 Channel 展平计算，梯度更稳定)
        pred_sig = torch.sigmoid(y_pred)
        
        # 展平 H 和 W 维度: shape 变成 (B, C, H*W)
        pred_flat = pred_sig.view(pred_sig.size(0), pred_sig.size(1), -1)
        true_flat = y_true.view(y_true.size(0), y_true.size(1), -1)
        
        intersection = (pred_flat * true_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + true_flat.sum(dim=2)
        
        # 得到每个 batch 样本的 dice，然后求全局平均
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        # 3. 计算 Focal Loss (替代普通的 BCE，专注困难像素)
        # reduction='none' 获取每个像素的 loss，尺寸保持 (B, C, H, W)
        bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
        
        # 计算 pt (预测正确的概率)
        pt = torch.exp(-bce_loss) 
        
        # Focal Loss 公式
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        focal_loss = focal_loss.mean() # 最终求全局平均标量

        return dice_loss + focal_loss


# =========================================================================
# 以下为你原来的代码（保留并作了底层修复，如果你想退回原版可以继续用）
# =========================================================================
class dice_bce_loss(nn.Module):
    def __init__(self, batch=False):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        # [修复] 直接使用官方带有 mean 归约的 Loss，解决维度冲突
        self.bce_loss = nn.BCEWithLogitsLoss() 

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 1.0
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(dim=(1,2,3))
            j = y_pred.sum(dim=(1,2,3))
            intersection = (y_true * y_pred).sum(dim=(1,2,3))
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def dice_metric(self, y_true, y_pred):
        dice = self.soft_dice_coeff(y_true, y_pred)
        return dice

    def __call__(self, y_true, y_pred):
        # 尺寸对齐
        if y_true.shape[2:] != y_pred.shape[2:]:
            y_true = F.interpolate(y_true.float(), size=y_pred.shape[2:], mode='nearest')
        y_true = y_true.float()

        # [修复] loss1 现在是一个 Scalar 标量了，可以安全地和 loss2(标量) 相加
        loss1 = self.bce_loss(y_pred, y_true)
        
        y_pred_sig = torch.sigmoid(y_pred)
        loss2 = self.soft_dice_loss(y_true, y_pred_sig)
        
        return loss1 + loss2


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        eps = 1e-7
        loss_y1 = -1 * self.alpha * \
                  torch.pow((1 - preds) , self.gamma) * \
                  torch.log(preds + eps) * labels
        loss_y0 = -1 * (1 - self.alpha) * torch.pow(preds,
                                                    self.gamma) * torch.log(1 - preds + eps) * (1 - labels)
        loss = loss_y0 + loss_y1
        return torch.mean(loss)