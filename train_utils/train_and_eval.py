import math
import torch
from torch.nn import functional as F
import train_utils.distributed_utils as utils

# --- [修改点 1] 将原来的 dice_bce_loss 替换为新写的 Improved_Dice_Focal_Loss ---
# 原代码: from train_utils.loss import dice_bce_loss
from train_utils.loss import Improved_Dice_Focal_Loss

import torch.distributed as dist
import numpy as np
from medpy.metric.binary import hd95


# --- [修改点 2] 修改 criterion 包装函数 ---
def criterion(inputs, target):
    # 原代码: loss_= dice_bce_loss()
    # 原代码: loss =loss_(target, inputs) 
    
    # 【替换为以下两行】
    loss_ = Improved_Dice_Focal_Loss()
    # ⚠️ 必须把 inputs 放在前面，target 放在后面！
    loss = loss_(inputs, target) 
    
    return loss

# ====== 下面的代码（soft_dice_coeff, evaluate, train_one_epoch等）统统保持原样，完全不用动 ======
def soft_dice_coeff( y_true, y_pred,batch=True):
    smooth = 1.0
    if batch:
        i = torch.sum(y_true)
        j = torch.sum(y_pred)
        intersection = torch.sum(y_true * y_pred)
    else:
        i = y_true.sum(1).sum(1).sum(1)
        j = y_pred.sum(1).sum(1).sum(1)
        intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
    score = (2. * intersection + smooth) / (i + j + smooth)
    return score.mean()
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

'''def evaluate(model, data_loader, device):
    model.eval()
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    hd95_list = []

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    iter_num = len(data_loader)
    print(iter_num)
    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            torch.cuda.empty_cache()
            images, targets = images.to(device), targets.to(device)
            pet = torch.unsqueeze(images[:, 0, :, :], 1).repeat(1, 3, 1, 1)
            ct = torch.unsqueeze(images[:, 1, :, :], 1).repeat(1, 3, 1, 1)
            output = model(ct, pet)
            output = torch.sigmoid(output)
            pred_full = output.clone().detach().cpu().numpy()
            target = targets.clone().cpu().numpy()
            pred_full[pred_full >= 0.5] = 1
            pred_full[pred_full < 0.5] = 0

            gt_bin = target.astype(np.uint8)
            pred_bin = pred_full.astype(np.uint8)

            tp = np.sum((gt_bin == 1) & (pred_bin == 1))
            fp = np.sum((gt_bin == 0) & (pred_bin == 1))
            fn = np.sum((gt_bin == 1) & (pred_bin == 0))
            tn = np.sum((gt_bin == 0) & (pred_bin == 0))

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

            if gt_bin.sum() == 0 and pred_bin.sum() == 0:
                hd = 0.0
            elif gt_bin.sum() == 0 or pred_bin.sum() == 0:
                pred_bin[..., 256, 256] = 255
                hd = hd95(pred_bin, gt_bin)
            else:
                try:
                    hd = hd95(pred_bin, gt_bin)
                except:
                    hd = np.sqrt(gt_bin.shape[0]**2 + gt_bin.shape[1]**2)
            hd95_list.append(hd)

        total_pixels = total_tp + total_fp + total_fn + total_tn
        # IoU
        denominator_iou = total_tp + total_fp + total_fn
        iou = 1.0 if denominator_iou == 0 else total_tp / denominator_iou
        # Dice/F1
        denominator_dice = 2 * total_tp + total_fp + total_fn
        dice = 1.0 if denominator_dice == 0 else (2 * total_tp) / denominator_dice
        # Accuracy
        acc = np.mean([total_tp/(total_tp+total_fn), total_tn/(total_tn+total_fp)])
        # HD95
        mean_hd95 = np.mean(hd95_list)

    return iou, dice, acc, mean_hd95
'''
def evaluate(model, data_loader, device):
    model.eval()
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    hd95_list = []

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            
            # 1. 数据准备
            pet = torch.unsqueeze(images[:, 0, :, :], 1).repeat(1, 3, 1, 1).to(device)
            ct = torch.unsqueeze(images[:, 1, :, :], 1).repeat(1, 3, 1, 1).to(device)
            targets = targets.to(device)

            # ==============================================================
            # [新增] 单模态消融实验拦截：强行将一种模态的数据覆盖给另一种模态
            # 保证网络参数量和计算图完全不改变，实现绝对公平的消融对比
            # ==============================================================
            import os
            exp_mode = os.environ.get('EXP_MODE', 'normal')
            if exp_mode == 'pet_only':
                ct = pet  # 用 PET 覆盖 CT，使得网络双流输入的都是 PET
            elif exp_mode == 'ct_only':
                pet = ct  # 用 CT 覆盖 PET，使得网络双流输入的都是 CT
            # ==============================================================

            # 2. 推理
            output = model(ct, pet)
            output = torch.sigmoid(output)
            
            # 3. 转 Numpy
            pred_full = output.clone().detach().cpu().numpy()
            target = targets.clone().cpu().numpy()
            
            # 二值化
            pred_full[pred_full >= 0.5] = 1
            pred_full[pred_full < 0.5] = 0

            gt_bin = target.astype(np.uint8)
            pred_bin = pred_full.astype(np.uint8)
            
            # 注意：evaluate 这里的 batch_size=1，所以 pred_bin shape 是 (1, 1, 512, 512)
            # 我们需要 squeeze 掉 batch 和 channel 维度，变成 (512, 512) 才能算 HD95
            if len(gt_bin.shape) == 4:
                gt_bin = gt_bin.squeeze()
            if len(pred_bin.shape) == 4:
                pred_bin = pred_bin.squeeze()

            # 4. 计算混淆矩阵 (TP/FP/FN/TN)
            tp = np.sum((gt_bin == 1) & (pred_bin == 1))
            fp = np.sum((gt_bin == 0) & (pred_bin == 1))
            fn = np.sum((gt_bin == 1) & (pred_bin == 0))
            tn = np.sum((gt_bin == 0) & (pred_bin == 0))

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

            # 5. HD95 防崩溃逻辑
            # 计算图像对角线作为最大惩罚距离
            max_dist = np.sqrt(gt_bin.shape[0]**2 + gt_bin.shape[1]**2)

            if np.sum(gt_bin) == 0:
                # 情况A: 真实切片无肿瘤 (健康肺)
                if np.sum(pred_bin) == 0:
                    hd = 0.0 # 预测也无 -> 完美
                else:
                    hd = max_dist # 预测有 -> 假阳性，给最大惩罚
            else:
                # 情况B: 真实切片有肿瘤
                if np.sum(pred_bin) == 0:
                    hd = max_dist # 预测无 -> 假阴性，给最大惩罚
                else:
                    try:
                        from medpy.metric.binary import hd95
                        hd = hd95(pred_bin, gt_bin)
                    except RuntimeError:
                        # 万一 medpy 内部几何计算出错
                        hd = max_dist
            
            hd95_list.append(hd)

    # 6. 汇总指标
    # IoU
    denominator_iou = total_tp + total_fp + total_fn
    iou = 1.0 if denominator_iou == 0 else total_tp / denominator_iou
    
    # Dice/F1
    denominator_dice = 2 * total_tp + total_fp + total_fn
    dice = 1.0 if denominator_dice == 0 else (2 * total_tp) / denominator_dice
    
    # Accuracy (Balanced Accuracy 推荐)
    # 防止分母为0
    sens = total_tp / (total_tp + total_fn + 1e-6)
    spec = total_tn / (total_tn + total_fp + 1e-6)
    acc = (sens + spec) / 2

    # HD95
    mean_hd95 = np.mean(hd95_list)

    return iou, dice, acc, mean_hd95
    
def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    # [新增] 用于记录这一轮所有的 step loss
    step_losses = []
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    #for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            pet = torch.unsqueeze(image[:, 0, :, :], 1).repeat(1,3,1,1)
            ct = torch.unsqueeze(image[:, 1, :, :], 1).repeat(1, 3, 1, 1)
            output = model(ct,pet)

            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        loss_value = loss.item()
        # [新增] 每 100 步记录一次 Loss (或者是 print_freq 步)
        # 建议和 print_freq 保持一致，或者是固定的 100
        if i % 10 == 0: # 或者是 if i % 100 == 0:
            step_losses.append(loss_value)

        metric_logger.update(loss=loss.item(), lr=lr)

    #return metric_logger.meters["loss"].global_avg, lr
    return metric_logger.meters["loss"].global_avg, lr_scheduler.get_last_lr()[0], step_losses

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-4):
    params_group = [{"params": [], "weight_decay": 0.},  # no decay
                    {"params": [], "weight_decay": weight_decay}]  # with decay

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            # bn:(weight,bias)  conv2d:(bias)  linear:(bias)
            params_group[0]["params"].append(param)  # no decay
        else:
            params_group[1]["params"].append(param)  # with decay

    return params_group
