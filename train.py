import os
import time
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
from train_utils import (train_one_epoch, evaluate, init_distributed_mode, save_on_master, mkdir,
                         create_lr_scheduler)
from torch.utils.data.distributed import DistributedSampler
from utils.PCLT_dataset import prepare_PETCT_dataset
import torch.distributed as dist
from models.builder import EncoderDecoder as segmodel
from utils.init_func import init_weight, group_weight
from utils.logger import get_logger
import argparse
import shutil
from easydict import EasyDict as edict
from utils.plot_utils import save_and_plot_history
from models.encoders.vmamba import selective_scan_flop_jit
from fvcore.nn import FlopCountAnalysis, parameter_count库 ---
import cv2
from medpy.metric.binary import hd95
from tqdm import tqdm

parser = argparse.ArgumentParser()
logger = get_logger()
C = edict()
config = C
C.backbone = 'sigma_tiny' # sigma_tiny / sigma_small / sigma_base
C.pretrained_model = None # do not need to change
C.decoder = 'MambaDecoder' # 'MLPDecoder'
C.decoder_embed_dim = 512
C.image_height = 512
C.image_width = 512
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.num_classes = 1

def params_count(model):
    return np.sum([p.numel() for p in model.parameters() if p.requires_grad]).item()

def create_model():
    model = segmodel(cfg=config, norm_layer=nn.BatchNorm2d)
    parameter = params_count(model)
    print('parameter:', parameter)
    return model

def processImage(pet_path, ct_path, mask_path, model, outPath, image_id, total_tp, total_fp, total_fn, total_tn, hd95_list, device):
    model.eval()
    
    # 读取图片
    pet_img = cv2.imread(pet_path, cv2.IMREAD_GRAYSCALE)
    ct_img = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)

    if pet_img is None or ct_img is None:
        return total_tp, total_fp, total_fn, total_tn, hd95_list

    # 数据预处理
    ct_img = np.expand_dims(ct_img, axis=2)
    pet_img = np.expand_dims(pet_img, axis=2)
    ct_img = np.array(ct_img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    pet_img = np.array(pet_img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    
    pet_img = pet_img[np.newaxis, :, :, :]
    ct_img = ct_img[np.newaxis, :, :, :]
    
    pet_img = torch.tensor(pet_img).to(device).repeat(1, 3, 1, 1)
    ct_img = torch.tensor(ct_img).to(device).repeat(1, 3, 1, 1)
    # --- [新增拦截代码] ---
    import os
    exp_mode = os.environ.get('EXP_MODE', 'normal')
    if exp_mode == 'pet_only':
        ct_img = pet_img 
    elif exp_mode == 'ct_only':
        pet_img = ct_img 
    # --------------------
    # 推理
    with torch.no_grad():
        pred = torch.sigmoid(model.forward(ct_img, pet_img))
    
    pred = pred.cpu().numpy()
    pred = np.squeeze(pred, axis=0).transpose(1, 2, 0) * 255.0
    pred = pred.astype(np.uint8)

    # 读取 Mask
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None: 
        mask_img = np.zeros_like(pred)
        
    gt_bin = (mask_img > 127).astype(np.uint8)
    pred_bin = (pred[:, :, 0] > 127).astype(np.uint8)

    # 计算 TP/FP/FN/TN
    tp = np.sum((gt_bin == 1) & (pred_bin == 1))
    fp = np.sum((gt_bin == 0) & (pred_bin == 1))
    fn = np.sum((gt_bin == 1) & (pred_bin == 0))
    tn = np.sum((gt_bin == 0) & (pred_bin == 0))

    total_tp += tp
    total_fp += fp
    total_fn += fn
    total_tn += tn

    max_dist = np.sqrt(gt_bin.shape[0]**2 + gt_bin.shape[1]**2)
    
    if np.sum(gt_bin) == 0:
        if np.sum(pred_bin) == 0:
            hd = 0.0  
        else:
            hd = max_dist 
    else:
        if np.sum(pred_bin) == 0:
            hd = max_dist 
        else:
            try:
                hd = hd95(pred_bin, gt_bin)
            except RuntimeError:
                hd = max_dist

    hd95_list.append(hd)

    return total_tp, total_fp, total_fn, total_tn, hd95_list

def evaluate_like_pred(model, args, device):
    val_list_path = os.path.join(args.split_train_val_test, 'val.txt')
    if not os.path.exists(val_list_path):
        val_list_path = os.path.join(args.split_train_val_test, 'test.txt')

    if not os.path.exists(val_list_path):
        print(f"⚠️ [Warning] Validation list not found: {val_list_path}. Skipping evaluation.")
        return 0.0, 0.0, 0.0, 0.0
        
    with open(val_list_path, 'r') as f:
        test_list =[x.strip() for x in f if x.strip()]
        
    outPath = "./results_eval/"
    if not os.path.exists(outPath):
        os.makedirs(outPath)
        
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    hd95_list =[]
    
    print(f"\nEvaluating like pred.txt on {len(test_list)} images...")
    for image_id in tqdm(test_list, desc="Evaluating"):
        pet_path = os.path.join(args.img_dir, f"{image_id.split('_')[0]}/{image_id}_PET.png")
        ct_path = os.path.join(args.img_dir, f"{image_id.split('_')[0]}/{image_id}_CT.png")
        mask_path = os.path.join(args.img_dir, f"{image_id.split('_')[0]}/{image_id}_mask.png")
        
        total_tp, total_fp, total_fn, total_tn, hd95_list = processImage(
            pet_path, ct_path, mask_path, model, outPath, image_id, 
            total_tp, total_fp, total_fn, total_tn, hd95_list, device
        )
        
    denominator_iou = total_tp + total_fp + total_fn
    iou = 1.0 if denominator_iou == 0 else total_tp / denominator_iou
    
    denominator_dice = 2 * total_tp + total_fp + total_fn
    dice = 1.0 if denominator_dice == 0 else (2 * total_tp) / denominator_dice
    
    # 防止除0
    acc_tp = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    acc_tn = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 1.0
    acc = np.mean([acc_tp, acc_tn])
    
    mean_hd95 = np.mean(hd95_list) if len(hd95_list) > 0 else 0.0
    
    return iou, dice, acc, mean_hd95
# =========================================================================

def main(args):
    if args.distributed:
        init_distributed_mode(args)
        print(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_dataset, val_dataset = prepare_PETCT_dataset(args, transforms=True)

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True,
        collate_fn=train_dataset.collate_fn, drop_last=True)
    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                        batch_size=1,
                                        sampler=val_sampler,
                                        num_workers=num_workers,
                                        pin_memory=True,
                                        shuffle=False,
                                        drop_last=False,
                                        collate_fn=val_dataset.collate_fn,
                                        )
    model = create_model()
    model.to(device)

    # --- 计算 FLOPs 和 参数量 ---
    if not args.distributed or (args.distributed and dist.get_rank() == 0):
        print("--------------------------------------------------")
        print("Calculating FLOPs and Params...")
        dummy_ct = torch.randn(1, 3, 512, 512).to(device)
        dummy_pet = torch.randn(1, 3, 512, 512).to(device)
        model_to_profile = model.module if hasattr(model, 'module') else model
        model_to_profile.eval()

        supported_ops = {
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanNRow": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit,
        }

        try:
            flops = FlopCountAnalysis(model_to_profile, (dummy_ct, dummy_pet))
            flops.set_op_handle(**supported_ops)
            flops_total = flops.total()
            params_total = parameter_count(model_to_profile)[""]
            
            print(f"Input Shape: (1, 3, 512, 512) x 2")
            print(f"GFLOPs: {flops_total / 1e9:.2f} G")
            print(f"Params: {params_total / 1e6:.2f} M")
            print("--------------------------------------------------")
        except Exception as e:
            print(f"⚠️ Error calculating FLOPs: {e}")
            print("Tip: Make sure fvcore is installed and input shape matches model definition.")
    
    model_without_ddp = model

    if args.distributed:
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        
    params_list =[]
    params_list = group_weight(params_list, model, nn.BatchNorm2d, args.lr)
    params_to_optimize =[p for p in model.parameters() if p.requires_grad]
    
    if args.optimizer =='Adam':
        optimizer = torch.optim.Adam(params=params_to_optimize, lr=args.lr)
    elif args.optimizer =='AdamW':
        optimizer = torch.optim.AdamW(params_list, args.lr, betas=(0.9, 0.999), eps=args.eps, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr= args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            params_to_optimize,
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=args.warm_up, warmup_epochs=args.warm_up_epoch)

    # --- [加载权重的逻辑修改部分] ---
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=torch.device('cuda'))
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
        if 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print(f"Resumed LR Scheduler from Epoch {checkpoint['epoch']}")
        else:
            print("Warning: No scheduler state in checkpoint. Manually stepping scheduler.")
            for _ in range(args.start_epoch):
                lr_scheduler.step()
    
    elif args.finetune:
        print(f"🌟 Starting Fine-tuning! Loading ONLY model weights from {args.finetune} ...")
        checkpoint = torch.load(args.finetune, map_location=torch.device('cuda'))
        # 兼容包含 'model' 键的字典，或者直接是 state_dict 的情况
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model_without_ddp.load_state_dict(state_dict, strict=True)
        print("✅ Pre-trained model weights loaded successfully. Epoch starts from 0.")
    # -------------------------------

    best_dice = 0  # <--- 初始化最高 Dice
    start_time = time.time()

    log_history = {
        'train_step_loss':[], 
        'train_epoch_loss': [], 
        'val_dice':[],
        'val_iou':[],
        'val_hd95':[]
    }

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        mean_loss, lr, current_step_losses = train_one_epoch(model, optimizer, train_loader, device, epoch, lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)
        
        # 判断是否是主进程 (防止非DDP环境报错)
        is_main_process = not args.distributed or (args.distributed and dist.get_rank() == 0)

        # ------------------ 只在主进程上做单卡验证和保存 ------------------
        if is_main_process:
            log_history['train_step_loss'].extend(current_step_losses)
            log_history['train_epoch_loss'].append(mean_loss)

            # 验证
            iou, dice, acc, mean_hd95 = evaluate_like_pred(model_without_ddp, args, device)
            
            # 判断是否刷新记录
            is_best_dice = dice > best_dice
            if is_best_dice:
                best_dice = dice
            
            result_log = (f"Epoch: {epoch}\n"
                      f"IoU: {iou:.6f}\n"
                      f"Dice: {dice:.6f}\n"
                      f"Acc: {acc:.6f}\n"
                      f"HD95: {mean_hd95:.3f}\n"
                      f"Best Dice: {best_dice:.6f}\n" 
                      f"---------------------\n")
            print(result_log) 

            # 输出每轮指标到 log txt 文件中
            save_path = os.path.join(WEIGHT_SAVE_DIR, results_file) 
            with open(save_path, "a") as f: 
                f.write(result_log)
                
            log_history['val_iou'].append(iou)
            log_history['val_dice'].append(dice)
            log_history['val_hd95'].append(mean_hd95)

            # 保存权重文件构建
            save_file = {"model": model_without_ddp.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": lr_scheduler.state_dict(),
                         "epoch": epoch,
                         "args": args}
            if args.amp:
                save_file["scaler"] = scaler.state_dict()

            # 保存当前 Epoch 权重
            torch.save(save_file, os.path.join(WEIGHT_SAVE_DIR, f"model_epoch_{epoch}.pth"))
            
            # 保存最佳权重：以 Dice 为准
            if is_best_dice:
                print(f"🌟 New Best Dice: {best_dice:.6f}, Updating best_model.pth...")
                torch.save(save_file, os.path.join(WEIGHT_SAVE_DIR, "best_model.pth"))

        # 等待主进程完成，再进入下一轮
        if args.distributed:
            dist.barrier()

    if not args.distributed or (args.distributed and dist.get_rank() == 0):
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("training time {}".format(total_time_str))
        print(f"Final best_metric: Dice: {best_dice:.6f}") # <--- 最后打印

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--data-path", default="./", help="DRIVE root")
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch_size", default=4, type=int)
    parser.add_argument("--epochs", default=30, type=int, metavar="N", help="number of total epochs to train")
    parser.add_argument('--eps', default=1e-8, type=float, help='adam eps')
    parser.add_argument('--lr', default=0.00006, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float, metavar='W', dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    
    # --- [参数修改部分] ---
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--finetune', default='./save_model/best_model/DPIN.pth', type=str, help='fine-tune from checkpoint (loads ONLY model weights)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    parser.add_argument("--amp", default=True, type=bool, help="Use torch.cuda.amp for mixed precision training")

    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('-j', '--workers', default=20, type=int, metavar='N', help='number of data loading workers (default: 4)')

    # 替换为你生成的新数据集路径
    parser.add_argument('--img_dir', type=str, default="./dataset/autoPET_lung/")
    parser.add_argument('--split_train_val_test', type=str, default='./dataset/autoPET_lung/')
    # ----------------------

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--weight_save_dir', type=str, default='./save_model')
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument("--distributed", default=True, type=bool)
    parser.add_argument('--sync_bn', type=bool, default=True, help='whether using SyncBatchNorm')

    parser.add_argument("--warm_up", default=False, type=bool)
    parser.add_argument("--warm_up_epoch", default=5, type=int)
    parser.add_argument("--wandb", default=False, type=bool)
    args = parser.parse_args()

    return args

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark =False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    WEIGHT_SAVE_DIR = os.path.join(args.weight_save_dir,
                                   time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime()) )
    os.makedirs(WEIGHT_SAVE_DIR, exist_ok=True)
    main(args)