import matplotlib.pyplot as plt
import os
import json
import numpy as np

def save_and_plot_history(log_history, save_dir):
    """
    log_history: 字典，包含 'train_step_loss', 'val_dice', 'val_iou' 等
    save_dir: 保存路径
    """
    # 1. 保存原始数据为 JSON (防止画图挂了数据丢了)
    json_path = os.path.join(save_dir, 'log_history.json')
    try:
        with open(json_path, 'w') as f:
            json.dump(log_history, f)
    except Exception as e:
        print(f"Error saving json: {e}")

    # 2. 画图
    try:
        # --- 图 1: 详细的 Training Loss (每100步一个点) ---
        plt.figure(figsize=(10, 5))
        steps = np.arange(len(log_history['train_step_loss'])) * 100 # 假设每100步记录一次
        plt.plot(steps, log_history['train_step_loss'], label='Train Step Loss', alpha=0.6, linewidth=0.5)
        
        # 平滑曲线 (Moving Average)
        if len(log_history['train_step_loss']) > 10:
            window = 10
            smooth_loss = np.convolve(log_history['train_step_loss'], np.ones(window)/window, mode='valid')
            plt.plot(steps[window-1:], smooth_loss, label='Smoothed Loss', color='red', linewidth=1.5)
            
        plt.title('Training Loss per Step')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'loss_curve_steps.png'))
        plt.close()

        # --- 图 2: Epoch 级别的对比 (Train Avg Loss vs Val Dice) ---
        # 注意：通常 Train Loss 和 Val Dice 量纲不同，用双Y轴
        if len(log_history['val_dice']) > 0:
            fig, ax1 = plt.subplots(figsize=(10, 5))
            
            epochs = range(1, len(log_history['val_dice']) + 1)
            
            # 左轴：Train Epoch Loss
            color = 'tab:red'
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Train Loss (Avg)', color=color)
            ax1.plot(epochs, log_history['train_epoch_loss'], color=color, marker='o', label='Train Loss')
            ax1.tick_params(axis='y', labelcolor=color)
            
            # 右轴：Val Dice
            ax2 = ax1.twinx()  
            color = 'tab:blue'
            ax2.set_ylabel('Val Dice', color=color)  
            ax2.plot(epochs, log_history['val_dice'], color=color, marker='s', label='Val Dice')
            ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title('Training Loss vs Validation Dice')
            fig.tight_layout()  
            plt.savefig(os.path.join(save_dir, 'epoch_metrics.png'))
            plt.close()
            
    except Exception as e:
        print(f"Error plotting: {e}")