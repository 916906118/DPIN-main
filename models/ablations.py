import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os

class DualResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stream_rgb = timm.create_model('resnet50', pretrained=True, features_only=True)
        self.stream_x = timm.create_model('resnet50', pretrained=True, features_only=True)

    def forward(self, rgb, x):
        feats_rgb = self.stream_rgb(rgb)
        feats_x = self.stream_x(x)
        out = []
        for i in range(1, 5):
            out.append(feats_rgb[i] + feats_x[i])
        return out

class DualSwin(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 恢复原生 224x224 初始化
        self.stream_rgb = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)
        self.stream_x = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)

        # 2. 加载权重
        ckpt_path = './pretrained/swin_tiny_patch4_window7_224.pth'
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
            self.stream_rgb.load_state_dict(state_dict, strict=False)
            self.stream_x.load_state_dict(state_dict, strict=False)
            print(f"✅ 成功解包并加载了 Swin Transformer 预训练权重！")

        # 3. [关键修复] 使用 Hook 机制，在降采样之前截获完美通道数的特征
        self.feats_rgb = []
        self.feats_x = []

        def get_hook(lst):
            def hook(module, input, output):
                # 获取特征 (兼容不同版本的返回格式)
                feat = output[0] if isinstance(output, tuple) else output
                lst.append(feat)
            return hook

        # 挂载 Hook 到每个阶段的最后一个 Block (降采样之前)
        for layer in self.stream_rgb.layers:
            layer.blocks[-1].register_forward_hook(get_hook(self.feats_rgb))
        for layer in self.stream_x.layers:
            layer.blocks[-1].register_forward_hook(get_hook(self.feats_x))

    def forward(self, rgb, x):
        # 1. 每次前向传播前，清空 Hook 拦截列表
        self.feats_rgb.clear()
        self.feats_x.clear()

        # 2. 将输入缩小到 Swin 期望的 224x224 进行无错前向传播
        rgb_224 = F.interpolate(rgb, size=(224, 224), mode='bilinear', align_corners=False)
        x_224 = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        _ = self.stream_rgb(rgb_224)
        _ = self.stream_x(x_224)
        
        out = []
        # Decoder 期望的 4 层空间尺寸
        target_sizes = [128, 64, 32, 16]
        
        # 3. 处理收集到的完美特征 (通道数保证是 96, 192, 384, 768)
        for i in range(4):
            f_rgb = self.feats_rgb[i]
            f_x = self.feats_x[i]
            
            B = f_rgb.shape[0]
            H_feat = 224 // (4 * (2 ** i))
            W_feat = 224 // (4 * (2 ** i))
            
            # 将 (B, L, C) 恢复成 (B, C, H, W)
            if f_rgb.ndim == 3:
                f_rgb = f_rgb.transpose(1, 2).reshape(B, -1, H_feat, W_feat).contiguous()
                f_x = f_x.transpose(1, 2).reshape(B, -1, H_feat, W_feat).contiguous()
            elif f_rgb.ndim == 4:
                f_rgb = f_rgb[:, :H_feat, :W_feat, :].permute(0, 3, 1, 2).contiguous()
                f_x = f_x[:, :H_feat, :W_feat, :].permute(0, 3, 1, 2).contiguous()

            # 融合并放大到 Decoder 所需尺寸
            f_fused = f_rgb + f_x
            f_fused = F.interpolate(f_fused, size=(target_sizes[i], target_sizes[i]), mode='bilinear', align_corners=False)
            out.append(f_fused)
            
        return out