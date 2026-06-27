import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from functools import partial
from models.encoders.local_vmamba.region_mamba import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import time
from utils.logger import get_logger
from models.encoders.vmamba import Backbone_VSSM, CrossMambaFusionBlock, ConcatMambaFusionBlock
from models.mamba_net_utils import ChannelRectifyModule

logger = get_logger()

class ProjectedSCMCA(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        
        self.norm_pet = nn.LayerNorm(dim)
        self.norm_ct = nn.LayerNorm(dim)

        self.q_conv = nn.Linear(dim, dim, bias=False)
        self.k_conv = nn.Linear(dim, dim, bias=False)
        self.v_conv = nn.Linear(dim, dim, bias=False)
        
        self.spatial_compress = nn.Linear(dim, 1)
        
        self.proj_out = nn.Linear(dim, dim)
        
        self.scale = dim ** -0.5
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)
        
        self.latest_s_map = None

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_pet, x_ct):
        B, C, H, W = x_pet.shape
        N = H * W

        pet_flat = x_pet.permute(0, 2, 3, 1).contiguous().view(B, N, C)
        ct_flat = x_ct.permute(0, 2, 3, 1).contiguous().view(B, N, C)
        
        pet_norm = self.norm_pet(pet_flat)
        ct_norm = self.norm_ct(ct_flat)

        #CPM

        q_ct = self.q_conv(ct_norm)
        k_pet = self.k_conv(pet_norm)
        v_pet = self.v_conv(pet_norm)
        
        q_pet = self.q_conv(pet_norm)
        k_ct = self.k_conv(ct_norm)
        v_ct = self.v_conv(ct_norm)
    
        with torch.cuda.amp.autocast(enabled=False):
            q_ct_f32 = q_ct.float()
            k_pet_f32 = k_pet.float()
            q_pet_f32 = q_pet.float()
            k_ct_f32 = k_ct.float()
            
            attn_pet2ct = torch.bmm(q_ct_f32.transpose(1, 2), k_pet_f32) * self.scale
            attn_ct2pet = torch.bmm(q_pet_f32.transpose(1, 2), k_ct_f32) * self.scale
            
            map_pet2ct = F.softmax(attn_pet2ct, dim=-1)
            map_ct2pet = F.softmax(attn_ct2pet, dim=-1)

        feat_c_pet2ct = torch.bmm(v_pet, map_pet2ct.type_as(v_pet).transpose(1, 2))
        feat_c_ct2pet = torch.bmm(v_ct, map_ct2pet.type_as(v_ct).transpose(1, 2))

        #SGM
        s_map_pet = torch.sigmoid(self.spatial_compress(pet_norm)) # (B, N, 1)
        s_map_ct = torch.sigmoid(self.spatial_compress(ct_norm))   # (B, N, 1)
        
        self.latest_s_map = s_map_pet.view(B, H, W, 1).detach().cpu()
        self.latest_c_map = map_pet2ct.detach().cpu()
        feat_s_pet2ct = ct_norm * s_map_pet
        feat_s_ct2pet = pet_norm * s_map_ct

        import os
        exp_mode = os.environ.get('EXP_MODE', 'normal')
        
        if exp_mode == 'no_cpm':
            # 消融 CPM
            enhanced = (feat_s_pet2ct + feat_s_ct2pet)
        elif exp_mode == 'no_sgm':
            # 消融 SGM
            enhanced = (feat_c_pet2ct + feat_c_ct2pet)
        else:
            enhanced = (feat_c_pet2ct + feat_c_ct2pet) + (feat_s_pet2ct + feat_s_ct2pet)

        enhanced = self.proj_out(enhanced) # (B, N, C)
        

        enhanced_map = enhanced.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        fused_out = x_ct + x_pet + self.gamma * self.dropout(enhanced_map)
        
        return fused_out

class RGBXTransformer(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 norm_layer=nn.LayerNorm,
                 depths=[2,2,27,2], 
                 dims=96,
                 pretrained=None,
                 mlp_ratio=4.0,
                 downsample_version='v1',
                 ape=False,
                 img_size=[512, 512],
                 patch_size=4,
                 drop_path_rate=0.2,
                 **kwargs):
        super().__init__()

        self.ape = ape
        self.vssm = Backbone_VSSM(
            pretrained=pretrained,
            norm_layer=norm_layer,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            mlp_ratio=mlp_ratio,
            downsample_version=downsample_version,
            drop_path_rate=drop_path_rate,
        )

        self.fusion_modules = nn.ModuleList()
        for i in range(len(depths)):
            current_channels = int(dims * (2 ** i))
            self.fusion_modules.append(
                ProjectedSCMCA(
                    dim=current_channels,
                    dropout=0.1 if drop_path_rate > 0 else 0.0
                )
            )

        if self.ape:
            self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
            self.absolute_pos_embed = []
            self.absolute_pos_embed_x = []
            for i_layer in range(len(depths)):
                input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                      self.patches_resolution[1] // (2 ** i_layer))
                dim=int(dims * (2 ** i_layer))
                absolute_pos_embed = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed, std=.02)
                absolute_pos_embed_x = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed_x, std=.02)

                self.absolute_pos_embed.append(absolute_pos_embed)
                self.absolute_pos_embed_x.append(absolute_pos_embed_x)

    def forward_features(self, x_rgb, x_e):
        outs_fused = []
        outs_rgb = self.vssm(x_rgb) 
        outs_x = self.vssm(x_e) 

        for i in range(len(outs_rgb)):
            if self.ape:
                out_rgb = self.absolute_pos_embed[i].to(outs_rgb[i].device) + outs_rgb[i]
                out_x = self.absolute_pos_embed_x[i].to(outs_x[i].device) + outs_x[i]
            else:
                out_rgb = outs_rgb[i]
                out_x = outs_x[i]
            
            x_fuse = self.fusion_modules[i](out_x, out_rgb)
            outs_fused.append(x_fuse)
            
        return outs_fused

    def forward(self, x_rgb, x_e):
        out = self.forward_features(x_rgb, x_e)
        return out

class vssm_tiny(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_tiny, self).__init__(
            depths=[2, 2, 9, 2],
            dims=96,
            pretrained='pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.1,
        )

class vssm_small(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_small, self).__init__(
            depths=[2, 2, 27, 2],
            dims=96,
            pretrained='pretrained/vmamba/vssmsmall_dp03_ckpt_epoch_238.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.3,
        )

class vssm_base(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_base, self).__init__(
            depths=[2, 2, 27, 2],
            dims=128,
            pretrained='pretrained/vmamba/vssmbase_dp06_ckpt_epoch_241.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.6, 
        )