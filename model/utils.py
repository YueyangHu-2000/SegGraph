import torch
import torch.nn as nn
from torch_scatter import scatter, scatter_mean



class JS2Weight(nn.Module):
    def __init__(self, in_dim,hidden_dim, layers_num=2):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.layers_num = layers_num
        self.fc_A = nn.Linear(in_dim, hidden_dim)
        self.fc_B = nn.Linear(in_dim, hidden_dim)
        
        current_dim = 2 * hidden_dim  # 拼接后的维度
        layers = []
        for i in range(layers_num):
            layers.append(nn.Linear(current_dim, current_dim))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        self.fc_out = nn.Linear(current_dim, 1)
    def forward(self, a, b):
        a = self.fc_A(a)
        b = self.fc_B(b)
        x = torch.cat([a,b], dim=-1)
        x = self.mlp(x)
        x = self.fc_out(x)
        return x

class GeoAwarePooling(nn.Module):
    """Pool point features to super points using geometric-aware weighting."""
    
    def __init__(self, channel_proj: int):
        super(GeoAwarePooling, self).__init__()
        
        self.pts_proj1 = nn.Sequential(
            nn.Linear(3, channel_proj),
            nn.LayerNorm(channel_proj),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(channel_proj, channel_proj),
            nn.LayerNorm(channel_proj)
        )
        self.pts_proj2 = nn.Sequential(
            nn.Linear(2 * channel_proj, channel_proj),
            nn.LayerNorm(channel_proj),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(channel_proj, 1, bias=False),
            nn.Sigmoid()
        )  # [num_segments, 1]

    def forward(
        self, 
        feat,
        xyz,
    ):
        mn = xyz.min(axis=0)[0]
        mx = xyz.max(axis=0)[0]
        dia = (mx-mn).max()
        center = xyz.mean(axis=0)
        xyz_norm = (xyz-center) / (dia+1e-4)
        
        xyz_feat = self.pts_proj1(xyz_norm)  # [N, channel_proj]
        glb_xyz_feat = torch.max(xyz_feat, dim=0)[0]
        glb_xyz_feat = glb_xyz_feat.unsqueeze(0).expand(xyz_feat.size(0), -1)
        xyz_cat_feat = torch.cat([xyz_feat, glb_xyz_feat], dim=-1)        
        # Get segment-wise max features
        # Compute weights
        xyz_w = self.pts_proj2(xyz_cat_feat) * 2  # [N, 1]
        
        ret_feat = (feat*xyz_w).mean(dim=0)

        return ret_feat
    
class MaskQuality(nn.Module):
    """Pool point features to super points using geometric-aware weighting."""
    
    def __init__(self, channel_proj: int):
        super(MaskQuality, self).__init__()
        
        self.pts_proj1 = nn.Sequential(
            nn.Linear(1, channel_proj),
            nn.LayerNorm(channel_proj),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(channel_proj, channel_proj),
            nn.LayerNorm(channel_proj)
        )
        self.pts_proj2 = nn.Sequential(
            nn.Linear(2 * channel_proj, channel_proj),
            nn.LayerNorm(channel_proj),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(channel_proj, 1, bias=False),
            nn.Sigmoid()
        )  # [num_segments, 1]

    def forward(
        self, 
        ang
    ):
        ang_feat = self.pts_proj1(ang)  # [N, channel_proj]
        max_ang_feat = torch.max(ang_feat, dim=0, keepdim=True)[0]
        ave_ang_feat = torch.mean(ang_feat, dim=0, keepdim=True)
        cat_feat = torch.cat([max_ang_feat, ave_ang_feat], dim=-1)        
        # Get segment-wise max features
        # Compute weights
        w = self.pts_proj2(cat_feat) * 2  # [N, 1]
        return w

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    def __init__(self, dim, reduction=4):
        super(AttentionFusion, self).__init__()
        hidden_dim = max(dim // reduction, 1)
        
        # 线性投影（可学习地调整输入）
        self.proj_shallow = nn.Linear(dim, dim)
        self.proj_deep = nn.Linear(dim, dim)
        
        # 注意力模块
        self.attention_mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid()
        )
        
        # 融合后的 MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )
    
    def forward(self, f1, fn):
        f1_proj = self.proj_shallow(f1)  # [N, D]
        fn_proj = self.proj_deep(fn)     # [N, D]
        
        # 拼接以生成注意力权重
        fusion_cat = torch.cat([f1_proj, fn_proj], dim=-1)  # [N, 2D]
        attn = self.attention_mlp(fusion_cat)               # [N, 2]
        attn_shallow, attn_deep = attn[:, 0:1], attn[:, 1:2]  # [N, 1], [N, 1]
        
        # 分别加权
        f1_weighted = f1_proj * attn_shallow
        fn_weighted = fn_proj * attn_deep
        
        # 加和融合
        fusion = f1_weighted + fn_weighted  # [N, D]
        
        # 融合后的进一步特征提升
        fusion_out = self.fusion_mlp(fusion)  # [N, D]
        
        return fusion_out
    
import torch
import torch.nn as nn

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4, ff_multiplier=4, num_layers=1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=output_dim * ff_multiplier,
            batch_first=True  # So input can be (B, N, D)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        x: (B, N, D) tensor
        Returns: (B, N, D') tensor
        """
        x = self.input_proj(x)  # (B, N, D') if needed
        x = self.encoder(x)     # (B, N, D')
        return x

