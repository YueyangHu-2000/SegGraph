import torch
import torch.nn as nn
import torch.nn.functional as F

class SPAttentionAggregation(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attn_mlp = nn.Linear(2 * feature_dim, 1)  # 用于计算注意力分数
        # self.softmax = nn.Softmax(dim=0)  # 归一化注意力权重

    def forward(self, f_points, f_ave):
        f_ave_expanded = f_ave.expand_as(f_points)  # (N, C)
        
        beta = self.attn_mlp(torch.cat([f_points, f_ave_expanded], dim=-1))  # (N, 1)

        f_aggre = (beta * f_points).sum(dim=0) / beta.sum()

        return f_aggre

class SPAttentionDownPropagation(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.propagation_mlp = nn.Linear(2 * feature_dim, 1)  # 计算影响系数

    def forward(self, f_points, f_superpoint):
        # 复制超点特征，使其与点特征形状匹配
        f_superpoint_expanded = f_superpoint.expand_as(f_points)  # (N, C)

        # 计算注意力权重
        beta = torch.sigmoid(self.propagation_mlp(torch.cat([f_points, f_superpoint_expanded], dim=-1)))  # (N, 1)

        # 反传特征
        f_points = f_points + beta * f_superpoint_expanded

        return f_points, beta * f_superpoint_expanded

class SuperpointAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.aggregation = SPAttentionAggregation(feature_dim)
        self.propagation = SPAttentionDownPropagation(feature_dim)

    def forward(self, f_points):
        # 1. 先聚合到超点
        f_superpoint, attn_weights_agg = self.aggregation(f_points)

        # 2. 再从超点反传到点
        f_points_updated, attn_weights_prop = self.propagation(f_points, f_superpoint)

        return f_points_updated, attn_weights_agg, attn_weights_prop
