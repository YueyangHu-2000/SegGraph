import torch
import torch.nn as nn
import torch.nn.functional as F

def proxy_contrastive_loss(pc_feat, label, classifier, temperature=0.1):
    """
    使用 classifier 作为 Proxy 计算对比损失
    :param pc_feat: 点云特征, shape (N, D)
    :param label: 点云标签, shape (N,)
    :param classifier: 线性分类层 (torch.nn.Linear), 其 weight 作为 proxy
    :param temperature: 对比学习温度超参数
    :return: loss 值
    """
    N, D = pc_feat.shape

    proxies = classifier.weight  # (num_classes, D)
    proxies = F.normalize(proxies, dim=-1)  # 归一化

    pc_feat_norm = F.normalize(pc_feat, dim=-1)  # (N, D)

    sim_matrix = torch.matmul(pc_feat_norm, proxies.T)  # (N, num_classes)

    # 取出每个点对应的正确类别代理的相似度
    # pos_sim = sim_matrix.gather(dim=-1, index=label.unsqueeze(-1)).squeeze(-1)  # (N,)

    # 计算 InfoNCE 对比损失
    exp_sim = torch.exp(sim_matrix / temperature)  # (N, num_classes)
    loss = -torch.log(exp_sim / exp_sim.sum(dim=-1, keepdim=True))  # (N, num_classes)
    loss = loss.gather(dim=-1, index=label.unsqueeze(-1)).squeeze(-1)  # (N,)

    return loss.mean()

def construct_positive_negative_samples(features, mask, sample_num = 1024):
    """
    根据 mask 生成正负样本对
    :param features: Tensor, (B, N, D) 表示 B 个点云，每个点云 N 个点，每个点 D 维特征
    :param mask: Tensor, (B, N) 表示每个点的类别标签
    :return: 正样本对 (P, 2, D) 和 负样本对 (Q, 2, D)
    """
    B, N, D = features.shape
    positive_pairs = []
    negative_pairs = []
    
    for b in range(B):
        unique_labels = mask[b].unique()
        for label in unique_labels:
            indices = (mask[b] == label).nonzero(as_tuple=True)[0]
            select_ind = torch.randint(0, len(indices), (sample_num,))
            if len(indices) > 1:
                pos_samples = torch.stack(
                    [features[b, indices[select_ind]],
                     features[b, indices[torch.randint(0, len(indices), (sample_num,))]]],
                    dim=0
                )  
                
                other_indices = (mask[b] != label).nonzero(as_tuple=True)[0]
                if len(indices) > 0 and len(other_indices) > 0:
                    neg_samples = torch.stack(
                        [features[b, indices[select_ind]],
                        features[b, other_indices[torch.randint(0, len(other_indices), (sample_num,))]]],
                        dim=0
                    )
                    positive_pairs.append(pos_samples)
                    negative_pairs.append(neg_samples)

    positive_pairs = torch.cat(positive_pairs, dim=1) if positive_pairs else torch.empty((0, 2, D))
    negative_pairs = torch.cat(negative_pairs, dim=1) if negative_pairs else torch.empty((0, 2, D))

    return positive_pairs, negative_pairs


class ContrastiveLoss(nn.Module):
    """
    对比损失 (NT-Xent)
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, features_pos, features_neg):
        """
        计算 InfoNCE 损失。
        
        参数：
            features_pos: 形状为 (2,N,D) 的正样本对特征
            features_neg: 形状为 (2,N,D) 的负样本对特征
        
        返回：
            loss: 标量，InfoNCE 损失
        """
        _, N, D = features_pos.shape

        # 计算正样本对的余弦相似度
        f1_pos, f2_pos = features_pos[0, :, :], features_pos[1, :, :]
        pos_sim = F.cosine_similarity(f1_pos, f2_pos, dim=-1) / self.temperature

        # 计算负样本对的余弦相似度
        f1_neg, f2_neg = features_neg[0, :, :], features_neg[1, :, :]
        neg_sim = F.cosine_similarity(f1_neg, f2_neg, dim=-1) / self.temperature

        # 拼接所有相似度（正样本 + 负样本）
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)
        labels = torch.zeros(N, dtype=torch.long).cuda()  # 正样本的索引为 0

        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)
        return loss

class PointCloudContrastiveModel(nn.Module):
    def __init__(self, feature_extractor):
        """
        :param feature_extractor: 预训练的特征提取器
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.contrastive_loss = ContrastiveLoss()

    def forward(self, x, mask):
        features = self.feature_extractor(x)  # (B, N, D)
        positive_pairs, negative_pairs = construct_positive_negative_samples(features, mask)
        
        if positive_pairs.shape[0] > 0 and negative_pairs.shape[0] > 0:
            loss = self.contrastive_loss(positive_pairs,negative_pairs)
        else:
            loss = torch.tensor(0.0, device=x.device)

        return loss

def triplet_loss(feat: torch.Tensor, triplets: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    feat: Tensor of shape [N, D] - the feature embeddings.
    triplets: Tensor of shape [M, 3] - each row contains indices [anchor_idx, pos_idx, neg_idx].
    margin: float - the margin hyperparameter in triplet loss.
    
    Returns:
        torch.Tensor - scalar loss value.
    """
    if triplets.numel() == 0:
        return torch.tensor(0.0, device=feat.device, requires_grad=True)
    anchor = feat[triplets[:, 0]]  # [M, D]
    positive = feat[triplets[:, 1]]  # [M, D]
    negative = feat[triplets[:, 2]]  # [M, D]

    # pos_dist = F.pairwise_distance(anchor, positive, p=2)  # [M]
    # neg_dist = F.pairwise_distance(anchor, negative, p=2)  # [M]
    pos_dist = ((anchor - positive) ** 2).sum(dim=1)  # [M]
    neg_dist = ((anchor - negative) ** 2).sum(dim=1)  # [M]
    # pos_dist = F.cosine_similarity(anchor, positive, dim=1)  # [M]
    # neg_dist = F.cosine_similarity(anchor, negative, dim=1) 

    losses = F.relu(pos_dist - neg_dist + margin)  # [M]

    return losses.mean()
