import torch
from transformers import Dinov2Model, Dinov2PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        Focal Loss 实现

        Args:
            alpha: 平衡因子，float 或 Tensor 类型，若为float，则表示所有类别统一权重
            gamma: 调节因子，控制对难分类样本的关注度
            reduction: 损失的聚合方式，'mean', 'sum' 或 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: 预测概率 (logits)，形状 [N, C, ...]，未经过 softmax
            targets: 真实标签，形状 [N, ...]

        Returns:
            计算得到的 Focal Loss
        """
        # 将 logits 转换为概率
        probs = F.softmax(inputs, dim=1)
        # 选择对应类别的概率 p_t
        targets = targets.long()
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [N, ...]

        # 计算Focal Loss公式中的 (1 - p_t)^gamma
        focal_weight = (1 - p_t).pow(self.gamma)

        # 计算 Cross Entropy 损失项
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 计算 Focal Loss
        focal_loss = focal_weight * ce_loss

        # 应用平衡因子
        if isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha.gather(0, targets)
            focal_loss *= alpha_t
        else:
            focal_loss *= self.alpha

        # 按照 reduction 参数进行聚合
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
            
class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=57, tokenH=57, num_labels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels, (1,1))
        self.conv2 = torch.nn.Conv2d(in_channels, in_channels, (1,1))
        self.conv3 = torch.nn.Conv2d(in_channels, in_channels, (1,1))
        self.conv4 = torch.nn.Conv2d(in_channels, num_labels, (1,1))

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        x = embeddings.permute(0,3,1,2)

        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        ret = self.conv4(x)
        return ret
    
class LinearClassifier2(torch.nn.Module):
    def __init__(self, in_channels, num_labels=1, ave_per_mask=0, tokenW=57, tokenH=57):
        super(LinearClassifier2, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.ave_per_mask = ave_per_mask
        
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels//8, (1,1))
        self.conv2 = torch.nn.Conv2d(in_channels//8, num_labels, (1,1))
        self.relu = torch.nn.ReLU()

    def forward(self, embeddings, sam_masks_label):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        x = embeddings.permute(0,3,1,2)
        x = self.relu(self.conv1(x))
        x = torch.nn.functional.interpolate(x, size=(800,800), mode="bilinear", align_corners=False)
        if self.ave_per_mask:
            x = x.permute(0,2,3,1)
            for b in range(x.shape[0]):
                for i in range(sam_masks_label[b].max()+1):
                    ind = (sam_masks_label[b] == i).nonzero(as_tuple=True)
                    if len(ind[0]) == 0:
                        continue
                    x_b_i = x[b][ind]
                    ave = x_b_i.mean(dim=0, keepdim=True)
                    x[b][ind] += ave.squeeze(0)
            x  = x.permute(0,3,1,2)
        
        ret = self.conv2(x)
        return ret

class SEBlock(torch.nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel, channel // reduction),
            torch.nn.ReLU(),
            torch.nn.Linear(channel // reduction, channel),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    
class LinearClassifier3(torch.nn.Module):
    def __init__(self, in_channels, num_labels=1, ave_per_mask=0, tokenW=57, tokenH=57):
        super(LinearClassifier3, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.ave_per_mask = ave_per_mask
        
        # self.conv1 = torch.nn.Conv2d(in_channels, in_channels//16, (1,1))
        self.se_block = SEBlock(in_channels)
        self.dropout = torch.nn.Dropout2d(0.5)
        self.conv2 = torch.nn.Conv2d(in_channels, num_labels, (1,1))
    
    def forward(self, embeddings, sam_masks_label):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        x = embeddings.permute(0,3,1,2)
        x = self.se_block(x)
        x = self.dropout(x)
        x = torch.nn.functional.interpolate(x, size=(800,800), mode="bilinear", align_corners=False)
        ret = self.conv2(x)
        return ret


class Dinov2ForSemanticSegmentationImg(Dinov2PreTrainedModel):
    def __init__(self, config, cpt_path,ave_per_mask=0):
        super().__init__(config)
        
        self.cpt_path = cpt_path
        self.ave_per_mask=ave_per_mask
        self.dinov2 = Dinov2Model(config)
        # print(config.hidden_size)
        # self.classifier = LinearClassifier(config.hidden_size, 57, 57, config.num_labels)
        self.linear = LinearClassifier2(config.hidden_size, config.num_labels, ave_per_mask=ave_per_mask)
        
    def forward(self, pixel_values, output_hidden_states=False, output_attentions=False, labels=None, mask=None, sam_masks_label=None):
        with torch.no_grad():
            outputs = self.dinov2(pixel_values,
                                    output_hidden_states=output_hidden_states,
                                    output_attentions=output_attentions)
        patch_embeddings = outputs.last_hidden_state[:,1:,:]
        
        logits = self.linear(patch_embeddings, sam_masks_label)

        loss = None
        if labels is not None:
            # print(f"----> {(labels==2).sum()/(labels==1).sum()}")
            loss_all = torch.nn.functional.cross_entropy(logits, labels.long(), reduction="none")
            loss = (loss_all * mask).sum()/mask.sum()
            # print("good")
            pass

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def apply_transform(self, images_tensor):
        
        images_tensor = images_tensor.permute(0,3,1,2)
        resized_tensor = tvF.resize(images_tensor, size=798, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)

        normalized_tensor = tvF.normalize(resized_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        
        return normalized_tensor

    def save_model(self):
        torch.save(self.linear.state_dict(), self.cpt_path)
    
    def load_model(self):
        self.linear.load_state_dict(torch.load(self.cpt_path))
    