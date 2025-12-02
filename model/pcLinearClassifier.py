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
from dataset.PartnetEpc import get_is_seen
import argparse
    
class Linear1(torch.nn.Module):
    def __init__(self, in_channels, out_channels, tokenW=57, tokenH=57):
        super(Linear1, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, (1,1))
        self.relu = torch.nn.ReLU()
        # self.relu = torch.nn.Identity()

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        x = embeddings.permute(0,3,1,2)
        x = self.relu(self.conv1(x))
        x = torch.nn.functional.interpolate(x, size=(800,800), mode="bilinear", align_corners=False)
        
        # upsampled = []
        # for i in range(10):
        #     sample = x[i].unsqueeze(0)
        #     sample_upsampled = F.interpolate(sample, size=(800, 800), mode="bilinear", align_corners=False)
        #     upsampled.append(sample_upsampled.squeeze(0)) 
        # x = torch.stack(upsampled, dim=0) 
        return x

def aggregate(npoint, img_feat, pc_idx, coords, args):
    nview=pc_idx.shape[0]
    device = img_feat.device
    
    nbatch = torch.repeat_interleave(torch.arange(0, nview)[:, None], npoint).view(-1, ).long()
    point_loc = coords.reshape(nview, -1, 2)
    xx, yy = point_loc[:, :, 0].long().reshape(-1), point_loc[:, :, 1].long().reshape(-1)
    point_feats = img_feat[nbatch, :, yy, xx].view(nview, npoint, -1)
    is_seen = get_is_seen(pc_idx, npoint).to(device)
    point_feats = torch.sum(point_feats * is_seen[:,:,None], dim=0)
    if not args.use_attn_map:  # 用attn map 就不需要平均了
        point_feats = point_feats/torch.sum(is_seen, dim=0)[:,None]
    # point_feats = torch.mean(point_feats, dim=0) 
    return point_feats
        
class Classifier(torch.nn.Module):
    def __init__(self, in_channels, n_label):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_channels, n_label)

    def forward(self, x):
        out = self.fc(x)  # (N, num_classes)
        return out
    

class AttentionMap(nn.Module):
    def __init__(self, kernel_size=7):
        super(AttentionMap, self).__init__()
        assert kernel_size in (3, 5, 7), "Kernel size must be 3, 5, or 7"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # [B, 1, W, H]
        mean_pool = torch.mean(x, dim=1, keepdim=True)  # [B, 1, W, H]

        attention = torch.cat([max_pool, mean_pool], dim=1)  # [B, 2, W, H]

        attention = self.conv(attention)  # [B, 1, W, H]

        attention = torch.sigmoid(attention)  # [B, 1, W, H]

        return attention


def compute_class_weights(labels, num_classes, eps=1.02):
    hist = torch.bincount(labels.flatten(), minlength=num_classes)
    freq = hist.float() / hist.sum()
    weights = 1.0 / torch.log(freq + eps)
    return weights

class Dinov2ForSemanticSegmentationPc(Dinov2PreTrainedModel):
    def __init__(self, config, cpt_path,ave_per_mask=0):
        super().__init__(config)
        
        self.cpt_path = cpt_path
        self.ave_per_mask=ave_per_mask
        self.num_labels = config.num_labels
        self.dinov2 = Dinov2Model(config)
        
        self.linear1 = Linear1(config.hidden_size, config.hidden_size//8)
        self.classifier = Classifier(config.hidden_size//8, config.num_labels)
        self.attn = AttentionMap()
        
    def forward(self, 
                pc_label,pixel_values,mask_label, pc_idx, coords, args,
                output_hidden_states=False, output_attentions=False):
        
        """
        只处理一个batch的数据，并把batch的维度压缩掉

        Args:
            pixel_values (_type_): [view_num,W,H,3]

        Returns:
            _type_: _description_
        """
        n_pc = pc_label.shape[0]
        with torch.no_grad():
            outputs = self.dinov2(pixel_values,
                                    output_hidden_states=output_hidden_states,
                                    output_attentions=output_attentions)
        patch_embeddings = outputs.last_hidden_state[:,1:,:]
        
        img_feat = self.linear1(patch_embeddings)
        
        if args.use_attn_map:
            attn_map = self.attn(img_feat)
            img_feat = img_feat*attn_map
        
        pc_feat = aggregate(n_pc, img_feat, pc_idx, coords, args)
        
        
        # img_feat = img_feat.permute(0,2,3,1)
        # NN = (pc_idx!=-1).sum()
        # pc_feat = img_feat[pc_idx!=-1].reshape(NN,-1)
        # ind = pc_idx[pc_idx!=-1].reshape(NN)
        # pc_label = pc_label[ind]
        
        logits = self.classifier(pc_feat)
        
        loss = None
        if pc_label is not None:
            # print(f"----> {(pc_label==1).sum()/(pc_label==0).sum()}")
            weight = compute_class_weights(pc_label, self.num_labels)
            loss_all = torch.nn.functional.cross_entropy(logits, pc_label.long(), weight=weight, reduction="none")
            # loss = (loss_all * mask).mean()
            loss = loss_all.mean()
            pass
        
        # logits = logits.permute(1,0).unsqueeze(dim=0)  # 当拓展batch时删除.unsqueeze(dim=0)
        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
        ), pc_label
    
        
        
    
    def apply_transform(self, images_tensor):
        images_tensor = images_tensor.permute(0,3,1,2)
        resized_tensor = tvF.resize(images_tensor, size=798, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)

        normalized_tensor = tvF.normalize(resized_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        
        return normalized_tensor

    def save_model(self):
        torch.save({
            'linear1': self.linear1.state_dict(),
            'classifier': self.classifier.state_dict(),
            'attn':self.attn.state_dict(),
        }, self.cpt_path)
    
    def load_model(self):
        checkpoint = torch.load(self.cpt_path)
        self.linear1.load_state_dict(checkpoint['linear1'])
        self.classifier.load_state_dict(checkpoint['classifier'])
        self.attn.load_state_dict(checkpoint['attn'])