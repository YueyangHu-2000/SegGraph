import os
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModel
import torchvision.transforms.functional as tvF
from torchvision import transforms
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import torch.nn.functional as F
# from src.dift.dift_sd import SDFeaturizer
import math
# from src.clip import clip
import torch.nn as nn

class CrossScaleFusion(nn.Module):
    def __init__(self, in_channels=96, num_layers=2, num_heads=4, hidden_dim=96, out_size=(800, 800)):
        super(CrossScaleFusion, self).__init__()
        self.out_size = out_size

        self.flatten_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.AvgPool2d(kernel_size=4 if i == 0 else 2, stride=4 if i == 0 else 2)  # 下采样
            )
            for i in range(3)
        ])

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )

        self.final_proj = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)

    def forward(self, features):  # features: List of 3 tensors
        B = features[0].shape[0]
        patch_tokens = []
        spatial_shapes = []

        for i in range(3):
            feat = self.flatten_proj[i](features[i])  # (B, C, H', W')
            B, C, H, W = feat.shape
            spatial_shapes.append((H, W))
            patch_tokens.append(feat.flatten(2).transpose(1, 2))  # (B, H*W, C)

        tokens = torch.cat(patch_tokens, dim=1)  # (B, N_total, C) ~1400 tokens

        fused_tokens = self.transformer(tokens)  # (B, N_total, C)

        # Split fused tokens back to each level
        fused_feats = []
        start = 0
        for (H, W) in spatial_shapes:
            L = H * W
            x = fused_tokens[:, start:start+L, :].transpose(1, 2).reshape(B, -1, H, W)
            fused_feats.append(x)
            start += L

        # Upsample each fused feature to (800, 800) and sum
        upsampled_feats = [F.interpolate(f, size=self.out_size, mode='bilinear', align_corners=False)
                           for f in fused_feats]

        out = self.final_proj(sum(upsampled_feats) / len(upsampled_feats))  # (B, 96, 800, 800)

        return out
    
def upsample_imgfeat(img_feats, img_size=(224,224)):
    b, nv, hw, c = img_feats.size(0), img_feats.size(1), img_feats.size(2), img_feats.size(3)
    img_feats = img_feats.reshape(b * nv, hw, c)
    
    upsample = torch.nn.Upsample(size=img_size, mode='bilinear')  # nearest, bilinear
    avgpool = torch.nn.AvgPool2d(6, 1, 0)
    padding = torch.nn.ReplicationPad2d((2, 3, 2, 3))
    
    img_feats = img_feats.permute(0, 2, 1).reshape(-1, c, int(hw**0.5), int(hw**0.5))
    img_feats = avgpool(padding(img_feats))
    output = upsample(img_feats)
    return output
    
class Dinov2Encoder(torch.nn.Module):
    def __init__(self, use_cache, out_channels=None, tokenW=57, tokenH=57):
        super(Dinov2Encoder, self).__init__()
        self.use_cache = use_cache
        self.hidden_size = 768
        self.out_channels = out_channels if not(out_channels is None) else self.hidden_size//8 # 96
        if not use_cache:
            self.dinov2 = AutoModel.from_pretrained('facebook/dinov2-base')
            for param in self.dinov2.parameters():
                param.requires_grad = False
        self.conv1 = torch.nn.Conv2d(self.hidden_size, self.out_channels, (1,1))
        self.decode = torch.nn.Conv2d(self.out_channels, self.hidden_size, (1,1))
        self.relu = torch.nn.ReLU()
        self.cache = {}
        # self.feature_enhance = torch.nn.Sequential(
        #     torch.nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
        #     torch.nn.BatchNorm2d(self.out_channels),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
        #     torch.nn.BatchNorm2d(self.out_channels),
        #     torch.nn.ReLU(inplace=True),
        # )
        
    def apply_transform(self, images_tensor):
        B,W,H,C = images_tensor.shape
        images_tensor = images_tensor.permute(0,3,1,2)
        img_size = W if W!=800 else 798
        resized_tensor = tvF.resize(images_tensor, size=img_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)

        normalized_tensor = tvF.normalize(resized_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        
        return normalized_tensor
    
    def encode_img(self, img):
        B,W,H,d = img.shape
        img = self.apply_transform(img)
        with torch.no_grad():
            outputs = self.dinov2(img, output_hidden_states=False,output_attentions=False)
        patch_embeddings = outputs.last_hidden_state[:,1:,:]
        ebd_wh = int(math.sqrt(patch_embeddings.shape[1]))
        patch_embeddings = patch_embeddings.reshape(-1, ebd_wh, ebd_wh, self.hidden_size)
        img_feat = patch_embeddings.permute(0,3,1,2)
        return img_feat
    
    def forward(self, id, img, data_path=None):
        """_summary_

        Args:
            img (_type_): [B,W,H,3]
        """
        # print("ffffffffffffffforward", self.use_cache, id)
        B,W,H,C = img.shape
        if img.shape[1]==768:
            img_feat = img
        else:
            if id !=-1:
                if id in self.cache:
                    # print("use cache")
                    img_feat = self.cache[id]
                else:
                    img_feat = self.encode_img(img)
                    self.cache[id] = img_feat
            else:
                img_feat = self.encode_img(img)
                
        x = self.conv1(img_feat)
        img_feat_pred = self.decode(x)
        loss = F.mse_loss(img_feat_pred, img_feat)
        # x = self.feature_enhance(x)+x
        x = torch.nn.functional.interpolate(x, size=(W,H), mode="bilinear", align_corners=False)
        
        return x, loss


class ImageEncoder(torch.nn.Module):
    def __init__(self, img_encoder, use_cache):
        super(ImageEncoder, self).__init__()
        self.img_encoder = img_encoder
        self.use_cache = use_cache
        if self.img_encoder == "dinov2":
            self.encoder = Dinov2Encoder(use_cache)
            self.out_dim = self.encoder.out_channels

        
    def forward(self, id, x, data_path=None):
        """_summary_

        Args:
            x (_type_): [B,W,H,3]

        Returns:
            _type_: [B,W,H,D]
        """
        if self.img_encoder == "dinov2_sam":
            # print("dinov2_sam")
            img_feat_dinov2, loss1 = self.dinov2(id, x, data_path)
            img_feat_sam, loss2 = self.sam(id, x, data_path)
            img_feat = torch.cat([img_feat_dinov2, img_feat_sam], dim=1)
            loss = loss1+loss2
        else:
            img_feat, loss = self.encoder(id, x, data_path)
        return img_feat, loss
        
    
if __name__ == "__main__":
    model = ImageEncoder()