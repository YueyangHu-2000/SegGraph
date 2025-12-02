import torch
from transformers import Dinov2Model, Dinov2PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
import torch
from dataset.PartnetEpc import get_is_seen
from model.Segmentor import Segmentor
import argparse
    
def compute_class_weights(labels, num_classes, eps=1.02):
    hist = torch.bincount(labels.flatten(), minlength=num_classes)
    freq = hist.float() / hist.sum()
    weights = 1.0 / torch.log(freq + eps)
    return weights

class Dinov2PcSegmentor(Dinov2PreTrainedModel):
    def __init__(self, config, cpt_path,args):
        super().__init__(config)
        
        self.cpt_path = cpt_path
        self.ave_per_mask=args.ave_per_mask
        self.num_labels = config.num_labels
        self.dinov2 = Dinov2Model(config)
        self.segmentor = Segmentor(config.hidden_size, config.num_labels, args)
        
    def forward(self, 
                pc, pc_label,pixel_values,mask_label, pc_idx, coords, args, epoch,
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
            outputs = self.dinov2(pixel_values,output_hidden_states=output_hidden_states,output_attentions=output_attentions)
        patch_embeddings = outputs.last_hidden_state[:,1:,:]

        logits = self.segmentor(pc, patch_embeddings, n_pc, pc_idx, coords, mask_label, epoch)
        
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
        )
    
        
    def apply_transform(self, images_tensor):
        images_tensor = images_tensor.permute(0,3,1,2)
        resized_tensor = tvF.resize(images_tensor, size=798, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)

        normalized_tensor = tvF.normalize(resized_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        
        return normalized_tensor

    def save_model(self):
        torch.save(self.segmentor.state_dict(), self.cpt_path)
    
    def load_model(self):
        checkpoint = torch.load(self.cpt_path)
        self.segmentor.load_state_dict(checkpoint)