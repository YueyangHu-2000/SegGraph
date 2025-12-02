import os
import json
import argparse
from tqdm import tqdm

import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import open3d as o3d

from dataset.utils import DATA_PATH, load_label_from_ori, load_pc_from_ori

def load_img(img_path):
    img_list = []
    for i in range(10):
        img = cv2.cvtColor(cv2.imread(os.path.join(img_path, str(i)+".png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img_list.append(img)
    img_list = np.array(img_list)/255
    return img_list

def load_pc_norm(path):
    pc = o3d.io.read_point_cloud("/raid0/yyhu/dataset/partnetE/PartSLIP/data/PartSTAD_few_shot_2/rendered_pc/Bottle/3625/norm.ply")
    xyz = np.array(pc.points)
    norm = np.array(pc.normals)
    xyz = np.concatenate([xyz, norm], axis=-1)
    return xyz


def generate_is_seen(idx, npoint):
    nview = idx.shape[0]
    # 初始化 is_seen 张量
    is_seen = torch.zeros((nview, npoint), dtype=idx.dtype, device=idx.device)

    # 使用 torch.unique 和 torch.bincount 来避免显式循环
    for i in range(nview):
        # 获取当前视图的索引，并过滤掉 -1
        valid_idx = idx[i][idx[i] != -1]
        # 使用 torch.bincount 统计每个索引的出现次数
        counts = torch.bincount(valid_idx, minlength=npoint)
        # 标记被看到的点
        is_seen[i] = counts.clamp(max=1)  # 将大于 1 的值限制为 1

    return is_seen

def get_is_seen(idx, npoint):
    is_numpy = False
    if isinstance(idx, np.ndarray):
        is_numpy=True
        idx = torch.from_numpy(idx)
    
    nview = idx.shape[0]
    is_seen = torch.zeros((nview, npoint), dtype = idx.dtype, device=idx.device)
    for i in range(nview):
        tmp_idx = torch.unique(idx[i].reshape(-1))
        tmp_idx = tmp_idx[tmp_idx!=-1]
        is_seen[i,tmp_idx] = 1
    return is_seen
    
    if is_numpy:
        idx = idx.numpy()
    return is_seen



def get_seen_data(pc, label, idx, coor):
    N = label.shape[0]
    
    is_seen = get_is_seen(idx, N)
    
    idx_seen = np.unique(idx[idx != -1].flatten())
    
    mapping = -np.ones(N, dtype=idx.dtype)
    mapping[idx_seen] = np.arange(idx_seen.shape[0], dtype=idx.dtype)
    pc_idx_seen = mapping[idx]
    pc_idx_seen[idx == -1] = -1
    
    pc_seen = pc[idx_seen]
    label_seen = label[idx_seen]
    coor_seen = coor[:, idx_seen]
    is_seen = is_seen[:, idx_seen]
    
    N = label.shape[0]
    
    return pc_seen, label_seen, pc_idx_seen, coor_seen


class Shape():
    def __init__(self,pc ,pc_label,img,mask_label,pc_idx,coords):   
        self.pc=pc
        self.pc_label=pc_label
        self.img=img
        self.mask_label=mask_label
        self.pc_idx=pc_idx
        self.coords=coords
    def unpack(self):
        return self.pc_label, self.img, self.mask_label, self.pc_idx, self.coords
    

class PartnetEpc(Dataset):
    def __init__(self, split, category):
        super().__init__()
        self.split = split
        self.category = category
        self.data_path = DATA_PATH[self.split]
        self.shape_list = self.load_data()
    
    def load_data(self):
        cate_path = os.path.join(self.data_path, "rendered_pc", self.category)
        if self.split == "val":
            val_list = json.load(open("val_list.json"))
            shape_id_list = val_list[self.category]
        else:
            shape_id_list=os.listdir(cate_path)
        shape_list = []
        for ii, shape_id in tqdm(enumerate(shape_id_list), total=len(shape_id_list), desc=f"loading {self.split} data"):
            img_path = os.path.join(cate_path, shape_id, "rendered_img")
            img = load_img(img_path)
            
            mask_path = f"{self.data_path}/sam_uniform_point_prompts/{self.category}/{shape_id}"
            mask_label = np.load(os.path.join(mask_path, "mask.npy")).astype(np.int32)
            
            # norm_path = f"{self.data_path}/rendered_pc/{self.category}/{shape_id}/norm.ply"
            # pc = load_pc_norm(norm_path)
            pc = load_pc_from_ori(self.category, shape_id, self.split)
            pc_label = load_label_from_ori(self.category, shape_id, self.split)
            
            img_meta_path = f"{self.data_path}/rendered_pc/{self.category}/{shape_id}/img_meta"
            pc_idx = np.load(os.path.join(img_meta_path, "pc_idx.npy"))
            coords = np.load(os.path.join(img_meta_path, "screen_coords.npy"))
            
            norm_path = f"{self.data_path}/rendered_pc/{self.category}/{shape_id}/norm.ply"
            
            pc, pc_label, pc_idx, coords  = get_seen_data(pc, pc_label, pc_idx, coords)
            shape_list.append(Shape(pc, pc_label,img,mask_label,pc_idx,coords))
            
            # break
            
        
        return shape_list
    
    def __len__(self):
        return len(self.shape_list)
    
    def __getitem__(self, index):
        shape = self.shape_list[index]        
        return shape.unpack()
        # return pc, pc_label, img, mask_label, pc_idx, coords
        return {
            # "pc": pc,
            "pc_label": pc_label,
            "img": img,
            "mask_label": mask_label,
            "pc_idx": pc_idx,
            "coords": coords
        }
        
    def calc_ave_seen_vew_num(self):
        b = []
        for ii in range(len(self.shape_list)):
            pc_idx = torch.tensor(self.shape_list[ii].pc_idx)
            a = []
            for jj in range(10):
                pc_idx_perview = pc_idx[jj].flatten()
                pc_idx_perview, count = torch.unique(pc_idx_perview, return_counts=True)
                a.append(pc_idx_perview)
            a = torch.cat(a)
            _, count = torch.unique(a, return_counts=True)
            print(torch.mean(count.float()))
            b.append(torch.mean(count.float()))
        print("total", torch.mean(torch.tensor(b)))    
        pass
            
if __name__ == "__main__":
    dataset = PartnetEpc("test","Bottle","cpu")
    dataset.calc_ave_seen_vew_num()