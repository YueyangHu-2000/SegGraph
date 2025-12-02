import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import json
from dataset.utils import DATA_PATH, load_label_from_ori

class PartnetEimg(Dataset):
    def __init__(self, split, category):
        super().__init__()
        self.split = split
        self.category = category
        self.data_path = DATA_PATH[self.split]
        self.data, self.label, self.mask, self.mask_label= self.load_data_4()
        
        
    def load_data(self):
        cate_path = os.path.join(self.data_path, "rendered_pc", self.category)
        if self.split == "val":
            val_list = json.load(open("val_list.json"))
            shape_id_list = val_list[self.category]
        else:
            shape_id_list=os.listdir(cate_path)
        img_list = []
        label_list = []
        for ii, shape_id in tqdm(enumerate(shape_id_list), total=len(shape_id_list), desc=f"loading {self.split} data"):
            shape_path = os.path.join(cate_path, shape_id, "rendered_img")
            for i in range(10):
                img = cv2.cvtColor(cv2.imread(os.path.join(shape_path, str(i)+".png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                img_list.append(np.expand_dims(img, axis=0))
                
            label_pc = load_label_from_ori(self.category, shape_id, self.split)
            pc_idx_path = os.path.join(cate_path, shape_id, "img_meta", "pc_idx.npy")
            pc_idx = np.load(pc_idx_path)
            label_img = label_pc[pc_idx]+1
            invalid_ind = pc_idx==-1
            label_img[invalid_ind]=0
            label_list.append(label_img)
            
        img_list = np.concatenate(img_list)  # [B*10,W,H,3]
        # print(img_list.shape)
        img_list = img_list.astype(np.float32)/255
        label_list = np.concatenate(label_list) # [B*10, W, H]
        return img_list, label_list
    
    def load_data_3(self):
        cate_path = os.path.join(self.data_path, "rendered_pc", self.category)
        if self.split == "val":
            val_list = json.load(open("val_list.json"))
            shape_id_list = val_list[self.category]
        else:
            shape_id_list=os.listdir(cate_path)
        img_list = []
        label_list = []
        mask_list = []
        mask_label_list = []
        for ii, shape_id in tqdm(enumerate(shape_id_list), total=len(shape_id_list), desc=f"loading {self.split} data"):
            shape_path = os.path.join(cate_path, shape_id, "rendered_img")
            mask_path = f"{self.data_path}/sam_uniform_point_prompts/{self.category}/{shape_id}"
            for i in range(0, 10, 3):
                img = cv2.cvtColor(cv2.imread(os.path.join(shape_path, str(i)+".png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                img_list.append(np.expand_dims(img, axis=0))
                mask = cv2.cvtColor(cv2.imread(os.path.join(mask_path, "sam_auto_mask"+str(i)+".png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                mask_list.append(np.expand_dims(mask, axis=0))
            mask_label = np.load(os.path.join(mask_path, "mask.npy")).astype(np.int32)
            mask_label_list.append(mask_label)
            ind = np.array([0,3,6,9])
            label_pc = load_label_from_ori(self.category, shape_id, self.split)
            pc_idx_path = os.path.join(cate_path, shape_id, "img_meta", "pc_idx.npy")
            pc_idx = np.load(pc_idx_path)
            pc_idx = pc_idx[ind] ## 
            label_img = label_pc[pc_idx]+1
            invalid_ind = pc_idx==-1
            label_img[invalid_ind]=0
            label_list.append(label_img)
            
        img_list = np.concatenate(img_list)  # [B*10,W,H,3]
        img_list = img_list.astype(np.float32)/255
        label_list = np.concatenate(label_list) # [B*10, W, H]
        mask_list = np.concatenate(mask_list)
        mask_list = mask_list.astype(np.float32)/255
        mask_label_list = np.concatenate(mask_label_list)
        return img_list, label_list, mask_list, mask_label_list
    
    
    def load_data_4(self):
        cate_path = os.path.join(self.data_path, "rendered_pc", self.category)
        if self.split == "val":
            val_list = json.load(open("val_list.json"))
            shape_id_list = val_list[self.category]
        else:
            shape_id_list=os.listdir(cate_path)
        img_list = []
        label_list = []
        mask_list = []
        mask_label_list = []
        for ii, shape_id in tqdm(enumerate(shape_id_list), total=len(shape_id_list), desc=f"loading {self.split} data"):
            shape_path = os.path.join(cate_path, shape_id, "rendered_img")
            mask_path = f"{self.data_path}/sam_uniform_point_prompts/{self.category}/{shape_id}"
            for i in range(1): # 
                img = cv2.cvtColor(cv2.imread(os.path.join(shape_path, str(i)+".png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                img_list.append(np.expand_dims(img, axis=0))
                mask = cv2.cvtColor(cv2.imread(os.path.join(mask_path, "sam_auto_mask"+str(i)+".png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                mask_list.append(np.expand_dims(mask, axis=0))
            mask_label = np.load(os.path.join(mask_path, "mask.npy")).astype(np.int32)
            mask_label_list.append(mask_label[0:1]) # 
            label_pc = load_label_from_ori(self.category, shape_id, self.split)
            pc_idx_path = os.path.join(cate_path, shape_id, "img_meta", "pc_idx.npy")
            pc_idx = np.load(pc_idx_path)
            label_img = label_pc[pc_idx]+1  # 再次+1
            invalid_ind = pc_idx==-1
            label_img[invalid_ind]=0
            label_list.append(label_img[0:1]) #
            
        img_list = np.concatenate(img_list)  # [B*10,W,H,3]
        img_list = img_list.astype(np.float32)/255
        label_list = np.concatenate(label_list) # [B*10, W, H]
        mask_list = np.concatenate(mask_list)
        mask_list = mask_list.astype(np.float32)/255
        mask_label_list = np.concatenate(mask_label_list)
        return img_list, label_list, mask_list, mask_label_list
    
    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.mask[index], self.mask_label[index]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="few_shot")
    parser.add_argument("--category", type=str, default="Chair")
    
    args = parser.parse_args()
    dataset = PartnetEimg("few_shot","Chair")