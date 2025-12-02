import os
import json
import argparse
from tqdm import tqdm

import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import open3d as o3d
from pytorch3d.ops import ball_query
from src.lib_o3d import batch_geo_feature,estimate_geo_feature
from pytorch3d.ops import sample_farthest_points
from process.conn import get_pc_mask_online
import pickle

import h5py

all_categories = ['Phone','CoffeeMachine','Laptop',  'Bottle',  'Cart', 'Refrigerator',  'Box', 'Bucket', 'Camera', 'Chair', 'Clock', 'Dishwasher', 'Dispenser', 'Display', 'Door', 'Eyeglasses', 'Faucet', 'FoldingChair', 'Globe', 'Kettle', 'Keyboard', 'KitchenPot', 'Knife', 'Lamp', 'Lighter', 'Microwave', 'Mouse', 'Oven', 'Pen', 'Pliers', 'Printer', 'Remote', 'Safe', 'Scissors', 'Stapler', 'Suitcase', 'Switch', 'Table', 'Toaster', 'Toilet', 'TrashCan', 'USB', 'WashingMachine', 'Window', 'StorageFurniture']
# all_categories = ['Oven', 'Pen', 'Pliers', 'Printer', 'Remote', 'Safe', 'Scissors', 'Stapler', 'Suitcase', 'Switch', 'Table', 'Toaster', 'Toilet', 'TrashCan', 'USB', 'WashingMachine', 'Window', 'StorageFurniture']

def load_img(img_path, num_view):
    img_list = []
    for i in range(num_view):
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

def load_label_from_ori(label_path):
    label_path = os.path.join(label_path)
    label = np.load(label_path, allow_pickle=True)
    label_dict = label.item()
    label = label_dict['semantic_seg']+1
    return label

def load_pc_from_ori(pc_path):
    pc_path = os.path.join(pc_path)
    pc = o3d.io.read_point_cloud(pc_path)
    xyz = np.asarray(pc.points)
    xyz = xyz - xyz.mean(axis=0)
    xyz = xyz / np.linalg.norm(xyz, axis=1, ord=2).max()
    return xyz

def get_fpfh(pc, voxel_size=0.01):
    normal, fpfh = estimate_geo_feature(pc, voxel_size=voxel_size)
    return normal, fpfh

def get_seen_data(pc, label, idx, coor, pc_fpfh, pc_norm):
    N = label.shape[0]
    
    # is_seen = get_is_seen(idx, N)
    idx_seen = np.unique(idx[idx != -1].flatten())
    
    mapping = -np.ones(N, dtype=idx.dtype)
    mapping[idx_seen] = np.arange(idx_seen.shape[0], dtype=idx.dtype)
    pc_idx_seen = mapping[idx]
    pc_idx_seen[idx == -1] = -1
    
    pc_seen = pc[idx_seen]
    label_seen = label[idx_seen]
    coor_seen = coor[:, idx_seen]
    pc_fpfh_seen = pc_fpfh[idx_seen]
    pc_norm_seen = pc_norm[idx_seen]
    
    all_indices = np.arange(N)
    idx_unseen = ~np.isin(all_indices, idx_seen)
    pc_unseen = pc[idx_unseen]
    pc_label_unseen = label[idx_unseen]
    

    return pc_seen, label_seen, pc_unseen, pc_label_unseen, pc_idx_seen, coor_seen,pc_fpfh_seen, pc_norm_seen, idx_seen


class Shape():
    def __init__(self,id, pc ,pc_label,unseen_pc, unseen_pc_label,img,mask_label,pc_idx,coords, pc_fpfh, pc_norm):   
        self.id = id
        self.pc=pc
        self.pc_label=pc_label
        self.unseen_pc = unseen_pc
        self.unseen_pc_label = unseen_pc_label
        self.img=img
        self.mask_label=mask_label
        self.pc_idx=pc_idx
        self.coords=coords
        self.pc_fpfh = pc_fpfh
        self.pc_norm = pc_norm
    def unpack(self):
        return self.id, self.pc, self.pc_label, self.unseen_pc, self.unseen_pc_label, self.img, self.mask_label, self.pc_idx, self.coords, self.pc_fpfh, self.pc_norm


class EdgeIndex():
    def __init__(self):
        self.s_nodes = []
        self.t_nodes = []
    def add(self, u,v):
        self.s_nodes.append(u)
        self.t_nodes.append(v)
    def get_edge_index(self):
        edge_index = torch.stack([torch.tensor(self.s_nodes), torch.tensor(self.t_nodes)])
        return edge_index                  

def get_edge_index(edges):
    source_pt_list = []
    target_pt_list = []
    N = edges.shape[0]
    for i in range(N):
        for j in range(N):
            if edges[i,j]==1:
                source_pt_list.append(i)
                target_pt_list.append(j)
    edge_index = torch.stack([torch.tensor(source_pt_list), torch.tensor(target_pt_list)])
    return edge_index

class PartnetEpcNoise(Dataset):
    def __init__(self, split, category, args, save_data=False, show_figure=False, tsne=None):
        super().__init__()
        self.split = split
        if self.split == "few_shot":
            self.split = "train"
        self.category = category
        self.show_figure = show_figure
        self.tsne=tsne
        self.shot = args.shot
        
        partnete_meta = json.load(open("PartNetE_meta.json"))
        self.path_list = json.load(open("PartNetE_split_path_lists.json", "r"))
        self.num_label = len(partnete_meta[category]) + 1
        
        self.use_cache = args.use_cache
        
        self.debug = args.debug
        self.use_propagate = args.use_propagate
        self.eliminate_sparseness = args.eliminate_sparseness
        self.use_gnn = args.use_gnn
        self.back_to_edges = args.back_to_edges
        self.use_pseudo_label = args.use_pseudo_label
        self.args = args
        self.shape_list = self.load_data(save_data)
        
        
    def norm_unit_sphere(self, cloud):
        """
        cloud: [n_pts, 3]
        """
        xmax = cloud[:,0].max()
        xmin = cloud[:,0].min()
        xcen = (xmax+xmin)/2.0
        ymax = cloud[:,1].max()
        ymin = cloud[:,1].min()
        ycen = (ymax+ymin)/2.0
        zmax = cloud[:,2].max()
        zmin = cloud[:,2].min()
        zcen = (zmax+zmin)/2.0
        center = np.array([xcen,ycen,zcen])
        # zero centering
        cloud = cloud - center
        # scale to unit sphere
        scaler = np.linalg.norm(cloud, axis=-1, ord=2).max()
        cloud = cloud / scaler
        
        return cloud
    
    
    def get_pc_ind(self, pc, view, label, mask_label, pc_idx, coords, nearest_index, grouped_indices):
        n_point = pc_idx.max()+3
        img_ind = mask_label[view]==label
        pc_ind = pc_idx[view, img_ind]
        pc_ind = pc_ind[pc_ind!=-1]
        pc_seen = np.zeros(n_point).astype(np.int32)
        pc_seen[pc_ind]=1
        if pc_ind.size ==0:
            return np.empty((0))
        centers = nearest_index[pc_ind]
        centers = np.unique(centers)
        ret_ind = []
        for i in range(centers.shape[0]):
            g_ind = grouped_indices[centers[i]]
            cnt_seen = pc_seen[g_ind].sum()
            if cnt_seen/g_ind.shape[0]<0.0: # 0.4
                new_ind = g_ind[pc_seen[g_ind]==1]
                # ret_ind.append(g_ind[pc_seen[g_ind]==1])
            else:
                ret_coords = coords[view, g_ind].astype(np.int32)
                in_mask = img_ind[ret_coords[:,1], ret_coords[:,0]]
                new_ind = g_ind[in_mask]
                # ret_ind.append(g_ind[in_mask])
            ret_ind.append(new_ind)
        ret_ind = np.concatenate(ret_ind)
        # if ret_ind.size > 0:
        #     largest_component_ind, _ = get_largest_connected_component_indices(pc[ret_ind], radius=0.005)
        #     ret_ind = ret_ind[largest_component_ind]
        return ret_ind
        
    def get_group_ind(self, view, label, mask_label, pc_idx, coords, nearest_index, grouped_indices):
        n_point = pc_idx.max()+3
        img_ind = mask_label[view]==label
        pc_ind = pc_idx[view, img_ind]
        pc_ind = pc_ind[pc_ind!=-1]
        pc_seen = np.zeros(n_point).astype(np.int32)
        pc_seen[pc_ind]=1
        if pc_ind.size ==0:
            return np.empty((0))
        centers = nearest_index[pc_ind]
        centers = np.unique(centers)
        ret_ind = []
        for i in range(centers.shape[0]):
            g_ind = grouped_indices[centers[i]]
            
            ret_coords = coords[view, g_ind].astype(np.int32)
            in_mask = img_ind[ret_coords[:,1], ret_coords[:,0]]
            if in_mask.sum()/g_ind.shape[0] > 0.5:
                ret_ind.append(i)
        return np.array(ret_ind)
    
    def get_conf_edges(self, graph, pc_label):
        edges_cnt = graph["edges_cnt"]
        mask_area = graph["mask_area"]
        self_i  = torch.arange(edges_cnt.shape[0])
        edges_cnt[self_i, self_i] = mask_area
        n_mask = max(graph["mask2id"].values())+1
        edges = np.zeros((n_mask, n_mask),dtype=np.int64)
        ratio_bar = self.args.conf_bar
        for i in range(edges.shape[0]):
            for j in range(i+1, edges.shape[1]):
                ri = edges_cnt[i,j] / mask_area[i]
                rj = edges_cnt[i,j] / mask_area[j]
                if ri>=ratio_bar and rj >=ratio_bar : #and (conf_i>=0.9 and conf_j>=0.9 and lb_i==lb_j): # and edges_cnt[i,j]>50:
                    edges[i,j]=1
                    edges[j,i]=1
                else:
                    edges[i,j]=0
                    edges[j,i]=0
                        
        return edges, get_edge_index(edges)
        
    def get_mask_norm_P(self, pc):
        mn = pc.min(axis=0)
        mx = pc.max(axis=0)
        dia = (mx-mn).max()
        center = pc.mean(axis=0)
        pc = (pc-center) / (dia+1e-4)
        return pc
    def get_normAng(self, pc, norm, pos, sample=False):
        N = pc.shape[0]
        v1 = pc-pos
        v2 = norm
        v1 = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-8) 
        v2 = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-8) 
        Ang = np.sum(v1 * v2, axis=1, keepdims=True)
        sample_num = 50
        if sample and N>sample_num:
            indices = np.random.choice(N, sample_num, replace=False)
            Ang = Ang[indices]
            
        return np.abs(Ang)
    
    def load_data(self, save_data = False):
        shape_list = []
        
        path_list = self.path_list[self.category][self.split]
        for ii, (pc_path, process_path) in tqdm(enumerate(path_list), total=len(path_list), desc=f"loading {self.category}_{self.split} data"):
            pc_path = os.path.join(self.args.pc_root_path, pc_path)
            process_path = os.path.join(self.args.preprocess_root_path, process_path)
            
            img_meta_path = os.path.join(process_path, "rendered_pc", "img_meta")
            pc_idx = np.load(os.path.join(img_meta_path, "pc_idx.npy"))
            coords = np.load(os.path.join(img_meta_path, "screen_coords.npy"))
            num_view = pc_idx.shape[0]
            
            img_path = os.path.join(process_path,"rendered_pc","rendered_img")
            img = load_img(img_path, num_view)
            
            pc = load_pc_from_ori(os.path.join(pc_path, "pc.ply"))
            pc_label = load_label_from_ori(os.path.join(pc_path, "label.npy"))
                       
            
            pc_norm, pc_fpfh = get_fpfh(pc, voxel_size=0.02)
            
            pc, pc_label, unseen_pc, unseen_pc_label, pc_idx, coords, pc_fpfh, pc_norm, seen_ind= get_seen_data(pc, pc_label, pc_idx, coords, pc_fpfh, pc_norm)
            # print(self.category,  pc.shape[0]+unseen_pc.shape[0], unseen_pc.shape[0], unseen_pc.shape[0]/(pc.shape[0]+unseen_pc.shape[0]))
            # continue
            mask_label = np.load(os.path.join(process_path, "sam_results", "mask_2.npy")).astype(np.int32)
            graph = {}
            graph["visualize_shape_id"] = os.path.basename(process_path)
            if self.use_propagate:                
                graph_path = os.path.join(process_path, "graph_cnt.pkl")
                with open(graph_path, "rb") as f:
                    mask_area, mask2id, id2mask, edges_cnt = pickle.load(f)
                    
                    graph["edges_cnt"] = edges_cnt
                    graph["mask_area"] = mask_area
                    graph["mask2id"] = mask2id
                    graph["id2mask"] = id2mask
                
                if self.eliminate_sparseness:
                    graph_path = os.path.join(process_path, "pc_mask_ind_eli_2.pkl")
                
                with open(graph_path, "rb") as f:
                    mask_pc_ind = pickle.load(f)
                    graph["mask_pc_ind"] = mask_pc_ind
                
                conf_edges, conf_edge_index = self.get_conf_edges(graph, pc_label)
                
                adj_edges_path = os.path.join(process_path, "adj_edges.npy")
                # adj_edges = np.load(adj_edges_path)
                adj_edges = np.load(adj_edges_path, allow_pickle=True)
                tmp_adj_edges = edges_cnt!=0
                adj_edges |= tmp_adj_edges
                adj_edges -=  conf_edges
                adj_edges[adj_edges==-1]=0
                
                graph["adj_edges"] = adj_edges
                graph["weak_edges"] = adj_edges
                
                
                
                sampled_data_path = os.path.join(process_path, "seen_randsampled204810.npy")
                # with open(sampled_data_path, 'rb') as f:
                #     sampled_data = pickle.load(f)
                
                
                sampled_data = np.load(sampled_data_path, allow_pickle=True).item()
                centers = sampled_data["sampled_index"]
                nearest_index = sampled_data["nearest_index"]
                
                # with open(sampled_data_path, 'rb') as f:
                #     sampled_data = pickle.load(f)
                # centers = sampled_data["sampled_index"]
                # nearest_index = sampled_data["nearest_index"]
                    
                if nearest_index.shape[-1] == 1:
                    nearest_index = np.squeeze(nearest_index, axis=-1)
                mapping = -np.ones(pc.shape[0], dtype = centers.dtype)
                mapping[centers] = np.arange(centers.shape[0], dtype = centers.dtype)
                inverse_mapping = np.empty(centers.shape[0], dtype=centers.dtype)
                inverse_mapping[np.arange(centers.shape[0])] = centers
                
                graph["nearest_index"] = nearest_index
                graph["centers"] = centers
                
                
                id2mask = graph["id2mask"]
                mask_pc_ind = graph["mask_pc_ind"]
                mask_num = max(graph["mask2id"].values())+1
                
                mask_group_ind = []
                mask_pc = []
                mask_normAng = []
                if num_view<=10:
                    view_pos = np.load("my_view.npy")
                else:
                    view_pos = np.load(f"view_{num_view}.npy")
                sample_pc_ind = []
                for i in range(mask_num):
                    view, label = id2mask[i]
                    pc_ind_i = mask_pc_ind[i]
                    inter = np.intersect1d(pc_ind_i, centers)
                    if inter.size > 0:
                        if self.args.sample_pc and inter.size > self.args.sample_pc:
                            tmp_pc = pc[inter]
                            centroid = np.mean(tmp_pc, axis=0)
                            distances = np.linalg.norm(tmp_pc - centroid, axis=1)
                            nearest_indices = np.argsort(distances)[:self.args.sample_pc]
                            inter = inter[nearest_indices]
                        mask_group_ind.append(mapping[inter])
                        mask_pc.append(pc[inter])
                        mask_normAng.append(self.get_normAng(pc[inter], pc_norm[inter], view_pos[view], sample=False))
                        if self.args.sample_pc:
                            sample_pc_ind.append(inter)               
                    else:
                        mask_group_ind.append(np.empty((0)))
                        mask_pc.append(np.empty((0)))
                        mask_normAng.append(np.empty((0)))
                if self.args.sample_pc:
                    sample_pc_ind = np.concatenate(sample_pc_ind, axis=-1)
                    sample_pc_ind = np.unique(sample_pc_ind)
                    graph["sample_pc_ind"] = sample_pc_ind
                graph["mask_group_ind"] = mask_group_ind
                graph["mask_pc"] = mask_pc
                graph["mask_normAng"] = mask_normAng
                
                
                if self.args.All_graph:
                    
                    n_group = centers.shape[0]
                    n_mask = max(mask2id.values())+1
                    
                    strong_edge_index = EdgeIndex()
                    weak_edge_index = EdgeIndex()
                    
                    weak_edges = graph["weak_edges"]
                    
                    for i in range(n_mask):
                        for j in range(i, n_mask):
                            if conf_edges[i,j]==1 and mask_group_ind[i].size>0 and mask_group_ind[j].size>0:
                            # if (conf_edges[i,j]==1 or weak_edges[i,j]==1) and mask_group_ind[i].size>0 and mask_group_ind[j].size>0:
                                
                                strong_edge_index.add(i, j) 
                                strong_edge_index.add(j, i)
                    
                    # weak_edges_fpfh = []
                    # save_edges_fpfh = np.zeros((weak_edges.shape[0], weak_edges.shape[0], 33))
                    for i in range(n_mask):
                        
                        for j in range(i, n_mask):
                            if weak_edges[i,j]==1 and mask_group_ind[i].size>0 and mask_group_ind[j].size>0:
                                weak_edge_index.add(i, j) 
                                weak_edge_index.add(j, i)
                                
                                # weak_edges_fpfh.append(edges_fpfh[i,j])
                                # weak_edges_fpfh.append(edges_fpfh[j,i])

                    graph["strong_edge_index"] = strong_edge_index.get_edge_index()
                    graph["weak_edge_index"] = weak_edge_index.get_edge_index()
            else:
                pass
            
            graph["feat_path"] = "tmp"
            shape_list.append(Shape(ii, pc, pc_label, unseen_pc, unseen_pc_label,img, mask_label,pc_idx,coords, graph, pc_norm))
            if self.debug:
                break
            
        
        return shape_list
    
    def __len__(self):
        return len(self.shape_list)
    
    def __getitem__(self, index):
        shape = self.shape_list[index]        
        return shape.unpack()
        
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=int, default=0)
    parser.add_argument("--use_pretrain", type=int, default=0)
    
    
    parser.add_argument("--category", type=str, nargs='+', default=['Mouse']) # ['CoffeeMachine', 'Bottle', 'Cart', 'Refrigerator', 'Laptop', 'Phone']
    parser.add_argument("--shot", type=int, default=8)
    # parser.add_argument("--cuda", type=str, default="cuda:2")
    parser.add_argument("--mode", type=str, default="train")  #!!!!! if change test change output_dir
    parser.add_argument("--output_dir", type=str, default="res/tmp5")
    parser.add_argument("--epoch", type=int, default=20) 
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001) 
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience_limit", type=int, default=5) 
     
    
    parser.add_argument("--use_2d_feat", type=int, default=1) 
    parser.add_argument("--use_3d_feat", type=int, default=0)
    
    parser.add_argument("--img_encoder", type=str, default="dinov2")
    parser.add_argument("--use_cache", type=int, default=0)
    
    parser.add_argument("--sample_pc", type=int, default=0)
    parser.add_argument("--transformer", type=int, default=0)
    parser.add_argument("--conf_bar", type=float, default=0.05)
    parser.add_argument("--use_pseudo_label", type=int, default=0)

    parser.add_argument("--up_method", type=str, default="GA_pooling") # ave GA_pooling
    parser.add_argument("--down_method", type=str, default="MQA_unpooling") # ave MQA_unpooling
    parser.add_argument("--select_edges", type=str, nargs='+', default=["strong","weak"]) # All strong weak
    parser.add_argument("--LH_method", type=str, default="ave") # ave
    
    
    
    parser.add_argument("--use_W_imgfeat", type=int, default=0) # 2
    
    parser.add_argument("--use_propagate", type=int, default=1) # 2
    parser.add_argument("--eliminate_sparseness", type=int, default=1)
    parser.add_argument("--ave_per_mask", type=int, default=0)
    parser.add_argument("--use_gnn", type=int, default=1) 
    
    parser.add_argument("--ave_inter_mask", type=int, default=0)
    
    parser.add_argument("--use_slow_start", type=int, default=-2)
    parser.add_argument("--use_new_classifier", type=int, default=0) 
    parser.add_argument("--use_js2weight", type=int, default=0) 
    parser.add_argument("--use_attn_ave", type=int, default=0)
    parser.add_argument("--use_ave_best", type=int, default=0) 
    parser.add_argument("--back_to_edges", type=int, default=0)
    
    parser.add_argument("--conf_label_edge", type=int, default=0) 
    parser.add_argument("--gt_label_edge", type=int, default=0)
    parser.add_argument("--ps_label_edge", type=int, default=0)
    parser.add_argument("--img_feat_on_mask", type=int, default=0)
    
    parser.add_argument("--All_graph", type=int, default=1)
    parser.add_argument("--use_ball_propagate", type=int, default=0)
    parser.add_argument("--graph4", type=int, default=0)
    
    parser.add_argument("--self_supervised", type=int, default=0)

    parser.add_argument("--use_proxy_contrast_loss", type=int, default=0)
    parser.add_argument("--use_contrast_loss2", type=int, default=0)
    parser.add_argument("--use_ref_loss", type=int, default=0) 
    parser.add_argument("--use_mask_consist_loss", type=int, default=0) 
    parser.add_argument("--use_triplet_loss", type=int, default=0) 
    
    parser.add_argument("--pc_root_path", type=str, default="/raid0/yyhu/dataset/partnetE/PartSLIP/data")
    parser.add_argument("--preprocess_root_path", type=str, default="/data/huyy23/dataset2/PartSLIP/rebuttal_noise/noise_0.01")
    
    parser.add_argument("--num_view", type=int, default=10)
    
    # CUDA_VISIBLE_DEVICES=2 python train_pc_segmentor_newAllG.py --output_dir ./res/tmp5 --category All --use_propagate 1 --eliminate_sparseness 1 --All_graph 1 --ave_per_mask 1 --up_method ave --down_method ave 
    
    parser.add_argument("--debug", type=int, default=0)
    
    args = parser.parse_args()
    
    dataset_test = PartnetEpc("test", "Mouse", args)