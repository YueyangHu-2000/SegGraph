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
from dataset.utils import DATA_PATH, load_label_from_ori, load_pc_from_ori
from src.lib_o3d import batch_geo_feature
from pytorch3d.ops import sample_farthest_points
from process.conn import get_pc_mask_online
import pickle

import h5py

all_categories = ['Phone','CoffeeMachine','Laptop',  'Bottle',  'Cart', 'Refrigerator',  'Box', 'Bucket', 'Camera', 'Chair', 'Clock', 'Dishwasher', 'Dispenser', 'Display', 'Door', 'Eyeglasses', 'Faucet', 'FoldingChair', 'Globe', 'Kettle', 'Keyboard', 'KitchenPot', 'Knife', 'Lamp', 'Lighter', 'Microwave', 'Mouse', 'Oven', 'Pen', 'Pliers', 'Printer', 'Remote', 'Safe', 'Scissors', 'Stapler', 'Suitcase', 'Switch', 'Table', 'Toaster', 'Toilet', 'TrashCan', 'USB', 'WashingMachine', 'Window', 'StorageFurniture']

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

# from torkit3d.ops.sample_farthest_points import sample_farthest_points
# def fps(pc,num_groups = 2048):
#     pc = torch.from_numpy(pc).unsqueeze(dim=0).cuda()
#     fps_idx = sample_farthest_points(pc, num_groups)
#     centers = batch_index_select(pc, fps_idx, dim=1)
#     return centers.squeeze(dim=0).cpu().numpy(), fps_idx.squeeze(dim=0).cpu().numpy()


def fps(xyz, K=2048):
    xyz = torch.from_numpy(xyz).cuda()
    xyz, sampled_indices = sample_farthest_points(xyz[None,:], K=K)
    return xyz[0].cpu().numpy(), sampled_indices[0].cpu().numpy()

def nearest_neighbors(point_cloud, sampled_indices, k=1):
    """
    计算点云 `point_cloud` (N,3) 到采样点 `sampled_indices` (M,) 的最近邻。
    
    参数:
        point_cloud (torch.Tensor): N x 3 点云
        sampled_indices (torch.Tensor): M x 1 采样点索引
        k (int): 选择最近的 k 个邻居，默认 k=1
    
    返回:
        nearest_indices (torch.Tensor): N x k 最近邻索引
        neighborhoods (torch.Tensor): M x k 对应的邻域索引
    """
    point_cloud = torch.from_numpy(point_cloud)
    sampled_indices = torch.from_numpy(sampled_indices)
    sampled_points = point_cloud[sampled_indices]  # 取出采样点 (M,3)
    
    # 计算所有点到采样点的欧式距离 (N, M)
    dists = torch.cdist(point_cloud, sampled_points, p=2)
    
    # 找到每个点最近的采样点索引 (N,)
    nearest_indices = dists.argmin(dim=1)# 转置后 M x N
    
    return nearest_indices.numpy()

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
    
    # is_seen = get_is_seen(idx, N)
    
    idx_seen = np.unique(idx[idx != -1].flatten())
    
    mapping = -np.ones(N, dtype=idx.dtype)
    mapping[idx_seen] = np.arange(idx_seen.shape[0], dtype=idx.dtype)
    pc_idx_seen = mapping[idx]
    pc_idx_seen[idx == -1] = -1
    
    pc_seen = pc[idx_seen]
    label_seen = label[idx_seen]
    coor_seen = coor[:, idx_seen]
    # is_seen = is_seen[:, idx_seen]
    
    all_indices = np.arange(N)
    idx_unseen = ~np.isin(all_indices, idx_seen)
    pc_unseen = pc[idx_unseen]
    pc_label_unseen = label[idx_unseen]

    return pc_seen, label_seen, pc_unseen, pc_label_unseen, pc_idx_seen, coor_seen

def sample_pc(pc, label, idx, coor, pc_fpfh, pc_norm, K=2048*16):
    N = label.shape[0]
    
    # is_seen = get_is_seen(idx, N)
    
    # idx_seen = np.unique(idx[idx != -1].flatten())
    # pc_sampled, idx_sampled = fps(pc, K)
    if N<=K:
        return pc_seen, label_seen, pc_idx_seen, coor_seen, pc_fpfh, pc_norm
        
    idx_sampled = torch.randperm(N)[:K]
    
    mapping = -np.ones(N, dtype=idx.dtype)
    mapping[idx_sampled] = np.arange(idx_sampled.shape[0], dtype=idx.dtype)
    pc_idx_seen = mapping[idx]
    pc_idx_seen[idx == -1] = -1
    
    pc_seen = pc[idx_sampled]
    label_seen = label[idx_sampled]
    coor_seen = coor[:, idx_sampled]
    # is_seen = is_seen[:, idx_sampled]
    pc_fpfh = pc_fpfh[idx_sampled]
    pc_norm = pc_norm[idx_sampled]
    
    return pc_seen, label_seen, pc_idx_seen, coor_seen, pc_fpfh, pc_norm

def get_fpfh(pc, voxel_size=0.008):
    pc = torch.from_numpy(pc)
    normal, fpfh = batch_geo_feature(pc[None,:,:], voxel_size=0.008)
    return normal[0].cpu().numpy(), fpfh[0].cpu().numpy()


def get_ball_index(pc):
    pc = torch.from_numpy(pc).cuda().unsqueeze(dim=0).float()
    idx = ball_query(pc, pc, K=20, radius=0.01)
    return idx[1].squeeze(dim=0).cpu().numpy()

def calc_mask_conf(mask_label):
    hist = np.bincount(mask_label, minlength=10)
    lb = np.argmax(hist)
    conf = hist[lb]/(hist.sum()+1e-6)
    return lb, conf

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
    
def get_PN(mask_num, mask_group_ind, inverse_mapping, mask_area, flyd_edges, mask_label, id2mask, pc_idx, img, pc_label):
    
    valid_mask = []
    for i in range(mask_num):
        view, label = id2mask[i]
        img_ind = mask_label[view]==label
        pc_ind = pc_idx[view,img_ind]
        pc_ind = pc_ind[pc_ind!=-1]
        valid_mask.append(pc_ind.shape[0]/img_ind.sum()>0.9)
    valid_mask = np.array(valid_mask)
    mask_area = np.array(mask_area)
    valid_mask = valid_mask & (mask_area>100)
    PN_triples = []  # [(anchor_idx, pos_idx, neg_idx)]
    
    def save_img(i, mask_label, id2mask, img, save_dir):
        view, label = id2mask[i]
        img_ind = mask_label[view]==label
        cur_img = (img[view]*255).astype(np.uint8)
        cur_img[img_ind] = np.array([128,0,255])
        cv2.imwrite(save_dir, cur_img)
        
    save_dir = "./output/PN"
    os.makedirs(save_dir, exist_ok=True)
    for i in range(mask_num):
        if not valid_mask[i] or mask_area[i]>5000:
            continue
        pc_ind = mask_group_ind[i]
        num_points = len(pc_ind)
        if num_points < 2:
            continue
        pc_ind = inverse_mapping[pc_ind]

        pos_sample_idx = np.random.choice(pc_ind, size=num_points, replace=True) # shape: [num_points]

        # 构造负样本 mask 的采样概率
        tmp_mask_area = np.abs(mask_area - mask_area[i])
        valid_neg_mask = (flyd_edges[i] == 0) & valid_mask
        tmp_mask_area[~valid_neg_mask] = 0

        if np.sum(tmp_mask_area) == 0:
            continue  # 没有可采的负样本 mask

        # prob = tmp_mask_area / np.sum(tmp_mask_area)
        alpha = 1.0  
        prob = tmp_mask_area ** alpha
        prob = prob / np.sum(prob)

        # save_img(i, mask_label, id2mask, img, os.path.join(save_dir,"cur.png"))
        
        # 为每个 anchor 采一个负样本 mask，再采其中一个点
        neg_mask_ids = np.random.choice(mask_num, size=num_points, p=prob)
        neg_sample_idx = []
        for id, neg_id in enumerate(neg_mask_ids):
            neg_pc = mask_group_ind[neg_id]
            
            # if i==16:
            #     save_img(neg_id , mask_label, id2mask, img, os.path.join(save_dir,"neg.png"))
            if len(neg_pc) == 0:
                neg_sample_idx.append(-1)  # 占位标记无效
            else:
                neg_pc = inverse_mapping[neg_pc]
                neg_sample_idx.append(np.random.choice(neg_pc, replace=True))
            aa = 1
        aa=1

        # 过滤掉无效负样本
        neg_sample_idx = np.array(neg_sample_idx)
        good_mask = neg_sample_idx != -1
        # print(good_mask.shape)
        anchors = pc_ind[good_mask]
        positives = pos_sample_idx[good_mask]
        negatives = neg_sample_idx[good_mask]
        
        # a_label = pc_label[np.array(anchors)]
        # n_label = pc_label[np.array(negatives)]
        # acc = (a_label!=n_label).sum()/a_label.shape[0]
        

        PN_triples.extend(zip(anchors, positives, negatives))
    return np.array(PN_triples)

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
        pass

class PartnetEpc(Dataset):
    def __init__(self, split, category, args, save_data=False, show_figure=False):
        super().__init__()
        self.split = split
        self.category = category
        self.data_path = DATA_PATH[self.split]
        self.show_figure = show_figure
        
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
    
    
    def get_pc_ind(self, view, label, mask_label, pc_idx, coords, nearest_index, grouped_indices):
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
                ret_ind.append(g_ind[pc_seen[g_ind]==1])
            else:
                ret_coords = coords[view, g_ind].astype(np.int32)
                in_mask = img_ind[ret_coords[:,1], ret_coords[:,0]]
                ret_ind.append(g_ind[in_mask])
        # ret_ind = torch.cat(ret_ind)
        return np.concatenate(ret_ind)
        
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
    
    def get_conf_edges(self, graph, pc_label, pseudo_label):
        edges_cnt = graph["edges_cnt"]
        mask_area = graph["mask_area"]
        adj_edges = graph["adj_edges"]
        self_i  = torch.arange(edges_cnt.shape[0])
        edges_cnt[self_i, self_i] = mask_area
        n_mask = max(graph["mask2id"].values())+1
        mask_pc_ind = graph["mask_pc_ind"]
        edges_adjacency = adj_edges
        edges = np.zeros((n_mask, n_mask))
        ratio_bar = 0.05
        cnt1, cnt2, cnt3 = 0,0,0
        
        conn_label = pc_label if not self.use_pseudo_label else pseudo_label+1
        # conn_label = pc_label if self.split=="few_shot" else pseudo_label+1
        # conn_label = pc_label
        for i in range(edges.shape[0]):
            pc_ind_i = mask_pc_ind[i]
            lb_i, conf_i = calc_mask_conf(conn_label[pc_ind_i]) if pc_ind_i.size>0 else (0,0)
            edges[i,i]=1 
            cnt1+=1
            for j in range(i+1, edges.shape[1]):
                ri = edges_cnt[i,j] / mask_area[i]
                rj = edges_cnt[i,j] / mask_area[j]
                pc_ind_j = mask_pc_ind[j]
                lb_j, conf_j = calc_mask_conf(conn_label[pc_ind_j])  if pc_ind_j.size>0 else (0,0)
                if ri>=ratio_bar and rj >=ratio_bar : #and (conf_i>=0.9 and conf_j>=0.9 and lb_i==lb_j): # and edges_cnt[i,j]>50:
                    edges[i,j]=1
                    edges[j,i]=1
                    cnt1+=1
                elif self.args.gt_label_edge and edges_adjacency[i,j]==1:
                # elif self.split == "few_shot" and edges_adjacency[i,j]==1:
                    if (conf_i>=0.9 and conf_j>=0.9 and lb_i==lb_j) : #or torch.rand(1).item()>0.8:
                        edges[i,j]=1
                        edges[j,i]=1
                        cnt2+=1
                    
                    else:
                        edges[i,j]=0
                        edges[j,i]=0
                        cnt3+=1
                elif  self.args.ps_label_edge and edges_adjacency[i,j]==1:
                # elif self.split != "few_shot" and edges_adjacency[i,j]==1:
                    if conf_i>=0.9 and conf_j>=0.9 and lb_i!=0 and lb_i==lb_j:
                        # print("link psuedo !!!!!!!!!!!!!!!!!!!!!!")
                        edges[i,j]=1
                        edges[j,i]=1
                        cnt2+=1
                    else:
                        edges[i,j]=0
                        edges[j,i]=0
                        cnt3+=1
                else:
                    edges[i,j]=0
                    edges[j,i]=0
        
        if self.args.ave_inter_mask:
            # print("floyd--------------------------->")
            print(n_mask)
            for k in range(n_mask):
                for i in range(n_mask):
                    if edges[i,k]==0:
                        continue
                    for j in range(n_mask):
                        if edges[i,k]==1 and edges[k,j]==1:
                            edges[i,j]=1
                        
        # print(cnt1, cnt2, cnt3)
        return edges, get_edge_index(edges)
        
        
        
    def load_data(self, save_data = False):
        
        cate_path = os.path.join(self.data_path, "rendered_pc", self.category)

        if self.split == "val":
            val_list = json.load(open("val_list.json"))
            shape_id_list = val_list[self.category]
        else:
            shape_id_list=os.listdir(cate_path)
        if self.show_figure:
            shape_id_list = ["179"]
        shape_id_list = sorted(shape_id_list)
        shape_list = []
        # shape_id_list  = [shape_id_list[33]]
        for ii, shape_id in tqdm(enumerate(shape_id_list), total=len(shape_id_list), desc=f"loading {self.category}_{self.split} data"):
            img_path = os.path.join(cate_path, shape_id, "rendered_img")
            
            if not self.use_cache:
                img = load_img(img_path)
            else:
                img_path = f"{self.data_path}/sam_uniform_point_prompts/{self.category}/{shape_id}/seen_dino_feature.npy"
                img = np.load(img_path)
            pc = load_pc_from_ori(self.category, shape_id, self.split)
            pc_label = load_label_from_ori(self.category, shape_id, self.split)
                       
            img_meta_path = f"{self.data_path}/rendered_pc/{self.category}/{shape_id}/img_meta"
            pc_idx = np.load(os.path.join(img_meta_path, "pc_idx.npy"))
            coords = np.load(os.path.join(img_meta_path, "screen_coords.npy"))
            
            pc, pc_label, unseen_pc, unseen_pc_label, pc_idx, coords= get_seen_data(pc, pc_label, pc_idx, coords)
            
            mask_path = f"{self.data_path}/sam_uniform_point_prompts/{self.category}/{shape_id}"
            mask_label = np.load(os.path.join(mask_path, "mask_2.npy")).astype(np.int32)
             
            # graph_path = f"{self.data_path}/sam_uniform_point_prompts/{self.category}/{shape_id}/graph_seen.pkl"
            # with open(graph_path, "rb") as f:
            #     graph = pickle.load(f)
            #     graph["weak_edges"] = graph["edges"].copy()
            
            graph = {}
            graph_path = f"{self.data_path}/sam_uniform_point_prompts/{self.category}/{shape_id}/graph_cnt.pkl"
            with open(graph_path, "rb") as f:
                mask_area, mask2id, id2mask, edges_cnt = pickle.load(f)
                graph["edges_cnt"] = edges_cnt
                graph["mask_area"] = mask_area
                graph["mask2id"] = mask2id
                graph["id2mask"] = id2mask
            
            if self.eliminate_sparseness:
                graph_path = f"{self.data_path}/sam_uniform_point_prompts/{self.category}/{shape_id}/pc_mask_ind_eli.pkl"
            else:
                graph_path = f"{self.data_path}/sam_uniform_point_prompts/{self.category}/{shape_id}/pc_mask_ind.pkl"
            with open(graph_path, "rb") as f:
                mask_pc_ind = pickle.load(f)
                graph["mask_pc_ind"] = mask_pc_ind["mask_pc_ind"]
            
            adj_edges_path = f"{self.data_path}/sam_uniform_point_prompts/{self.category}/{shape_id}/adj_edges.npy"
            adj_edges = np.load(adj_edges_path)
            graph["adj_edges"] = adj_edges
            
            edges = edges_cnt.copy().astype(np.int64)
            edges[edges!=1] = 1
            edges |= adj_edges
            graph["edges"] = edges
            graph["weak_edges"] = edges
            
            
            if self.use_pseudo_label:
                pseudo_label_path = f"{self.data_path}/sam_uniform_point_prompts/{self.category}/{shape_id}/base_seen_pseudo_label.npy"
                pseudo_label = np.load(pseudo_label_path)
            else:
                pseudo_label = None
            
            conf_edges, conf_edge_index = self.get_conf_edges(graph, pc_label, pseudo_label)
            if self.get_conf_edges:
                graph["edges"] = conf_edges
                graph["edge_index"] = conf_edge_index
                
            if self.args.All_graph:
                
                sampled_data_path = f"{self.data_path}/sam_uniform_point_prompts/{self.category}/{shape_id}/seen_randsampled204810.npy"
                with open(sampled_data_path, 'rb') as f:
                    sampled_data = pickle.load(f)
                centers = sampled_data["sampled_index"]
                nearest_index = sampled_data["nearest_index"]
                # grouped_indices = sampled_data["grouped_indices"]
                if nearest_index.shape[-1] == 1:
                    nearest_index = np.squeeze(nearest_index, axis=-1)
                mapping = -np.ones(pc.shape[0], dtype = centers.dtype)
                mapping[centers] = np.arange(centers.shape[0], dtype = centers.dtype)
                
                inverse_mapping = np.empty(centers.shape[0], dtype=centers.dtype)
                inverse_mapping[np.arange(centers.shape[0])] = centers
                
                graph["nearest_index"] = nearest_index
                # graph["grouped_indices"] = grouped_indices
                graph["centers"] = centers
                mask_num = max(graph["mask2id"].values())+1
                mask_group_ind = []
                
                id2mask = graph["id2mask"]
                mask_pc_ind = graph["mask_pc_ind"]
 
                for i in range(mask_num):
                    view, label = id2mask[i]
                    pc_ind_i = mask_pc_ind[i]
                    inter = np.intersect1d(pc_ind_i, centers)
                    if inter.size > 0:
                        mask_group_ind.append(mapping[inter])
                    else:
                        mask_group_ind.append(np.empty((0)))
                graph["mask_group_ind"] = mask_group_ind
                                
                n_group = centers.shape[0]
                edge_index = []
                n_mask = max(mask2id.values())+1
                edges = graph["edges"]
                source_pt_list = []
                target_pt_list = []
                source_pt_list_maskNode = []
                target_pt_list_maskNode = []
                source_pt_list_maskNode_weak = []
                target_pt_list_maskNode_weak = []
                for i in range(n_mask):
                    group_ind = mask_group_ind[i]
                    if group_ind.size>0:
                        for j in group_ind:
                            source_pt_list.append(j)
                            target_pt_list.append(n_group+i)
                            source_pt_list.append(n_group+i)
                            target_pt_list.append(j)
                cnt = 0
                for i in range(n_mask):
                    for j in range(n_mask):
                        if edges[i,j]==1 and mask_group_ind[i].size>0 and mask_group_ind[j].size>0:
                            source_pt_list_maskNode.append(i)
                            target_pt_list_maskNode.append(j)
                            source_pt_list_maskNode.append(j)
                            target_pt_list_maskNode.append(i)          
                            cnt+=1
                
                cnt2 = 0
                if self.args.conf_label_edge:
                    weak_edges = graph["weak_edges"]
                    for i in range(n_mask):
                        for j in range(n_mask):
                            if weak_edges[i,j]==1 and mask_group_ind[i].size>0 and mask_group_ind[j].size>0:
                                source_pt_list_maskNode_weak.append(i)
                                target_pt_list_maskNode_weak.append(j)
                                source_pt_list_maskNode_weak.append(j)
                                target_pt_list_maskNode_weak.append(i)
                                cnt2+=1

                # print(f"mask_edge cnt : {cnt}, weak_edge cnt : {cnt2}")
                edge_index = torch.stack([torch.tensor(source_pt_list), torch.tensor(target_pt_list)])                  
                edge_index_maskNode = torch.stack([torch.tensor(source_pt_list_maskNode), torch.tensor(target_pt_list_maskNode)])
                edge_index_maskNode_weak = torch.stack([torch.tensor(source_pt_list_maskNode_weak), torch.tensor(target_pt_list_maskNode_weak)])                  
                                  
                graph["edge_index"] = edge_index
                graph["edge_index_maskNode"] = edge_index_maskNode
                graph["edge_index_maskNode_weak"] = edge_index_maskNode_weak
                
                flyd_edges = conf_edges.copy()
                for k in range(n_mask):
                    for i in range(n_mask):
                        if flyd_edges[i,k]==0:
                            continue
                        for j in range(n_mask):
                            if flyd_edges[k,j]==1:
                                flyd_edges[i,j]=1
                graph["flyd_edges"] = flyd_edges
                
                if self.args.use_triplet_loss:
                    PN_tri = get_PN(mask_num, mask_group_ind, inverse_mapping, mask_area,flyd_edges, mask_label, id2mask, pc_idx, img, pc_label)
                    graph["PN_tri"] = PN_tri

            if self.back_to_edges:
                mask_entropy_path = f"{self.data_path}/sam_uniform_point_prompts/{self.category}/{shape_id}/mask_entropy.npy"
                mask_entropy = np.load(mask_entropy_path)
                weight = np.exp(-1 * mask_entropy)
                weight[mask_entropy==-1]=0
                graph["weight"] = weight
                edges *= weight[:, None]
                self_i  = torch.arange(edges_cnt.shape[0])
                edges_cnt[self_i, self_i] = 1
                graph["edges"] = edges
                    # print("good work")
                    
            if self.use_pseudo_label:
                pc_norm = pseudo_label
            else:
                pc_norm = pc_label
            
            if self.args.use_ball_propagate:
                ball_ind = get_ball_index(pc)
                pc_norm = ball_ind
                
            # pc_norm  = f"{self.data_path}/sam_uniform_point_prompts/{self.category}/{shape_id}"  
            
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

def data_process(args):
    for category in tqdm(all_categories):
        dataset = PartnetEpc("few_shot", category, args, save_data=True)
        dataset = PartnetEpc("test", category, args, save_data=True)
    exit(0)

def calc_conn_ratio(args):
    def count_connected_components(adj_matrix):
        n = adj_matrix.shape[0]
        visited = np.zeros(n, dtype=bool)

        def dfs(node):
            visited[node] = True
            neighbors = np.where(adj_matrix[node] > 0)[0]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    dfs(neighbor)

        count = 0
        for i in range(n):
            if not visited[i]:
                dfs(i)
                count += 1
        return count
    # category = "Clock"
    # all_categories = ['KitchenPot','Lighter']
    print(len(all_categories))
    data_to_write = []
    lines = []
    edges_ratio = []
    for category in all_categories:
        dataset = PartnetEpc("test", category, args)
        ratios = []
        for iii in tqdm(range(len(dataset))):
            ii, pc, pc_label, unseen_pc, unseen_pc_label,img, mask_label,pc_idx,coords, graph, pc_norm = dataset[iii]
            from src.utils import get_batchlabel_color
            # from src.render_pc import render_pc
            mask2id = graph["mask2id"]
            id2mask = graph["id2mask"]
            mask_pc_ind = graph["mask_pc_ind"]
            # edge_index = graph["edge_index"]
            edges = graph["edges"]
            
            conn_cnt = count_connected_components(edges)
            # print(edges.shape[0], conn_cnt)
            ratios.append(conn_cnt/edges.shape[0])
            edges_ratio.append(edges.sum()/edges.shape[0])
        line = f"{category}, {sum(ratios)/len(ratios)}, {sum(edges_ratio)/len(edges_ratio)}\n"
        print(line)
        lines.append(line)
        with open("./output/conn_ratio.csv","w") as f:
            f.writelines(lines)

def calc_good_bad_edge(args):
    category = "Clock"
    # all_categories = ['KitchenPot','Lighter']
    print(len(all_categories))
    data_to_write = []
    for category in all_categories:
        dataset = PartnetEpc("test", category, args)
        goods = []
        mids = []
        for iii in tqdm(range(len(dataset))):
            ii, pc, pc_label, unseen_pc, unseen_pc_label,img, mask_label,pc_idx,coords, graph, pc_norm = dataset[iii]
            from src.utils import get_batchlabel_color
            # from src.render_pc import render_pc
            mask2id = graph["mask2id"]
            id2mask = graph["id2mask"]
            mask_pc_ind = graph["mask_pc_ind"]
            # edge_index = graph["edge_index"]
            edges = graph["edges"]
            # mask_weight = graph["weight"]
            
            edges_cnt = graph["edges_cnt"]
            mask_area = graph["mask_area"]
            
            n_mask = max(mask2id.values())+1
            n_view=10
            rgb_mask_label = (get_batchlabel_color(mask_label)*255).astype(np.uint8)
            img_label = pc_label[pc_idx]+1
            img_label[pc_idx==-1]=0
            rgb_img_label = (get_batchlabel_color(img_label)*255).astype(np.uint8)
            
            def check_mask(img_ind_i, pc_idx_i):
                cnt1 = (pc_idx_i!=-1).sum()
                cnt2 = img_ind_i.sum()
                return cnt1 / cnt2 > 0.9

            
            good_edge_cnt = 0
            bad_edge_cnt = 0
            mid_edge_cnt = 0
            for i in range(n_mask):
                view, label = id2mask[i]
                img_ind = mask_label[view]==label
                # pc_ind = pc_idx[view,img_ind]
                # pc_ind = pc_ind[pc_ind!=-1]
                pc_ind_i = mask_pc_ind[i]
                if pc_ind_i.size>0:
                    mask_true_label = pc_label[pc_ind_i]
                    hist_i = np.bincount(mask_true_label, minlength=10)
                    lb_i = np.argmax(hist_i)
                    conf_i = hist_i[lb_i]/(hist_i.sum()+1e-6)
                else:
                    conf_i=0  
                cnt = 0
                for j in range(n_mask):
                    if edges[i,j] ==0:
                        continue
                    view, label = id2mask[j]
                    img_ind = mask_label[view]==label
                    # pc_ind = pc_idx[view,img_ind]
                    # pc_ind = pc_ind[pc_ind!=-1]
                    pc_ind_j = mask_pc_ind[j]
                    if pc_ind_j.size>0:
                        mask_true_label = pc_label[pc_ind_j]
                        hist_j = np.bincount(mask_true_label, minlength=10)
                        lb_j = np.argmax(hist_j)
                        conf_j = hist_j[lb_j]/(hist_j.sum()+1e-6)
                    else:
                        conf_j=0
                    if conf_i>0.9 and conf_j>0.9 and lb_i==lb_j:
                        set1 = mask_pc_ind[i]
                        set2 = mask_pc_ind[j]
                        intersection = np.intersect1d(set1, set2)
                        union = np.union1d(set1, set2)
                        if intersection.size/union.size > 0.9:
                            mid_edge_cnt+=1
                        else:
                            good_edge_cnt+=1
                    else:
                        bad_edge_cnt+=1
            goods.append(good_edge_cnt/(good_edge_cnt+bad_edge_cnt+mid_edge_cnt))
            mids.append(mid_edge_cnt/(good_edge_cnt+bad_edge_cnt+mid_edge_cnt))
            
        line = f"{category}, {sum(goods)/len(goods)}, {sum(mids)/len(mids)}\n"
        print(line)
        data_to_write.append(line)
        # if category == "Bottle":
        #     break
        with open("./output/good_mid_ratio005.csv", "w") as f_out:
            f_out.writelines(data_to_write)

def visualize_conn(args):
    category = "CoffeeMachine"
    dataset = PartnetEpc("few_shot", category, args)
    from src.utils import get_batchlabel_color
    
    for iii in tqdm(range(len(dataset))):
        ii, pc, pc_label, unseen_pc, unseen_pc_label,img, mask_label,pc_idx,coords, graph, pc_norm = dataset[iii]
        
        mask2id = graph["mask2id"]
        id2mask = graph["id2mask"]
        mask_pc_ind = graph["mask_pc_ind"]
        edges = graph["edges"]
        adj_edges = graph["adj_edges"]
        edges_cnt = graph["edges_cnt"]
        mask_area = graph["mask_area"]
        n_mask = max(mask2id.values())+1
        n_view=10
        rgb_mask_label = (get_batchlabel_color(mask_label)*255).astype(np.uint8)
        img_label = pc_label[pc_idx]+1
        img_label[pc_idx==-1]=0
        rgb_img_label = (get_batchlabel_color(img_label)*255).astype(np.uint8)
        path = f"output/check_mask_conn/"
        os.makedirs(path, exist_ok=True)
        # for k in range(n_mask):
        #     for p in range(n_mask):
        #         if edges[p,k]==0:
        #             continue
        #         for q in range(n_mask):
        #             if edges[p,k]==1 and edges[k,q]==1:
        #                 edges[p,q]=1
        for i in range(n_mask):
            view, label = id2mask[i]
            img_ind = mask_label[view]==label
            # if not check_mask(img_ind, pc_idx[view, img_ind]):
            #     continue
            img_curr = (img[view]*255).astype(np.uint8)
            cv2.imwrite(os.path.join(path,f"aimg.png"), img_curr)
            img_curr[img_ind] = np.array([20,20,200])
            cv2.imwrite(os.path.join(path,f"vis_curr.png"), img_curr)
            cv2.imwrite(os.path.join(path,f"aimg_label.png"), rgb_img_label[view])
            
            # cnt = 0
            # for j in range(n_mask):
            #     if edges[i,j] ==0:
            #         continue
                
            #     view, label = id2mask[j]
            #     img_ind = mask_label[view]==label
            #     # if not check_mask(img_ind, pc_idx[view, img_ind]):
            #     #     continue
            #     img_curr = (img[view]*255).astype(np.uint8)
            #     img_curr[img_ind] = np.array([20,20,200])
            #     cv2.imwrite(os.path.join(path,f"zvis_curr_{cnt}.png"), img_curr)
            #     print(
            #         f"i={i} "
            #         # f"weight_i={mask_weight[i]:.4f} "
            #         f"cnt={cnt} "
            #         # f"edges[i,j]={edges[i,j]:.4f} "
            #         f"(view, j) = ({view}, {j}) "
            #         # f"weight_j={mask_weight[j]:.4f} "
            #         f"(area_i,area_j,I) = ({mask_area[i]}, {mask_area[j]}, {edges_cnt[i,j]})"
            #         f"(r_i,r_j) = ({edges_cnt[i,j]/mask_area[i]:.4f}, {edges_cnt[i,j]/mask_area[j]:.4f})"
                    
            #     )
            #     cnt+=1 
            # print("good")
        print("good2")

def check_psuedo_label(args):
    from src.utils import get_batchlabel_color
    from test_pc import get_legend
    
    category = "Keyboard"
    meta = json.load(open("PartNetE_meta.json"))
    label_name = ["BK","uncertain","None"]
    label_name.extend(meta[category])
    dataset = PartnetEpc("test", category, args)
    save_dir = "./output/check_psuedo_label"
    os.makedirs(save_dir, exist_ok=True)
    for iii in tqdm(range(len(dataset))):
        ii, pc, pc_label, unseen_pc, unseen_pc_label,img, mask_label,pc_idx,coords, graph, pc_norm = dataset[iii]
        mask2id = graph["mask2id"]
        id2mask = graph["id2mask"]
        mask_pc_ind = graph["mask_pc_ind"]
        edges = graph["edges"]
        edges_cnt = graph["edges_cnt"]
        mask_area = graph["mask_area"]
        n_mask = max(mask2id.values())+1
        psuedo_label = pc_norm
        
        img_label = psuedo_label[pc_idx]+2
        img_label[pc_idx==-1]=0
        rgb_img_label = (get_batchlabel_color(img_label)*255).astype(np.uint8)
        
        for view in range(10):
            rgb = (img[view]*255).astype(np.uint8)
            cv2.imwrite(f"{save_dir}/{view}_img.png", rgb)
            
        for view in range(10):
            rgb = rgb_img_label[view]
            rgb[img_label[view]==0] = np.array([0,0,0])
            rgb = get_legend(rgb, img_label[view], label_name)
            cv2.imwrite(f"{save_dir}/{view}.png", rgb)
        
        img_label = pc_label[pc_idx]+2
        img_label[pc_idx==-1]=0
        rgb_img_label = (get_batchlabel_color(img_label)*255).astype(np.uint8)
        for view in range(10):
            rgb = rgb_img_label[view]
            rgb[img_label[view]==0] = np.array([0,0,0])
            rgb = get_legend(rgb, img_label[view], label_name)
            cv2.imwrite(f"{save_dir}/{view}_gt.png", rgb)  
        print("good")          
        
def render_mask(args):
    from src.render_pc import render_pc
    from src.utils import get_seg_color
    category = "CoffeeMachine"
    dataset = PartnetEpc("test", category, args)
    def render(pc, pc_label, save_dir="output/render", device="cuda:0"):
        pc_colors = get_seg_color(pc_label)
        render_pc(pc, pc_colors, save_dir, device=device)
        
    for iii in tqdm(range(len(dataset))):
        ii, pc, pc_label, unseen_pc, unseen_pc_label,img, mask_label,pc_idx,coords, graph, pc_norm = dataset[iii]
        mask_pc_ind = graph["mask_pc_ind"]
        
        ind = mask_pc_ind[121]
        render(pc[ind], pc_label[ind])
        print("good")
        # mask_num = max(graph["mask2id"].values())+1
        # for i in range(mask_num):
        #     ind = mask_pc_ind[i]
        #     render(pc[ind], pc_label[ind])
        #     print("good")
        
def check_PN_quality(args):
    # all_categories = ["Phone"]
    for category in all_categories:
        data_to_write = []
        dataset = PartnetEpc("test", category, args)
        acc_P = []
        acc_N = []
        for iii in tqdm(range(len(dataset))):
            ii, pc, pc_label, unseen_pc, unseen_pc_label,img, mask_label,pc_idx,coords, graph, pc_norm = dataset[iii]
            PN_tri = graph["PN_tri"]
            # print(PN_tri.shape)
            anchor_label = pc_label[PN_tri[:,0]]
            pos_label = pc_label[PN_tri[:,1]]
            neg_label = pc_label[PN_tri[:,2]]
            accp = (anchor_label==pos_label).sum()/PN_tri.shape[0]
            
            accn = (anchor_label!=neg_label).sum()/PN_tri.shape[0]
            acc_P.append(accp)
            acc_N.append(accn)
            
        line = f"{category}, {sum(acc_P)/len(acc_P)}, {sum(acc_N)/len(acc_N)}\n"
        print(line)
        data_to_write.append(line)
        with open("./output/tri_alpha2.csv", "w") as f_out:
            f_out.writelines(data_to_write)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, nargs='+', default=["CoffeeMachine"])# ["Box","Bucket","Camera","Cart","Clock"]
    # parser.add_argument("--cuda", type=str, default="cuda:2")
    parser.add_argument("--mode", type=str, default="train")  #!!!!! if change test change output_dir
    parser.add_argument("--output_dir", type=str, default="res/tmp5")
    parser.add_argument("--epoch", type=int, default=20) 
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01) 
    
    parser.add_argument("--use_2d_feat", type=int, default=1) 
    parser.add_argument("--use_3d_feat", type=int, default=0)
    
    parser.add_argument("--img_encoder", type=str, default="dinov2")
    parser.add_argument("--use_cache", type=int, default=0)
    
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
    parser.add_argument("--use_pseudo_label", type=int, default=0)
    
    parser.add_argument("--conf_label_edge", type=int, default=1) 
    parser.add_argument("--gt_label_edge", type=int, default=0)
    parser.add_argument("--ps_label_edge", type=int, default=0)
    parser.add_argument("--img_feat_on_mask", type=int, default=0)
    
    
    
    parser.add_argument("--All_graph", type=int, default=1)
    parser.add_argument("--4_graph", type=int, default=0)
    parser.add_argument("--use_ball_propagate", type=int, default=0)
    

    parser.add_argument("--use_proxy_contrast_loss", type=int, default=0)
    parser.add_argument("--use_contrast_loss2", type=int, default=0)
    parser.add_argument("--use_ref_loss", type=int, default=0) 
    parser.add_argument("--use_mask_consist_loss", type=int, default=0) 
    parser.add_argument("--use_triplet_loss", type=int, default=1) 
    
    
    parser.add_argument("--debug", type=int, default=0)
    args = parser.parse_args()
    
    check_PN_quality(args)
    # calc_good_bad_edge(args)
    # check_psuedo_label(args)
    # visualize_conn(args)
    # render_mask(args)
    
    # '/home/huyy23/data/PartSLIP/data/PartSTAD_test_2/sam_uniform_point_prompts/CoffeeMachine/103143/graph_cnt.pkl'