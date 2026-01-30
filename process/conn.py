import os
from os.path import join as ospj
from dataset.utils import load_label_from_ori
import torch
import numpy as np
# from tmp1.tmp5 import connected_components_bfs 
from tqdm import tqdm
from src.utils import get_seg_color,get_batchlabel_color
import cv2
import h5py

def connected_components_bfs(num_nodes, edges):
    """
    主函数：建图并对连通块染色
    Args:
        num_nodes: 节点数量
        edges: 边的列表 [(u1, v1), (u2, v2), ...]

    Returns:
        labels: 每个节点的标签数组
    """
    # 建图（邻接表表示）
    graph = {i: [] for i in range(num_nodes)}
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    # 初始化
    visited = np.zeros(num_nodes, dtype=bool)  # 访问标记
    labels = np.full(num_nodes, -1)            # 染色情况
    current_label = 0
    
    # 遍历所有节点
    for node in range(num_nodes):
        if not visited[node]:
            bfs(graph, node, visited, labels, current_label)
            current_label += 1
    
    return labels

def get_seen_data(label, idx):
    N = label.shape[0]
    
    # is_seen = get_is_seen(idx, N)
    
    idx_seen = np.unique(idx[idx != -1].flatten())
    
    mapping = -np.ones(N, dtype=idx.dtype)
    mapping[idx_seen] = np.arange(idx_seen.shape[0], dtype=idx.dtype)
    pc_idx_seen = mapping[idx]
    pc_idx_seen[idx == -1] = -1
    
    label_seen = label[idx_seen]

    return label_seen, pc_idx_seen

class Graph():
    def __init__(self, n_view, max_label):
        self.n_view = n_view
        self.n_label = max_label+1
        self.n_node = self.n_view*self.n_label
        self.edge = []
    
    def get_id(self, view, label):
        return view*self.n_label+label
    
    def add(self, u,v):
        self.edge.append((u,v))
    
    def get_edge(self):
        return self.edge
def get_pc_mask_online(gt_label, pc_idx, mask_label):
    n_point = gt_label.shape[0]
    n_view=10
    cnt = np.zeros((n_view, gt_label.shape[0])).astype(np.int32)
    pc_label = -1*np.ones((n_view, gt_label.shape[0])).astype(np.int32)
    # ccnt = 0
    for view in range(n_view):
        for label in range(mask_label.max()+1):
            ind = mask_label[view]==label
            if ind.sum()!=0:
                # ccnt+=1            
                pc_ind = pc_idx[view,ind]
                pc_ind = pc_ind[pc_ind!=-1]
                cnt[view,pc_ind]+=1
                pc_label[view,pc_ind]=label
    # print("ccnt= ", ccnt)
    graph = Graph(n_view, mask_label.max())
    n_node = graph.n_node
    conn_cnt = np.zeros((n_node, n_node))
    
    
    for i in tqdm(range(n_point)):
        if cnt[:,i].sum()==0:
            continue
        nodes = []
        for view in range(n_view):
            if cnt[view,i]!=1:
                continue
            curr_node_id = graph.get_id(view, pc_label[view,i])
            for node in nodes:
                conn_cnt[node, curr_node_id]+=1
            nodes.append(curr_node_id)
        
    for i in range(n_node):
        for j in range(n_node):
            if conn_cnt[i,j]+conn_cnt[j,i]>300:
                graph.add(i,j)

    new_label = connected_components_bfs(graph.n_node, graph.get_edge())
    # print(len(new_label))
    new_mask_label = np.zeros_like(mask_label)
    for view in range(n_view):
        for label in range(mask_label.max()+1):
            ind = mask_label[view]==label
            node_id = graph.get_id(view, label)
            new_mask_label[view,ind]=new_label[node_id]
    
    unique_labels = np.unique(new_mask_label[pc_idx!=-1].reshape(-1))  # [1, 2, 3]
    label_map = np.zeros(new_mask_label.max()+1).astype(np.int32)
    label_map[unique_labels] = np.arange(unique_labels.shape[0])
    new_mask_label = label_map[new_mask_label]
    
    
    pc_mask = []
    for j in range(new_mask_label[view].max()+1):
        label = np.zeros(n_point).astype(np.int32)
        for view in range(n_view):
            ind = pc_idx[view,new_mask_label[view]==j]
            ind = ind[ind!=-1]
            label[ind]=1
        indices = np.where(label != 0)[0]
        if len(indices) >0:
            # print(len(indices))
            pc_mask.append(indices)
    return pc_mask

def get_pc_mask(category, shape_id, split):
    root_path = f"/raid0/yyhu/dataset/partnetE/PartSLIP/data/PartSTAD_{split}_2/rendered_pc/{category}/{shape_id}"
    gt_label = load_label_from_ori(category, shape_id, split)
    pc_idx = np.load(ospj(root_path, "img_meta","pc_idx.npy"))
    mask_path = f"/raid0/yyhu/dataset/partnetE/PartSLIP/data/PartSTAD_{split}_2/sam_uniform_point_prompts/{category}/{shape_id}/mask.npy"
    save_path = f"/raid0/yyhu/dataset/partnetE/PartSLIP/data/PartSTAD_{split}_2/sam_uniform_point_prompts/{category}/{shape_id}/pc_mask_seen.h5"
    
    gt_label, pc_idx = get_seen_data(gt_label, pc_idx)
    
    mask_label = np.load(mask_path)
        
    n_point = gt_label.shape[0]
    n_view=10
    cnt = np.zeros((n_view, gt_label.shape[0])).astype(np.int32)
    pc_label = -1*np.ones((n_view, gt_label.shape[0])).astype(np.int32)
    # ccnt = 0
    for view in range(n_view):
        for label in range(mask_label.max()+1):
            ind = mask_label[view]==label
            if ind.sum()!=0:
                # ccnt+=1            
                pc_ind = pc_idx[view,ind]
                pc_ind = pc_ind[pc_ind!=-1]
                cnt[view,pc_ind]+=1
                pc_label[view,pc_ind]=label
    # print("ccnt= ", ccnt)
    graph = Graph(n_view, mask_label.max())
    n_node = graph.n_node
    conn_cnt = np.zeros((n_node, n_node))
    
    
    for i in tqdm(range(n_point)):
        if cnt[:,i].sum()==0:
            continue
        nodes = []
        for view in range(n_view):
            if cnt[view,i]!=1:
                continue
            curr_node_id = graph.get_id(view, pc_label[view,i])
            for node in nodes:
                conn_cnt[node, curr_node_id]+=1
            nodes.append(curr_node_id)
        
    for i in range(n_node):
        for j in range(n_node):
            if conn_cnt[i,j]+conn_cnt[j,i]>300:
                graph.add(i,j)

    new_label = connected_components_bfs(graph.n_node, graph.get_edge())
    # print(len(new_label))
    new_mask_label = np.zeros_like(mask_label)
    for view in range(n_view):
        for label in range(mask_label.max()+1):
            ind = mask_label[view]==label
            node_id = graph.get_id(view, label)
            new_mask_label[view,ind]=new_label[node_id]
    
    unique_labels = np.unique(new_mask_label[pc_idx!=-1].reshape(-1))  # [1, 2, 3]
    label_map = np.zeros(new_mask_label.max()+1).astype(np.int32)
    label_map[unique_labels] = np.arange(unique_labels.shape[0])
    new_mask_label = label_map[new_mask_label]
    
    
    pc_mask = []
    for j in range(new_mask_label[view].max()+1):
        label = np.zeros(n_point).astype(np.int32)
        for view in range(n_view):
            ind = pc_idx[view,new_mask_label[view]==j]
            ind = ind[ind!=-1]
            label[ind]=1
        indices = np.where(label != 0)[0]
        if len(indices) >0:
            # print(len(indices))
            pc_mask.append(indices)    
    # return 
    with h5py.File(save_path, "w") as f:
        for i, array in enumerate(pc_mask):
            f.create_dataset(f"array_{i}", data=array)
            
    """
    exit(0)
    rgb_new_mask_label = (get_batchlabel_color(new_mask_label)*255).astype(np.uint8)
    rgb_mask_label = (get_batchlabel_color(mask_label)*255).astype(np.uint8)
    label = gt_label[pc_idx]
    rgb_label = (get_batchlabel_color(label)*255).astype(np.uint8)
    
    print(new_mask_label.max()) 
    for view in range(n_view):
        ind = pc_idx[view]==-1
        
        for j in range(new_mask_label[view].max()):
            ind = new_mask_label[view]!=j
            rgb = rgb_new_mask_label[view].copy()
            rgb[ind]=np.array([0,0,0])
            cnt = (~ind).sum()
            path = f"output/conn_mask3/{view}"
            os.makedirs(path, exist_ok=True)
            cv2.imwrite(ospj(path,f"{j}_{cnt}.png"), rgb)
            
        
        ind = pc_idx[view]==-1
        rgb = rgb_new_mask_label[view]
        rgb[ind]=np.array([0,0,0])
        cv2.imwrite(f"output/conn_mask/{view}_mask.png", rgb)
        
        rgb = rgb_mask_label[view]
        rgb[ind]=np.array([0,0,0])
        cv2.imwrite(f"output/conn_mask/{view}_mask_ori.png", rgb)
        
        rgb = rgb_label[view]         
        rgb[ind]=np.array([0,0,0])
        cv2.imwrite(f"output/conn_mask/{view}_gt.png", rgb)
    print("good")
    """


DATA_ROOT_PATH  = {
    "few_shot":"/raid0/yyhu/dataset/partnetE/PartSLIP/data/PartSTAD_few_shot_2",
    "test": "/raid0/yyhu/dataset/partnetE/PartSLIP/data/PartSTAD_test_2"
}
if __name__ == "__main__":
    
    # max(mask_label)=458
    # category = "CoffeeMachine" # "USB" #'Keyboard' # "Cart"
    # shape_id = "103043" # "100061" #'12738'    # "100858"
    # split = "test"
    
    
    # all_categories = ['Cart', 'Laptop', 'Refrigerator', 'Phone', 'CoffeeMachine', 'Bottle', 'Box', 'Bucket', 'Camera', 'Chair', 'Clock', 'Dishwasher', 'Dispenser', 'Display', 'Door', 'Eyeglasses', 'Faucet', 'FoldingChair', 'Globe', 'Kettle', 'Keyboard', 'KitchenPot', 'Knife', 'Lamp', 'Lighter', 'Microwave', 'Mouse', 'Oven', 'Pen', 'Pliers', 'Printer', 'Remote', 'Safe', 'Scissors', 'Stapler', 'Suitcase', 'Switch', 'Table', 'Toaster', 'Toilet', 'TrashCan', 'USB', 'WashingMachine', 'Window', 'StorageFurniture']
    all_categories = ['Bottle', 'Box', 'Bucket', 'Camera', 'Cart', 'Chair', 'Clock', 'CoffeeMachine', 'Dishwasher', 'Dispenser', 'Display', 'Door', 'Eyeglasses', 'Faucet', 'FoldingChair', 'Globe', 'Kettle', 'Keyboard', 'KitchenPot', 'Knife', 'Lamp', 'Laptop', 'Lighter', 'Microwave', 'Mouse', 'Oven', 'Pen', 'Phone', 'Pliers', 'Printer', 'Refrigerator', 'Remote', 'Safe', 'Scissors', 'Stapler', 'Suitcase', 'Switch', 'Table', 'Toaster', 'Toilet', 'TrashCan', 'USB', 'WashingMachine', 'Window', 'StorageFurniture']

    for category in all_categories:
        for split in ["few_shot","test"]:
            data_root_path = DATA_ROOT_PATH[split]
            img_root_path = ospj(data_root_path, "rendered_pc")
            cate_path = ospj(img_root_path,category)
            shape_id_list = os.listdir(cate_path)
            for ii, shape_id in tqdm(enumerate(shape_id_list), total=len(shape_id_list), desc=f"{category}_{split}"):
                get_pc_mask(category, shape_id, split)
            
            
            
    # save_path = f"/raid0/yyhu/dataset/partnetE/PartSLIP/data/PartSTAD_{split}_2/sam_uniform_point_prompts/{category}/{shape_id}/pc_mask.h5"
    # with h5py.File(save_path, "r") as f:
    #     recovered_list = [f[key][()] for key in f.keys()]
    # print("good")