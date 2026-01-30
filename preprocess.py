import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import cv2
from os.path import join as ospj
import torch
import argparse
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from src.render_pc import render_pc
from src.utils import get_seg_color
import open3d as o3d
from tqdm import tqdm
from pytorch3d.ops import sample_farthest_points
import pickle
import time
import json
######################################################
def get_pc_path_list(pc_root_path, process_root_path, args):
    if args.dataset == "PartNetE":
        select_category = ['Bottle', 'Box', 'Bucket', 'Camera', 'Cart', 'Chair', 'Clock', 'CoffeeMachine', 'Dishwasher', 'Dispenser', 'Display', 'Door', 'Eyeglasses', 'Faucet', 'FoldingChair', 'Globe', 'Kettle', 'Keyboard', 'KitchenPot', 'Knife', 'Lamp', 'Laptop', 'Lighter', 'Microwave', 'Mouse', 'Oven', 'Pen', 'Phone', 'Pliers', 'Printer', 'Refrigerator', 'Remote', 'Safe', 'Scissors', 'Stapler', 'Suitcase', 'Switch', 'Table', 'Toaster', 'Toilet', 'TrashCan', 'USB', 'WashingMachine', 'Window', 'StorageFurniture']
        splits = ["few_shot","test"]
        pc_path_list = []
        save_path_list = []
    
        # for split in splits:
        #     split_path = os.path.join(pc_root_path, split)
        #     categories = os.listdir(split_path)
        #     for category in select_category:
        #         category_path = os.path.join(split_path, category)
        #         shapeids = os.listdir(category_path)
        #         for shapeid in shapeids:
        #             pc_path_list.append(os.path.join(pc_root_path, split, category, shapeid, "pc.ply"))
        #             save_path_list.append(os.path.join(process_root_path, category, shapeid))
        path_split = json.load(open("PartNetE_split_path_lists.json"))
        for category in tqdm(select_category, total=len(select_category)):
            for split in ["train", "val","test"]:
                for _, path in path_split[category][split]:
                    
                    if split=="train":
                        source_path = ospj(pc_root_path,  _, "pc.ply")
                    else:
                        source_path = ospj(pc_root_path,  _, "pc.ply")

                    target_path = ospj(process_root_path, path)
                    pc_path_list.append(source_path)
                    save_path_list.append(target_path)
                    
    elif args.dataset == "3DCoMPaTv2":
        select_categories = ["airplane", "bicycle", "chair"]
        pc_path_list = []
        save_path_list = []
        
        path_list = json.load(open("3DCoMPaTv2_path_lists.json", "r"))
        for category in select_categories:
            if category not in path_list.keys():
                continue
            for path in path_list[category]:
                shapeid = os.path.basename(path)
                pc_file_name = ("_").join( shapeid.split("_")[1:]) + ".npz"
                pc_path = os.path.join(pc_root_path, path, pc_file_name)
                save_path = os.path.join(process_root_path, path)
                pc_path_list.append(pc_path)
                save_path_list.append(save_path)
                
    return pc_path_list, save_path_list
  
                                
def load_pc(pc_path, args):
    
    if args.dataset == "PartNetE":
        noise_std = args.noise  # 例如设为 0.01, 0.02, 0.05, 0.10
        pc = o3d.io.read_point_cloud(pc_path)

        xyz = np.asarray(pc.points)
        rgb = np.asarray(pc.colors)

        # 中心化 + 归一化
        xyz = xyz - xyz.mean(axis=0)
        xyz = xyz / np.linalg.norm(xyz, axis=1, ord=2).max()

        # 添加高斯噪声（可选）
        if noise_std > 0:
            noise = np.random.normal(loc=0.0, scale=noise_std, size=xyz.shape)
            xyz = xyz + noise
    elif args.dataset == "3DCoMPaTv2":
        data = np.load(pc_path)
        xyz = data["pointcloud"]
        rgb = data["point_colors"].astype(np.float32) / 255.0
        xyz = xyz - xyz.mean(axis=0)
        xyz = xyz / np.linalg.norm(xyz, axis=1, ord=2).max()
    return xyz, rgb

######################################################

def render_pc_to_img(pc_path_list, save_path_list, device, args):
    for pc_path, save_path in tqdm(zip(pc_path_list, save_path_list), total=len(pc_path_list)):
        save_path = os.path.join(save_path, "rendered_pc")
        meta_save_path = os.path.join(save_path, "img_meta")
        os.makedirs(meta_save_path, exist_ok=True)
        
        xyz, rgb = load_pc(pc_path, args)
        img_dir, pc_idx, screen_coords = render_pc(xyz, rgb, save_path, device)
        np.save(f"{meta_save_path}/pc_idx.npy", pc_idx)
        np.save(f"{meta_save_path}/screen_coords.npy", screen_coords)
        # print(save_path)
        # exit(0)

def load_img(img_path):
    imgs = []
    for i in range(10):
        img = cv2.cvtColor(cv2.imread(os.path.join(img_path, str(i)+".png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        imgs.append(np.expand_dims(img, axis=0))
    imgs = np.concatenate(imgs)
    return imgs

def sam_generate_mask(save_path_list, device):
    print("<loading sam... ", end="")
    sam_checkpoint = "/home/huyy23/project/segment-anything-main/checkpoints/sam_vit_h_4b8939.pth"  # 替换为您本地的模型权重路径
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    mask_generator  = SamAutomaticMaskGenerator(sam, points_per_side=64, points_per_batch=128*2)
    print(" finish!>")
    
    for path_list in tqdm(save_path_list, total=len(save_path_list)):
        img_path = os.path.join(path_list, "rendered_pc","rendered_img")
        sam_path = os.path.join(path_list, "sam_results/masks")
        os.makedirs(sam_path, exist_ok=True)
        print(img_path)
        imgs = load_img(img_path)
        total_time=0
        for i in tqdm(range(10), total=10):
            start_time = time.time()
            masks = mask_generator.generate(imgs[i])
            end_time = time.time()
            total_time += end_time-start_time
            segms = [mask['segmentation'] for mask in masks]
            areas = [mask['area'] for mask in masks]
            scores = [mask['predicted_iou'] for mask in masks]
            
        #     np.savez_compressed(
        #         os.path.join(sam_path, f'mask_{i}.npz'),
        #         segmentation=np.array(segms, dtype=bool),
        #         areas=np.array(areas, dtype=np.int32),
        #         score=np.array(scores, dtype=np.float32)
        # )
        print(f"time: {total_time:.4f} s")

def get_bbox(mask):

    row, col = np.where(mask > 0)
    r_min, r_max = np.min(row), np.max(row)
    c_min, c_max = np.min(col), np.max(col)

    bbox = r_min, r_max, c_min, c_max

    return bbox

def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str
):
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask.astype(bool)).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True

def filt_mask(single_mask,N,W,H):
    all_masks = []
    for i in range(N):
        mask = np.zeros((W,H)).astype(np.int32)
        ind = single_mask==i
        mask[ind]= 1
        
        mask, change = remove_small_regions(mask, 100, 'holes')
        unchange = not change
        mask, change = remove_small_regions(mask, 100, 'islands')
        unchange = unchange and not change
        
        try:
            r_min, r_max, c_min, c_max = get_bbox(mask)
            if np.sum(mask) < 0.1*((r_max-r_min)*(c_max-c_min)):
                continue
        except:
            pass

        mask = mask.astype(np.uint8)

        if np.sum(mask) < 50:
            continue
        all_masks.append(mask)
    return all_masks

def process_non_overlap(data):
    # N = len(data)
    # masks = [x['segmentation'] for x in data]
    # areas = [x['area'] for x in data]
    masks = data['segmentation']
    areas = data['areas']
    N = masks.shape[0]
    masks = np.array(masks)
    areas = np.array(areas)
    sorted_id = np.argsort(areas)[::-1]
    W,H = masks[0].shape[0], masks[0].shape[1]
    single_mask = np.zeros((masks[0].shape[0], masks[0].shape[1])).astype(np.int32)
    
    for i in range(N):
        single_mask[masks[sorted_id[i]]]=i
    
    
    all_masks = filt_mask(single_mask,N,W,H)
    single_mask = np.zeros((masks[0].shape[0], masks[0].shape[1])).astype(np.int32)
    label_cnt = 1
    for i in range(len(all_masks)):
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(all_masks[i], connectivity=8)
        for j in range(1, n_labels):
            if (labels==j).sum()==0:
                print(j, n_labels)
            single_mask[labels==j]=label_cnt
            label_cnt +=1
    
    # all_masks = filt_mask(single_mask,N,W,H)
    # single_mask = np.zeros((masks[0].shape[0], masks[0].shape[1])).astype(np.int32)
    # label_cnt = 1
    # for i in range(len(all_masks)):
    #     n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(all_masks[i], connectivity=8)
    #     for j in range(1, n_labels+1):
    #         single_mask[labels==j]=label_cnt
    #         label_cnt +=1
            
    return single_mask

def process_sam_mask(save_path_list, device):
    
    for save_path in tqdm(save_path_list,total=len(save_path_list)):
        sam_path = os.path.join(save_path, "sam_results")
        sam_mask_path = os.path.join(sam_path, "masks")
        
        labels = []
        for i in range(10):
            masks = np.load(os.path.join(sam_mask_path, f"mask_{i}.npz"))
            label = process_non_overlap(masks)
            labels.append(np.expand_dims(label, axis=0))
            label = label.reshape(-1)
            rgb = get_seg_color(label, label.max()+1)
            rgb = rgb.reshape(800,800,-3)
            rgb  = (rgb  * 255).astype(np.uint8)
            sam_vis_path = os.path.join(sam_path, "visualization")
            os.makedirs(sam_vis_path, exist_ok=True)
            cv2.imwrite(os.path.join(sam_vis_path, f"sam_auto_mask{i}.png"), rgb)
            # print(save_path)
        labels = np.concatenate(labels)
        np.save(os.path.join(sam_path, "mask_2.npy"), labels)

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

def get_graph_cnt(mask_label, pc_idx, save_path):
    n_view = mask_label.shape[0]
    n_point = pc_idx.max()+5
    
    mask2id = {}
    id2mask = {}
    mask_area = []
    id_cnt = 0
    
    cnt = np.zeros((n_view, n_point)).astype(np.int32)
    pc_label = -1*np.ones((n_view, n_point)).astype(np.int32)
    for view in range(n_view):
        bd = mask_label[view].max()+1
        for label in range(bd):
            img_ind = mask_label[view]==label
            if img_ind.sum()>0:
                mask2id[(view, label)] = id_cnt
                id2mask[id_cnt] = (view, label)
                mask_area.append(img_ind.sum())
                id_cnt += 1
                
                pc_ind = pc_idx[view,img_ind]
                pc_ind = pc_ind[pc_ind!=-1]
                cnt[view,pc_ind]+=1
                pc_label[view,pc_ind]=label
    
    edges = np.zeros((id_cnt,id_cnt))
    # haha = 0  
    for i in range(n_point):
        if cnt[:,i].sum()==0:
            continue
        nodes = []
        for view in range(n_view):   
            if cnt[view,i]!=1:
                continue
            # curr_node_id = graph.get_id(view, pc_label[view,i])
            curr_node_id = mask2id[(view,pc_label[view,i])]
            for node in nodes:
                edges[node, curr_node_id]+=1
                edges[curr_node_id, node]+=1
            nodes.append(curr_node_id)
            
    with open(os.path.join(save_path, "graph_cnt.pkl"), 'wb') as f:
        pickle.dump((mask_area, mask2id, id2mask, edges), f)
    
    return mask_area, mask2id, id2mask, edges


def get_seen_data(pc, idx, coor):
    N = pc.shape[0]
    
    # is_seen = get_is_seen(idx, N)
    
    idx_seen = np.unique(idx[idx != -1].flatten())
    
    mapping = -np.ones(N, dtype=idx.dtype)
    mapping[idx_seen] = np.arange(idx_seen.shape[0], dtype=idx.dtype)
    pc_idx_seen = mapping[idx]
    pc_idx_seen[idx == -1] = -1
    
    pc_seen = pc[idx_seen]
    coor_seen = coor[:, idx_seen]
    
    return pc_seen, pc_idx_seen, coor_seen

def get_pc_ind(pc, view, label, mask_label, pc_idx, coords, nearest_index, grouped_indices):
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
    
def get_mask_pc_ind(pc, mask2id, id2mask, mask_label, pc_idx, coords):
    sampled_pc, sampled_index = fps(pc)
    # sampled_pc, sampled_index = random_sample(pc)
    nearest_index = nearest_neighbors(pc, sampled_index)
    unique_values = np.unique(nearest_index)  # 获取唯一值
    grouped_indices = [np.where(nearest_index == val)[0] for val in unique_values]  # 获取索引分组
    # pc_norm = {"nearest_index":nearest_index, "grouped_indices":grouped_indices}
    mask_num = max(mask2id.values())+1
    mask_pc_ind = []
    for i in range(mask_num):
        view, label = id2mask[i]
        pc_ind = get_pc_ind(pc, view, label, mask_label, pc_idx, coords, nearest_index, grouped_indices)
        mask_pc_ind.append(pc_ind)
    return mask_pc_ind

def get_masks_adjacency(mask_label, mask2id, id2mask):
    num_view, W,H = mask_label.shape
    n_mask = max(mask2id.values())+1
    adjacency_matrix = np.zeros((n_mask, n_mask), dtype=int)
    # 获取八连通邻接关系
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    def paint(img, x,y,color):
        for i in range(-4,5):
            for j in range(-4,5):
                nx, ny = x + i, y + j
                if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1]:
                    img[nx,ny] = color
        return img
    for view in range(10):
        image = -1*np.ones((W,H), dtype=np.int32)
        id_list = []
        for id in range(n_mask):
            view_curr, label = id2mask[id]
            if view_curr == view:
                img_ind = mask_label[view_curr]==label
                image[img_ind]=id
                id_list.append(id)
        # assert (image==-1).sum() ==0 
        for label in id_list:
            mask = (image == label).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                for point in contour:
                    x, y = point[0]
                    # 遍历八个邻域
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:
                            neighbor_label = image[ny, nx]
                            if neighbor_label != label and neighbor_label!=-1:
                                adjacency_matrix[label, neighbor_label] = 1
                       
    return adjacency_matrix

def seen_sample(pc, K):
    # sampled_pc, sampled_index = fps(pc, K=K)
    sampled_index = np.random.permutation(pc.shape[0])[:K]
    mapping = -np.ones(pc.shape[0], dtype = sampled_index.dtype)
    mapping[sampled_index] = np.arange(sampled_index.shape[0], dtype = sampled_index.dtype)
    nearest_index = nearest_neighbors(pc, sampled_index)
    
    return sampled_index, nearest_index
                             
def build_graph(pc_path_list, save_path_list, device):
    
    for pc_path, save_path in tqdm(zip(pc_path_list, save_path_list), total=len(save_path_list)):
        
        
        xyz, rgb = load_pc(pc_path, args)
        pc_idx = np.load(f"{save_path}/rendered_pc/img_meta/pc_idx.npy")
        coords = np.load(f"{save_path}/rendered_pc/img_meta/screen_coords.npy")
        
        pc, pc_idx, coords = get_seen_data(xyz, pc_idx, coords)
        
        mask_label = np.load(os.path.join(save_path, "sam_results", "mask_2.npy"))
        mask_area, mask2id, id2mask, edges = get_graph_cnt(mask_label, pc_idx, save_path)
        
        mask_pc_ind = get_mask_pc_ind(pc, mask2id, id2mask, mask_label, pc_idx, coords)
        with open(os.path.join(save_path, "pc_mask_ind_eli_2.pkl"), "wb") as f:
            pickle.dump(mask_pc_ind, f)
         
        adj_edges = get_masks_adjacency(mask_label, mask2id, id2mask)
        np.save(os.path.join(save_path,"adj_edges.npy"), adj_edges)
        
        centers, nearest_index = seen_sample(pc, K=2048*10)
        data2save = {
            "sampled_index": centers,
            "nearest_index": nearest_index
        }
        np.save(os.path.join(save_path, "seen_randsampled204810"), data2save)
        
        # print("--------<", save_path)
        # exit(0)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pc_root_path", type=str, default="/raid0/yyhu/dataset/partnetE/PartSLIP/data")
    parser.add_argument("--process_root_path", type=str, default="/data/huyy23/dataset2/PartSLIP/rebuttal_fastsam")
    parser.add_argument("--dataset", type=str, default="PartNetE")
    parser.add_argument("--noise", type=float, default=-1)
    args = parser.parse_args()
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    if args.noise > 0:
        args.process_root_path = args.process_root_path + "_" +str(args.noise)
        
    pc_path_list, save_path_list = get_pc_path_list(args.pc_root_path, args.process_root_path, args)
    
    os.makedirs(args.process_root_path, exist_ok=True)
    
    render_pc_to_img(pc_path_list, save_path_list, device, args)
    
    sam_generate_mask(save_path_list, device)
    
    process_sam_mask(save_path_list, device)
    
    build_graph(pc_path_list, save_path_list, device)
    pass