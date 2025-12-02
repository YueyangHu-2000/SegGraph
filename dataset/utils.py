import os
import torch
import numpy as np
import open3d as o3d
# from pytorch3d.io import IO


DATA_PATH = {
    "test":"/data/huyy23/dataset/partslip/PartSTAD_test_2",
    "val":"/data/huyy23/dataset/partslip/PartSTAD_test_2",
    "few_shot": "/data/huyy23/dataset/partslip/PartSTAD_few_shot_2"
}
ORI_DATA_PATH = {
    "test": "/data/huyy23/dataset/partslip/test",
    "val": "/data/huyy23/dataset/partslip/test",
    "few_shot":"/data/huyy23/dataset/partslip/few_shot"
}

def load_label_from_ori(category, shape_id, split):
    label_path = os.path.join(ORI_DATA_PATH[split], category, shape_id, "label.npy")
    label = np.load(label_path, allow_pickle=True)
    label_dict = label.item()
    label = label_dict['semantic_seg']+1
    return label


def load_pc_from_ori(category, shape_id, split):
    pc_path = os.path.join(ORI_DATA_PATH[split], category, shape_id, "pc.ply")
    pc = o3d.io.read_point_cloud(pc_path)
    xyz = np.asarray(pc.points)
    xyz = xyz - xyz.mean(axis=0)
    xyz = xyz / np.linalg.norm(xyz, axis=1, ord=2).max()

    return xyz

def load_pc_rgb_from_ori(category, shape_id, split):
    pc_path = os.path.join(ORI_DATA_PATH[split], category, shape_id, "pc.ply")
    pc = o3d.io.read_point_cloud(pc_path)
    rgb = np.asarray(pc.colors)
    xyz = np.asarray(pc.points)
    xyz = xyz - xyz.mean(axis=0)
    xyz = xyz / np.linalg.norm(xyz, axis=1, ord=2).max()
    return xyz, rgb
    
    