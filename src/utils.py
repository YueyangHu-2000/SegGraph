import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

cmap = plt.get_cmap("turbo")


def get_seg_color(seg, num_label=None):
    """_summary_

    Args:
        seg (np.array): N
        num_label (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    num_pts = seg.shape[0]
    segs = np.unique(seg)
    num_segs = segs.shape[0] if num_label is None else num_label
    rgb = np.zeros(num_pts)
    for i in range(num_segs):
        val = (i+1)/(num_segs+1)
        idx = segs[i] if num_label is None else i
        if idx == -1:
            val = 0
        rgb[seg==idx] = val
    
    rgb = cmap(rgb)[:,:3]
    return rgb
# def get_seg_color(seg, num_label=None):
#     """Assigns random bright colors to each segment.

#     Args:
#         seg (np.array): Array of segment labels.
#         num_label (int, optional): Number of labels to consider. Defaults to None.

#     Returns:
#         np.array: RGB values for each point in the point cloud.
#     """
#     num_pts = seg.shape[0]
#     segs = np.unique(seg)
#     num_segs = segs.shape[0] if num_label is None else num_label
#     rgb = np.zeros((num_pts, 3))
    
#     for i in range(num_segs):
#         # Generate a random color ensuring it's bright (no near-black colors)
#         random_color = np.random.rand(3) * 0.6 + 0.4  # Values between 0.3 and 1.0
        
#         # Assign the random color to the corresponding segment
#         idx = segs[i] if num_label is None else i
#         if idx == -1:
#             random_color = np.zeros(3)  # Set color to black for label -1
#         rgb[seg == idx] = random_color
    
#     return rgb

def get_imglabel_color(label, num_label=None):
    """_summary_

    Args:
        label (np.array): [W,H]
        num_label (_type_): _description_
    """
    if num_label is None:
        num_label = label.max()+1
    W,H = label.shape
    label = label.reshape(-1)
    rgb = get_seg_color(label, num_label)
    rgb = rgb.reshape(W,H,-1)
    return rgb

def get_batchlabel_color(label, num_label=None):
    """_summary_

    Args:
        label (np.array): [W,H]
        num_label (_type_): _description_
    """
    if num_label is None:
        num_label = label.max()+1
    B,W,H = label.shape
    label = label.reshape(-1)
    rgb = get_seg_color(label, num_label)
    rgb = rgb.reshape(B,W,H,-1)
    return rgb

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text, is_print=True):
        if is_print:
            print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
        
def copy_files_to_output_dir(file_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
    for file_path in file_list:
        file = Path(file_path)  # 转换为 Path 对象
        if not file.exists() or not file.is_file():
            print(f"文件 {file} 不存在或不是有效文件，跳过。")
            continue
        
        target_path = Path(output_dir) / file.name
        shutil.copy(file, target_path)
        print(f"已复制: {file} -> {target_path}")
        
        
        
from sklearn.decomposition import PCA
import torch
def pca_reduce_numpy(features: torch.Tensor, out_dim: int) -> torch.Tensor:
    """
    将 GPU 上的 N×D 特征张量转换到 NumPy，使用 sklearn PCA 降维为 H 维度，
    再转换回 GPU 上的 PyTorch 张量。

    参数:
        features (torch.Tensor): 输入张量 (N, D)，必须在 GPU 上。
        out_dim (int): 目标维度 H。

    返回:
        torch.Tensor: (N, H) 降维结果，位于 GPU 上。
    """
    assert features.dim() == 2, "features 必须是二维张量 (N, D)"
    assert out_dim <= features.shape[1], "out_dim 必须小于等于输入特征维度"
    assert features.is_cuda, "features 必须在 GPU 上"

    # 转到 CPU 并转为 NumPy
    features_np = features.detach().cpu().numpy()

    # PCA 降维
    pca = PCA(n_components=out_dim)
    reduced_np = pca.fit_transform(features_np)  # (N, H)

    # 转回 GPU 上的 Tensor
    reduced_tensor = torch.from_numpy(reduced_np).to(features.device).type_as(features)

    return reduced_tensor
