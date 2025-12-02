import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.decomposition import PCA
from pytorch3d.ops import sample_farthest_points
from torch_sparse import SparseTensor
from pytorch3d.ops import ball_query

# from tmp1.tmp_graph_visualize import visualize_superpoint_graph_with_labels
from model.GNN import MultiLayerRelationalHybridGATv2,TransformerEncoder

from model.utils import JS2Weight, GeoAwarePooling,MaskQuality, AttentionFusion# , SimpleTransformerEncoder
from model.partgeoze import PartGeoZe
from dataset.PartnetEpc import get_is_seen
from model.ImageEncoder import ImageEncoder
from model.SuperPointAggre import SPAttentionAggregation, SPAttentionDownPropagation
from loss.contrast_loss import proxy_contrastive_loss, construct_positive_negative_samples,ContrastiveLoss,triplet_loss
import torch_scatter 

class SimpleTransformerEncoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4, ff_multiplier=4, num_layers=1):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, output_dim) if input_dim != output_dim else torch.nn.Identity()
        
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=output_dim * ff_multiplier,
            batch_first=True  # So input can be (B, N, D)
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        x: (B, N, D) tensor
        Returns: (B, N, D') tensor
        """
        x = self.input_proj(x)  # (B, N, D') if needed
        x = self.encoder(x)     # (B, N, D')
        return x
    
def value_to_color(tensor):
    
    red = tensor.unsqueeze(-1)
    blue = torch.zeros_like(red)
    green = 1 - tensor.unsqueeze(-1)
    
    # 将通道拼接成 BGR 图像 (W, H, 3)
    color = torch.cat([blue, green, red], dim=-1)
    return color

def get_bbox(img_mask):
    nonzero_indices = torch.nonzero(img_mask, as_tuple=True)
    
    if len(nonzero_indices[0]) == 0:
        return None
    
    y_coords = nonzero_indices[0]  
    x_coords = nonzero_indices[1]  
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    return torch.tensor([x_min, y_min, x_max, y_max])

def knn_points(
    query: torch.Tensor,
    key: torch.Tensor,
    k: int,
    sorted: bool = False,
    transpose: bool = False,
):
    """Compute k nearest neighbors.

    Args:
        query: [B, N1, D], query points. [B, D, N1] if @transpose is True.
        key:  [B, N2, D], key points. [B, D, N2] if @transpose is True.
        k: the number of nearest neighbors.
        sorted: whether to sort the results
        transpose: whether to transpose the last two dimensions.

    Returns:
        torch.Tensor: [B, N1, K], distances to the k nearest neighbors in the key.
        torch.Tensor: [B, N1, K], indices of the k nearest neighbors in the key.
    """
    if transpose:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
    # Compute pairwise distances, [B, N1, N2]
    distance = torch.cdist(query, key)
    if k == 1:
        knn_dist, knn_ind = torch.min(distance, dim=2, keepdim=True)
    else:
        knn_dist, knn_ind = torch.topk(distance, k, dim=2, largest=False, sorted=sorted)
    return knn_dist, knn_ind

def compute_class_weights(labels, num_classes, eps=1.02):
    hist = torch.bincount(labels.flatten(), minlength=num_classes)
    freq = hist.float() / hist.sum()
    weights = 1.0 / torch.log(freq + eps)
    return weights

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, input_dims, i=0):
    if i == -1:
        return torch.nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


class CN_layer(torch.nn.Module):
    def __init__(
        self,
        in_channel=256,
        out_channel=1,
        num_block=3,
        he_init=False,
    ):
        super().__init__()

        self.layer_in = torch.nn.Linear(in_channel, in_channel)
        self.relu = torch.nn.ReLU()
    
    def forward(self,feature):
        feature_in = self.layer_in(feature)
        feature_mean = torch.mean(feature_in, dim=0, keepdim=True) # [1, in_channel]
        feature_std = torch.std(feature_in, dim=0, keepdim=True, correction=0) #[1, in_channel]
        cn_feature = (feature_in - feature_mean)/feature_std
        cn_feature = self.relu(cn_feature)

        return cn_feature


class WeightPredNetworkCNe(torch.nn.Module):
    def __init__(
        self,
        in_channel=256,
        out_channel=1,
        num_cn_layer=1,
        he_init=False,
        skip_connection=True,
    ):
        super().__init__()
        self.skip_connection = skip_connection

        self.CN_layers = torch.nn.ModuleList([CN_layer(in_channel) for i in range(num_cn_layer)])
        
        self.layer_out = torch.nn.Linear(in_channel,out_channel)

        if he_init:
            self.apply(self._init_weights_he)
        else:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.0001)
            module.bias.data.zero_()

    def _init_weights_he(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight.data)
            
            if module.bias is not None:
                module.bias.data.zero_()


    def forward(self, feature):
        """
            feature : [N,in_channel]
        """

        feature_in = feature
        
        for layer in self.CN_layers:
            cn_feature = layer(feature_in)
            if self.skip_connection:
                feature_in = feature_in + cn_feature

        out = self.layer_out(feature_in)

        return out 

def check(mask_weight, mask_label, nview, pc_idx, shape_idx):
    tmp_weight = mask_weight.clone()
    mn = torch.min(tmp_weight)
    mx = torch.max(tmp_weight)
    tmp_weight = (tmp_weight-mn)/(mx-mn)
    print(f"mx = {mx}, mn = {mn}, mean = {tmp_weight.mean()}")
    iidd=0
    save_dir = f"./output/mask_weight2/check{shape_idx}"
    os.makedirs(save_dir, exist_ok=True)
    for view in range(nview):
        mask_label_tmp = torch.zeros_like(mask_label[0]).float()
        for i in range(mask_label.max()+1):
            img_ind = mask_label[view]==i
            if img_ind.sum()!=0:
                mask_label_tmp[img_ind] = tmp_weight[iidd].item()
                iidd += 1
        mask_label_tmp[pc_idx[view]==-1] = 0
        rgb = value_to_color(mask_label_tmp).detach().cpu().numpy()
        rgb = (rgb*255).astype(np.uint8)
        cv2.imwrite(f"{save_dir}/{view}.png", rgb)
    print("good")
                    
def pairwise_js_divergence(A, eps=1e-10):
    """
    A: torch.Tensor of shape (N, M), each row is a probability distribution
    Returns: torch.Tensor of shape (N, N), pairwise JS divergence matrix
    """
    A = A + eps  # 避免 log(0)
    A = A / A.sum(dim=1, keepdim=True)  # 归一化为概率分布

    logA = torch.log(A)

    # 扩展维度用于广播：A_i -> (N, 1, M), A_j -> (1, N, M)
    P = A.unsqueeze(1)  # (N, 1, M)
    Q = A.unsqueeze(0)  # (1, N, M)
    M = 0.5 * (P + Q)   # (N, N, M)

    # log(P), log(Q), log(M)
    logP = logA.unsqueeze(1)  # (N, 1, M)
    logQ = logA.unsqueeze(0)  # (1, N, M)
    logM = torch.log(M)

    # KL(P || M)
    kl_pm = torch.sum(P * (logP - logM), dim=2)
    kl_qm = torch.sum(Q * (logQ - logM), dim=2)

    # JS(P || Q)
    js = 0.5 * (kl_pm + kl_qm)  # shape (N, N)

    return js
    
def graph_visualize(mask_num, mask_label, id2mask, edges):
    label_new = torch.zeros_like(mask_label)
    min_lb = (torch.ones(10)*10000).long()
    max_lb = torch.zeros(10).long()
    for i in range(mask_num):
        view, label = id2mask[i]
        view, label = view.item(), label.item()
        ind = mask_label[view]==label
        label_new[view][ind]=i
        min_lb[view]=min(min_lb[view], i)
        max_lb[view]=max(max_lb[view], i)
    os.makedirs("./output/graph", exist_ok=True)
    for i in range(10):
        visualize_superpoint_graph_with_labels(label_new[i], edges, min_lb[i], max_lb[i], save_path=f"./output/graph/graph_visualize_{i}.png")

def get_ball_index(pc):
    pc = pc.unsqueeze(dim=0)
    idx = ball_query(pc, pc, K=20, radius=0.01)
    return idx[1].squeeze(dim=0)

class SegmentorNew(torch.nn.Module):
    def __init__(self, num_labels, args):
        super().__init__()
        
        self.num_labels = num_labels
        self.use_2d_feat = args.use_2d_feat      
        self.use_3d_feat = args.use_3d_feat
        
        self.args = args
        self.use_propagate = args.use_propagate
        self.ave_per_mask = args.ave_per_mask
        self.eliminate_sparseness = args.eliminate_sparseness
        self.use_slow_start = args.use_slow_start
        self.use_new_classifier = args.use_new_classifier
        self.use_js2weight = args.use_js2weight
        self.use_attn_ave = args.use_attn_ave
        self.use_gnn = args.use_gnn
        
        self.use_contrast_loss2 = args.use_contrast_loss2
        self.use_proxy_contrast_loss = args.use_proxy_contrast_loss
        self.use_mask_consist_loss = args.use_mask_consist_loss
        self.use_ref_loss = args.use_ref_loss
        
        self.cam_pos = np.load("view.npy")
         
        self.pc_feat_dim = 0
        if self.use_2d_feat:
            self.img_encoder = ImageEncoder(args.img_encoder, args.use_cache)
            self.pc_feat_dim += self.img_encoder.out_dim
        if self.use_3d_feat:
            # self.pc_encoder = Point_M2AE_ReductionD()
            self.pc_feat_dim += self.pc_encoder.out_dim
        if self.use_contrast_loss2:
            self.contrastive_loss = ContrastiveLoss(temperature=0.5)
        if self.use_attn_ave:
            self.sp_aggre = SPAttentionAggregation(self.pc_feat_dim)
            self.sp_down = SPAttentionDownPropagation(self.pc_feat_dim)
            self.self_attn = torch.nn.MultiheadAttention(embed_dim=self.pc_feat_dim, num_heads=1, batch_first=True)
        if self.use_gnn:
            edge_type_info = {}
            for edge_type in self.args.select_edges:
                edge_type_info[edge_type]={'has_attr': False}
            # if self.args.select_edges == "All":
            #     edge_type_info = {
            #         'strong': {'has_attr': False},
            #         'weak': {'has_attr': False}
            #     }
            # elif self.args.select_edges == "weak":
            #     edge_type_info = {
            #         'weak': {'has_attr': False}
            #     }
            # elif self.args.select_edges == "strong":
            #     edge_type_info = {
            #         'strong': {'has_attr': False}
            #     }
            print(edge_type_info)
            self.gnn = MultiLayerRelationalHybridGATv2(self.pc_feat_dim, 
                                                        self.pc_feat_dim*4, 
                                                        self.pc_feat_dim, 
                                                        edge_type_info=edge_type_info,
                                                        num_layers=3)
            

        if self.args.transformer:
            self.transformer = SimpleTransformerEncoder(self.pc_feat_dim, self.pc_feat_dim, num_heads=1, ff_multiplier=4, num_layers=1)
        if self.args.up_method == "GA_pooling":
            self.ga_pooling = GeoAwarePooling(128)
        
        self.calc_mask_w_unpooling= MaskQuality(128)
        # self.calc_mask_w_img_feat = MaskQuality(128)
        
        self.fuse = AttentionFusion(self.pc_feat_dim)
                

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        # self.classifier = torch.nn.Linear(self.pc_feat_dim, num_labels)
        self.classifier = torch.nn.Sequential(
                            torch.nn.Linear(self.pc_feat_dim, self.pc_feat_dim//2),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(0.2),
                            torch.nn.Linear(self.pc_feat_dim//2, num_labels),
                            # torch.nn.ReLU(),
                            # torch.nn.Dropout(0.1),
                            # torch.nn.Linear(self.pc_feat_dim//4, num_labels)
                            # torch.nn.ReLU(),
                            # torch.nn.Dropout(0.1),
                            # torch.nn.Linear(128, num_labels),
                            # torch.nn.ReLU(),
                            # # torch.nn.Dropout(0.1),
                            # torch.nn.Linear(64, num_labels)
                        )
    
    def aggregate(self, npoint, img_feat, pc_idx, coords, mask_label, epoch):
        nview=pc_idx.shape[0]
        device = img_feat.device
        dtype = img_feat.dtype
        
        nbatch = torch.repeat_interleave(torch.arange(0, nview)[:, None], npoint).view(-1, ).long()
        point_loc = coords.reshape(nview, -1, 2)
        xx, yy = point_loc[:, :, 0].long().reshape(-1), point_loc[:, :, 1].long().reshape(-1)
        point_feats = img_feat[nbatch, :, yy, xx].view(nview, npoint, -1)
        is_seen = get_is_seen(pc_idx, npoint).to(device)
        
        point_feats = torch.sum(point_feats * is_seen[:,:,None], dim=0)/(torch.sum(is_seen, dim=0)[:,None]+1e-6)
        
        return point_feats

    def compute_avg_features_from_grouped(self, features, grouped_indices):
        avg_list = []
        for indices in grouped_indices:
            if len(indices) > 0:
                avg_feat = features[indices].mean(dim=0)
            else:
                avg_feat = torch.zeros(features.shape[1], device=features.device, dtype=features.dtype)
            avg_list.append(avg_feat)

        avg_features = torch.stack(avg_list, dim=0)
        return avg_features

    
    def simple_down(self, pc_feat_ori, prop_feat, mask_num, id2mask, mask_pc_ind):
        """
        使用 scatter_add 实现特征聚合，支持梯度传递。
        """
        n_point = pc_feat_ori.shape[0]
        C = pc_feat_ori.shape[1]

        # 构造索引和特征拼接
        all_pc_inds = []
        all_feats = []

        for i in range(mask_num):
            pc_ind = mask_pc_ind[i]
            if pc_ind.numel() > 0:
                all_pc_inds.append(pc_ind)
                all_feats.append(prop_feat[i].expand(pc_ind.shape[0], -1))  # [num_points_in_mask, C]

        if len(all_pc_inds) == 0:
            return torch.zeros_like(pc_feat_ori)

        all_pc_inds = torch.cat(all_pc_inds, dim=0)  # [Total_points]
        all_feats = torch.cat(all_feats, dim=0)      # [Total_points, C]

        feat_sum = torch_scatter.scatter_add(all_feats, all_pc_inds, dim=0, dim_size=n_point)

        ones = torch.ones(all_pc_inds.shape[0], device=pc_feat_ori.device, dtype=pc_feat_ori.dtype)
        counts = torch_scatter.scatter_add(ones, all_pc_inds, dim=0, dim_size=n_point)  # shape [n_point]

        # 防止除以0
        counts = counts.unsqueeze(1).clamp(min=1e-6)

        pc_feat = feat_sum / counts  # [n_point, C]
        return pc_feat
    
    def GA_pooling(self, group_feat, mask_group_ind, mask_pc):
        mask_feat_list = []
        for i, ind in enumerate(mask_group_ind):
            if len(ind)>0:
                mask_feat = self.ga_pooling(group_feat[mask_group_ind[i]], mask_pc[i])
            else:
                mask_feat = torch.zeros(group_feat.shape[1], device=group_feat.device, dtype=group_feat.dtype)
            mask_feat_list.append(mask_feat)
        mask_feat_list = torch.stack(mask_feat_list, dim=0)
        return mask_feat_list
    
    def get_all_mask_w(self, maskAng, dtype, device, calc_mask_w):
        w_list = []
        for i, ma in enumerate(maskAng):
            if ma.numel()>0:
                w = calc_mask_w(ma)
            else:
                w = torch.zeros((1,1), device=ma.device, dtype=ma.dtype)
            w_list.append(w)  
        w_list = torch.cat(w_list, dim=0)
        return w_list
            
    def get_sim_edges(self, mask_feat, K):
        N = mask_feat.size(0)
        with torch.no_grad():
            dist = torch.cdist(mask_feat, mask_feat)
            knn_idx = dist.topk(k=K + 1, largest=False).indices[:, 1:]
            src = torch.arange(N, device=mask_feat.device).unsqueeze(1).expand(-1, K)
            tgt = knn_idx
            edge_index = torch.cat([
                torch.stack([src.flatten(), tgt.flatten()], dim=1),
                torch.stack([tgt.flatten(), src.flatten()], dim=1)
            ], dim=0)
            self_loops = torch.arange(N, device=mask_feat.device).unsqueeze(1).repeat(1, 2)
            edge_index = torch.cat([edge_index, self_loops], dim=0)
        return edge_index.permute(1, 0)
        
    def propagate_All_graph(self, img_feat, pc_feat, graph, mask_label,pc_id):
        device = pc_feat.device
        mask2id = graph["mask2id"]
        id2mask = graph["id2mask"]
        centers = graph["centers"].squeeze(dim=0).long().to(device)
        mask_pc_ind = [x.squeeze(dim=0).to(device) for x in graph["mask_pc_ind"]]
        mask_group_ind = [x.squeeze(dim=0).to(device) for x in graph["mask_group_ind"]]
        # nearest_index = graph["nearest_index"].to(device).squeeze(dim=0) 
        # grouped_indices = [x.squeeze(dim=0).to(device) for x in graph["grouped_indices"]] 
        
        n_point = pc_feat.shape[0]
        mask_num = max(mask2id.values())+1
        group_num = centers.shape[0]
        group_feat = pc_feat[centers]
        
        if self.args.up_method=="ave":
            mask_feat = self.compute_avg_features_from_grouped(group_feat, mask_group_ind)
        elif self.args.up_method=="GA_pooling":
            mask_pc = [x.squeeze(dim=0).float().to(device) for x in graph["mask_pc"]]
            mask_feat = self.GA_pooling(group_feat, mask_group_ind, mask_pc)
        
        if self.ave_per_mask:
            if self.args.down_method=="MQA_unpooling":
                mask_normAng = [x.squeeze(dim=0).float().to(device) for x in graph["mask_normAng"]]
                mask_w = self.get_all_mask_w(mask_normAng, mask_feat.dtype, mask_feat.device, self.calc_mask_w_unpooling)
                mask_feat = mask_feat * mask_w
            pc_feat_ave = self.simple_down(pc_feat, mask_feat, mask_num, id2mask, mask_pc_ind) 
            return pc_feat_ave
    
        if self.args.transformer:
            gat_feat = self.transformer(mask_feat)
        else:
            all_edge_index = {}
            if "weak" in self.args.select_edges:
                all_edge_index["weak"] = graph["weak_edge_index"].squeeze(dim=0).to(device)
            if "strong" in self.args.select_edges:
                all_edge_index["strong"] = graph["strong_edge_index"].squeeze(dim=0).to(device)        
            if "sim" in self.args.select_edges:
                sim_edges = self.get_sim_edges(mask_feat, 5)
                all_edge_index["sim"] = sim_edges
            # if self.args.select_edges == "All":
            #     all_edge_index["strong"] = graph["strong_edge_index"].squeeze(dim=0).to(device)
            #     all_edge_index["weak"] = graph["weak_edge_index"].squeeze(dim=0).to(device)
            # elif self.args.select_edges == "weak":
            #     all_edge_index["weak"] = graph["weak_edge_index"].squeeze(dim=0).to(device)
            # elif self.args.select_edges == "strong":
            #     all_edge_index["strong"] = graph["strong_edge_index"].squeeze(dim=0).to(device)
                
            edge_attr = {}
            # edge_attr["weak"] = graph["weak_edges_fpfh"].squeeze(dim=0).float().to(device)
        
            gat_feat = self.gnn(mask_feat, all_edge_index, edge_attr)
        
        if self.args.down_method=="MQA_unpooling":
            mask_normAng = [x.squeeze(dim=0).float().to(device) for x in graph["mask_normAng"]]
            mask_w = self.get_all_mask_w(mask_normAng, mask_feat.dtype, mask_feat.device, self.calc_mask_w_unpooling)
            gat_feat = gat_feat * mask_w
        ret_pc_feat = self.simple_down(pc_feat, gat_feat, mask_num, id2mask, mask_pc_ind)         

        if self.args.LH_method == "None":
            ret_pc_feat = ret_pc_feat
        elif self.args.LH_method == "ave":
            ret_pc_feat = pc_feat + ret_pc_feat
        elif self.args.LH_method == "fuse": 
            ret_pc_feat = self.fuse(pc_feat, ret_pc_feat)
        else:
            print("LH_method Error")
            exit(0)
        return ret_pc_feat
        
    def forward(self, pc_id, pc, pc_label,img, mask_label, pc_idx, coords, graph, pc_norm, args, epoch, mode="train",true_pc_id=0):
        device = pc.device
        n_point = pc.shape[0]
        img_feat, loss_ref = self.img_encoder(pc_id, img, graph["feat_path"][0])
        
        pc_feat = self.aggregate(n_point, img_feat, pc_idx, coords, mask_label, epoch)
        
        if self.use_propagate:
            pc_feat = self.propagate_All_graph(img_feat, pc_feat, graph, mask_label,true_pc_id)

        n_label = pc_label
        
        logits = self.classifier(pc_feat)
        logits_ce = logits
        label_ce = n_label
        
        loss = 0
        if pc_label is not None:
            
            if self.args.sample_pc:
                sample_ind = graph["sample_pc_ind"].squeeze(dim=0).long().to(device)
                logits_ce = logits[sample_ind]
                label_ce = n_label[sample_ind]
                
            if self.args.use_pseudo_label and mode=="train":
                psd_label = graph["psd_label"].squeeze(dim=0).long().to(device)
                valid_ind = psd_label!=-1
                logits_ce = logits[valid_ind]
                label_ce = psd_label[valid_ind]
            
            if not self.args.pretrain and mode!="self":
                weight = compute_class_weights(label_ce, self.num_labels)
                loss_ce = torch.nn.functional.cross_entropy(logits_ce, label_ce.long(), weight=weight, reduction="none")

                loss_ce = loss_ce.mean()
                loss = loss_ce
            
            if self.use_proxy_contrast_loss:
                loss_contrast = proxy_contrastive_loss(pc_feat, n_label, self.classifier)
                loss += loss_contrast
            if self.use_contrast_loss2:
                positive_pairs, negative_pairs = construct_positive_negative_samples(pc_feat.unsqueeze(dim=0), n_label.unsqueeze(dim=0), sample_num=1024*16)
                if positive_pairs.shape[0] > 0 and negative_pairs.shape[0] > 0:
                    loss_contrast = self.contrastive_loss(positive_pairs, negative_pairs)
                    loss += loss_contrast
                    # print("contrastive loss  :", loss_contrast)
            losses = []
            if self.args.use_triplet_loss:
                PN_tri = graph["PN_tri"].squeeze(dim=0).long().to(device)
                loss_tri = triplet_loss(pc_feat, PN_tri)
                loss+=loss_tri
            if self.use_mask_consist_loss:
                mask2id = graph["mask2id"]
                mask_pc_ind = [x.squeeze(dim=0).to(device) for x in graph["mask_pc_ind"]]
                mask_num = max(mask2id.values())+1
                loss_mask_consist = []
                ave_feat = []
                for i in range(mask_num):
                    pc_ind = mask_pc_ind[i]
                    if pc_ind.numel()>0:
                        tmp_ave_feat = pc_feat[pc_ind].mean(dim=0)
                        ave_feat.append(tmp_ave_feat)
                        loss_tmp = torch.norm(pc_feat[pc_ind]-tmp_ave_feat[None,:], dim=-1, p=2)
                        loss_mask_consist.append(loss_tmp.mean())
                    else:
                        ave_feat.append(torch.zeros_like(pc_feat[0]))
                        
                ave_feat = torch.stack(ave_feat, dim=0)
                loss_ins = sum(loss_mask_consist)/len(loss_mask_consist)
                loss += loss_ins
                
            if self.use_ref_loss:
                loss += loss_ref
        return logits, loss, n_label, pc_feat
    
    def freeze(self):
        for name, param in self.named_parameters():
            if not name.startswith("weight_pred"):
                param.requires_grad = False

        