import torch
import torch.nn.functional as F
import numpy as np
from dataset.PartnetEpc import get_is_seen
import cv2

def value_to_color(tensor):
    """
    将 0-1 范围的 tensor 值映射为颜色，0 对应蓝色 (0, 0, 1)，1 对应红色 (1, 0, 0)
    :param tensor: W*H 的张量，值域为 [0, 1]
    :return: W*H*3 的颜色张量，RGB 格式
    """
    mn = tensor.min()
    mx = tensor.max()
    tensor = (tensor-mn)/(mx-mn)
    red = tensor.unsqueeze(-1)
    blue = torch.zeros_like(red)
    green = 1 - tensor.unsqueeze(-1)
    
    # 将通道拼接成 BGR 图像 (W, H, 3)
    color = torch.cat([blue, green, red], dim=-1)
    return color


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=1):
        super(ConvLayer, self).__init__() 
        if kernel_size>1:
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size//2, padding=kernel_size//2)
        elif kernel_size==1:
            self.conv = torch.nn.Conv2d(in_channels, out_channels, (1,1))
        
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class ConvCNLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=1):
        super(ConvCNLayer, self).__init__() 
        if kernel_size>1:
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size//2, padding=kernel_size//2)
        elif kernel_size==1:
            self.conv = torch.nn.Conv2d(in_channels, out_channels, (1,1))
        self.relu = torch.nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x_mean = torch.mean(x, dim=(2,3), keepdim=True)
        x_std = torch.std(x, dim=(2,3), keepdim=True, correction=0)
        cn_x = (x - x_mean)/x_std
        ret_x = self.relu(cn_x)
        return ret_x

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

class Linear1(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Linear1, self).__init__()

        self.in_channels = in_channels
        
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, (1,1))
        self.relu = torch.nn.ReLU()
        # self.relu = torch.nn.Identity()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = torch.nn.functional.interpolate(x, size=(800,800), mode="bilinear", align_corners=False)
        
        # upsampled = []
        # for i in range(10):
        #     sample = x[i].unsqueeze(0)
        #     sample_upsampled = F.interpolate(sample, size=(800, 800), mode="bilinear", align_corners=False)
        #     upsampled.append(sample_upsampled.squeeze(0)) 
        # x = torch.stack(upsampled, dim=0) 
        return x

def aggregate(npoint, img_feat, pc_idx, coords, use_attn_map):
    nview=pc_idx.shape[0]
    device = img_feat.device
    
    nbatch = torch.repeat_interleave(torch.arange(0, nview)[:, None], npoint).view(-1, ).long()
    point_loc = coords.reshape(nview, -1, 2)
    xx, yy = point_loc[:, :, 0].long().reshape(-1), point_loc[:, :, 1].long().reshape(-1)
    point_feats = img_feat[nbatch, :, yy, xx].view(nview, npoint, -1)
    is_seen = get_is_seen(pc_idx, npoint).to(device)
    point_feats = torch.sum(point_feats * is_seen[:,:,None], dim=0)/torch.sum(is_seen, dim=0)[:,None]
    # point_feats = torch.mean(point_feats, dim=0) 
    return point_feats
        
class AttentionMap0(torch.nn.Module):
    def __init__(self, kernel_size=7):
        super(AttentionMap0, self).__init__()
        assert kernel_size in (3, 5, 7), "Kernel size must be 3, 5, or 7"
        padding = kernel_size // 2
        self.conv = torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # [B, 1, W, H]
        mean_pool = torch.mean(x, dim=1, keepdim=True)  # [B, 1, W, H]

        attention = torch.cat([max_pool, mean_pool], dim=1)  # [B, 2, W, H]

        attention = self.conv(attention)  # [B, 1, W, H]

        attention = torch.sigmoid(attention)  # [B, 1, W, H]

        return attention

class AttentionMap(torch.nn.Module):
    def __init__(self, channels, hidden_channels, num_conv_layers):
        super(AttentionMap, self).__init__()
        self.layer_in = ConvLayer(channels, hidden_channels)
        self.conv_layers = torch.nn.ModuleList([ConvLayer(hidden_channels, hidden_channels) for i in range(num_conv_layers)])
        self.layer_out = torch.nn.Conv2d(hidden_channels, 1, (1,1))
    
    def forward(self, x):
        x_in = self.layer_in(x)
        for layer in self.conv_layers:
            x_conv = layer(x_in)
            x_in = x_in + x_conv
        out = self.layer_out(x_in)
        return out

class FrontAttentionMap(torch.nn.Module):
    def __init__(self,channels, pos_channels,num_conv_layers=1):
        super(FrontAttentionMap, self).__init__()
        self.pos_conv = torch.nn.Conv2d(pos_channels,pos_channels,kernel_size=(14,14), stride=14)
        self.relu = torch.nn.ReLU()
        hid_channels = channels+pos_channels
        self.conv_CN_layers = torch.nn.ModuleList([ConvCNLayer(hid_channels, hid_channels, kernel_size=3) for i in range(num_conv_layers)])
        self.layer_out = torch.nn.Conv2d(hid_channels, 1, (1,1))
    def forward(self, patch_embeddings, pos_embed):
        pos_feature = self.relu(self.pos_conv(pos_embed))
        
        feature_in = torch.cat([patch_embeddings, pos_feature], dim=1)
        for layer in self.conv_CN_layers:
            cn_feature = layer(feature_in)
            feature_in = feature_in + cn_feature
        out = self.layer_out(feature_in)
        weight = F.relu(10+out)
        return weight


class Segmentor(torch.nn.Module):
    def __init__(self, hidden_size, num_labels, args, tokenW=57, tokenH=57):
        super(Segmentor, self).__init__()
        self.cam_pos = np.load("view.npy")
        self.use_attn_map = args.use_attn_map   
        self.use_front_attn_map = args.use_front_attn_map
        self.ave_per_mask = args.ave_per_mask
        
        self.hidden_size=hidden_size
        self.width = tokenW
        self.height = tokenH
        
        self.linear1 = Linear1(hidden_size, hidden_size//8)
        if self.use_front_attn_map:
            embed_fn, embed_dim = get_embedder(multires=10, input_dims=6)
            self.embed_fn = embed_fn
            self.embed_dim = embed_dim
            self.attn = FrontAttentionMap(hidden_size, self.embed_dim, num_conv_layers=5)
        if self.use_attn_map:
            embed_fn, embed_dim = get_embedder(multires=10, input_dims=6)
            self.embed_fn = embed_fn
            self.embed_dim = embed_dim
            self.attn = AttentionMap(self.embed_dim+hidden_size//8, hidden_size//(8*8) , 2)
        self.classifier = torch.nn.Linear(hidden_size//8, num_labels)
        # self.attn = AttentionMap0()
    
    def get_pos_embed(self, pc, pc_idx):
        n_view, W, H = pc_idx.shape
        pc_on_img = pc[pc_idx]
        cam_pos = torch.from_numpy(self.cam_pos).to(pc.device).unsqueeze(1).unsqueeze(2)
        pos = torch.cat([pc_on_img, cam_pos.expand(-1, W, H, -1)], dim=-1)  # 直接广播
        pos_embed = self.embed_fn(pos.view(-1, pos.shape[-1])).float().view(n_view, W, H, -1)

        # 使用布尔掩码直接设置0
        pos_embed = pos_embed.permute(0, 3, 1, 2)
        pos_embed.masked_fill_(pc_idx.unsqueeze(1) == -1, 0)
        return pos_embed
    
    def forward(self, pc, patch_embeddings, n_pc, pc_idx, coords, mask_label, epoch):
        patch_embeddings = patch_embeddings.reshape(-1, self.height, self.width, self.hidden_size)
        patch_embeddings = patch_embeddings.permute(0,3,1,2)
        img_feat = self.linear1(patch_embeddings)

        if self.use_front_attn_map and (epoch==-1 or epoch > 4):
            pos_embed = self.get_pos_embed(pc,pc_idx)
            attn_map = self.attn(patch_embeddings, pos_embed)
            attn_map = torch.nn.functional.interpolate(attn_map, size=(800,800), mode="bilinear", align_corners=False)
            for view in range(10):
                rgb = value_to_color(attn_map[view].clone())
                rgb = rgb.squeeze(dim=0)
                rgb[pc_idx[view]==-1]=0
                rgb=rgb.detach().cpu().numpy()
                rgb  = (rgb  * 255).astype(np.uint8) 
                cv2.imwrite(f"output/check/output_{view}_pattn.png", rgb)
            img_feat *= attn_map
        elif self.use_attn_map and (epoch==-1 or epoch > 4):
            pos_embed = self.get_pos_embed(pc,pc_idx)
            # 特征拼接并计算注意力图
            img_feat_pos = torch.cat([img_feat, pos_embed], dim=1)
            attn_map = self.attn(img_feat_pos)
            img_feat *= attn_map
        else:
            img_feat *=10

            
            
        pc_feat = aggregate(n_pc, img_feat, pc_idx, coords, self.use_attn_map)
        
        
        if self.ave_per_mask:
            num_view = pc_idx.shape[0]
            n_point = pc.shape[0]
            ave_cnt = torch.ones(n_point).float().to(pc_feat.device)
            for view in range(num_view):
                for i in range(mask_label.max()+1):
                    img_ind = mask_label[view]==i
                    pc_ind = pc_idx[view,img_ind]
                    pc_ind = pc_ind[pc_ind!=-1]
                    if pc_ind.numel() > 0: 
                        ave_feat = pc_feat[pc_ind].mean(dim=0)
                        # pc_feat[pc_ind] = (pc_feat[pc_ind]+ave_feat)/2   
                        pc_feat[pc_ind] = pc_feat[pc_ind]+ave_feat
                        ave_cnt[pc_ind] += 1   
            pc_feat /= ave_cnt[:,None]
            pass
        # img_feat = img_feat.permute(0,2,3,1)
        # NN = (pc_idx!=-1).sum()
        # pc_feat = img_feat[pc_idx!=-1].reshape(NN,-1)
        # ind = pc_idx[pc_idx!=-1].reshape(NN)
        # pc_label = pc_label[ind]
        
        logits = self.classifier(pc_feat)
        return logits