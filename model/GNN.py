import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv,TransformerConv,NNConv
from torch_geometric.data import Data
import torch_geometric.nn
import torch_geometric.utils

class SimpleGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SimpleGNN, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GATv2Conv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        # x = F.relu(self.conv2(x, edge_index))
        # x = F.relu(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)
        return x
    
class GATNet(torch.nn.Module):
    def __init__(self,F_in, F_hidden, F_out, heads=8):
        super().__init__()
        self.gat1 = GATv2Conv(F_in, F_hidden, heads=heads, dropout=0.1)
        self.norm1 = torch.nn.LayerNorm(F_hidden * heads)
        self.gat2 = GATv2Conv(F_hidden * heads, F_out, heads=1, concat=False, dropout=0.1)
        self.norm2 = torch.nn.LayerNorm(F_out)

    def forward(self, x, edge_index, edge_index_weak=None):
        x = self.gat1(x, edge_index)  # [N, F_hidden * heads]
        x = self.norm1(x)   
        x = F.elu(x)
        x = self.gat2(x, edge_index)  # [N, F_out]
        x = self.norm2(x)       
        return x
    
class DualEdgeGATv2(torch.nn.Module):
    def __init__(self, F_in, F_hidden, F_out, heads=8, merge='concat'):
        super().__init__()
        self.gat_strong = GATNet(F_in, F_hidden, F_out)
        self.gat_weak = GATNet(F_in, F_hidden, F_out)
        self.merge = merge
        # self.output_dim = F_out * 2 if merge == 'concat' else F_out


    def forward(self, x, edge_index_strong, edge_index_weak):
        h_strong = self.gat_strong(x, edge_index_strong)  # e.g., high-confidence edges
        h_weak = self.gat_weak(h_strong, edge_index_weak)        # e.g., low-confidence edges
        return h_weak
    
class GatLayer(torch.nn.Module):
    def __init__(self,F_in, F_out, heads=8, concate=None):
        super().__init__()
        self.gat1 = GATConv(F_in, F_out, heads=heads, dropout=0.1)
        self.norm1 = torch.nn.LayerNorm(F_out*heads)
    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)  # [N, F_hidden * heads]
        x = self.norm1(x) 
        return x
class AllGraph(torch.nn.Module):
    def __init__(self, F_in, F_hidden, F_out, heads=8):
        super().__init__()
        self.up1 = GatLayer(F_in, F_hidden, heads, concate=True)
        self.prop1 = GatLayer(F_in, F_hidden, heads, concate=True)
        self.down1 = GatLayer(F_in, F_hidden, heads, concate=True)
        self.up2 = GatLayer(F_hidden*heads, F_out, heads=1, concate=False)
        self.prop2 = GatLayer(F_hidden*heads, F_out, heads=1, concate=False)
        self.down2 = GatLayer(F_hidden*heads, F_out, heads=1, concate=False)
    def forward(self, x, edges_index, group_num, edge_index_maskNode):
        x_up1 = self.up1(x, edge_index)
        x_prop1 = self.prop1(x_up1[group_num:], edge_index_maskNode)
        x_aggre1 = torch.cat([x_up1[:group_num], x_prop1], dim=0)
        x_gat1 = self.down1(x_aggre1, edge_index)



class MultiRelationalGATv2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_types, heads=8, concat=True):
        super().__init__()
        self.heads = heads
        self.concat = concat

        if concat:
            assert out_channels % heads == 0, "out_channels must be divisible by heads when concat=True"
            out_per_head = out_channels // heads
        else:
            out_per_head = out_channels  # 若不拼接，则直接设定为目标维度

        self.rel_convs = torch.nn.ModuleDict({
            str(r): GATv2Conv(
                in_channels=in_channels,
                out_channels=out_per_head,
                heads=heads,
                concat=concat
            )
            for r in edge_types
        })

    def forward(self, x, edge_index_dict):
        out = []
        for r, edge_index in edge_index_dict.items():
            h = self.rel_convs[r](x, edge_index)  # 输出维度受 concat 决定
            out.append(h)
        return torch.sum(torch.stack(out), dim=0)  # 可选：改为 concat + MLP

class MultiLayerRelationalGATv2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_types, num_layers=3):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.res_conns = torch.nn.ModuleList()

        # Input layer
        self.layers.append(MultiRelationalGATv2(in_channels, hidden_channels, edge_types))
        self.norms.append(torch.nn.LayerNorm(hidden_channels))
        self.res_conns.append(torch.nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else torch.nn.Identity())

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(MultiRelationalGATv2(hidden_channels, hidden_channels, edge_types))
            self.norms.append(torch.nn.LayerNorm(hidden_channels))
            self.res_conns.append(torch.nn.Identity())  # same dim, can directly add

        # Output layer
        self.layers.append(MultiRelationalGATv2(hidden_channels, out_channels, edge_types))
        self.norms.append(torch.nn.LayerNorm(out_channels))
        self.res_conns.append(torch.nn.Linear(hidden_channels, out_channels) if hidden_channels != out_channels else torch.nn.Identity())

    def forward(self, x, edge_index_dict):
        for layer, norm, res in zip(self.layers, self.norms, self.res_conns):
            x_res = res(x)  # transform input for residual if needed
            x = layer(x, edge_index_dict)
            x = norm(x + x_res)  # residual + norm
            x = F.relu(x)        # non-linearity
        return x

class MultiRelationalHybridGATv2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_type_info, heads=8, concat=True):
        super().__init__()
        self.heads = heads
        self.concat = concat
        self.edge_type_info = edge_type_info

        if concat:
            assert out_channels % heads == 0, "out_channels must be divisible by heads when concat=True"
            out_per_head = out_channels // heads
        else:
            out_per_head = out_channels

        self.rel_convs = torch.nn.ModuleDict()
        self.edge_mlps = torch.nn.ModuleDict()
        self.rel_norms = torch.nn.ModuleDict()

        for r, edge_info in edge_type_info.items():
            if edge_info["has_attr"]:
                edge_dim = edge_info["edge_dim"]
                mlp_hidden = 128

                self.edge_mlps[r] = torch.nn.Sequential(
                    torch.nn.Linear(edge_dim, mlp_hidden//2),
                    # torch.nn.BatchNorm1d(mlp_hidden//2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(mlp_hidden//2, mlp_hidden),
                    # torch.nn.BatchNorm1d(mlp_hidden),
                    torch.nn.ReLU()
                )

                if edge_info["edges_model"] == "NNConv":
                    # For NNConv, edge_attr must be processed to get weights
                    nn_linear = torch.nn.Sequential(
                        torch.nn.Linear(mlp_hidden, in_channels * out_per_head * heads)
                    )
                    self.rel_convs[r] = NNConv(
                        in_channels=in_channels,
                        out_channels=out_per_head * heads if concat else out_per_head,
                        nn=nn_linear,
                        aggr='mean'
                    )
                elif edge_info["edges_model"] == "TransformerConv":
                    self.rel_convs[r] = TransformerConv(
                        in_channels=in_channels,
                        out_channels=out_per_head,
                        heads=heads,
                        concat=concat,
                        edge_dim=mlp_hidden
                    )
                else:
                    self.rel_convs[r] = GATv2Conv(
                        in_channels=in_channels,
                        out_channels=out_per_head,
                        heads=heads,
                        concat=concat,
                        edge_dim=mlp_hidden
                    )
            else:
                self.rel_convs[r] = GATv2Conv(
                    in_channels=in_channels,
                    out_channels=out_per_head,
                    heads=heads,
                    concat=concat,
                    # dropout=0.5
                )

            # Normalization per relation
            self.rel_norms[r] = torch.nn.LayerNorm(out_per_head * heads if concat else out_per_head)

        # Fusion MLP after aggregating all relation-specific outputs
        self.fusion_mlp = torch.nn.Sequential(
            torch.nn.Linear(len(edge_type_info) * (out_per_head * heads if concat else out_per_head), out_channels),
            torch.nn.ReLU()
        )

    def forward(self, x, edge_index_dict, edge_attr_dict=None):
        outputs = []
        for r, edge_index in edge_index_dict.items():
            conv = self.rel_convs[r]

            # if isinstance(conv, (TransformerConv, NNConv, GATv2EdgeConv)):
            #     edge_attr = edge_attr_dict.get(r, None)
            #     assert edge_attr is not None
            #     edge_attr = self.edge_mlps[r](edge_attr)
            #     h = conv(x, edge_index, edge_attr=edge_attr)
            # else:
            #     h = conv(x, edge_index)
            
            edge_attr = edge_attr_dict.get(r, None)
            if edge_attr is not None:
                edge_attr = self.edge_mlps[r](edge_attr)
                # print("eeeeeeeeeeeeeeedge")
            h = conv(x, edge_index, edge_attr=edge_attr)

            # h = self.rel_norms[r](h)
            outputs.append(h)

        # Concatenate all relation outputs along feature dimension
        out = torch.cat(outputs, dim=-1)
        out = self.fusion_mlp(out)
        return out
    
class MultiLayerRelationalHybridGATv2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_type_info, num_layers=3):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.res_conns = torch.nn.ModuleList()

        # Input layer
        self.layers.append(MultiRelationalHybridGATv2(in_channels, hidden_channels, edge_type_info))
        self.norms.append(torch.nn.LayerNorm(hidden_channels))
        self.res_conns.append(torch.nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else torch.nn.Identity())

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(MultiRelationalHybridGATv2(hidden_channels, hidden_channels, edge_type_info))
            self.norms.append(torch.nn.LayerNorm(hidden_channels))
            self.res_conns.append(torch.nn.Identity())

        # Output layer
        self.layers.append(MultiRelationalHybridGATv2(hidden_channels, out_channels, edge_type_info))
        self.norms.append(torch.nn.LayerNorm(out_channels))
        self.res_conns.append(torch.nn.Linear(hidden_channels, out_channels) if hidden_channels != out_channels else torch.nn.Identity())

    def forward(self, x, edge_index_dict, edge_attr_dict={}):
        for layer, norm, res in zip(self.layers, self.norms, self.res_conns):
            x_res = res(x)
            x = layer(x, edge_index_dict, edge_attr_dict=edge_attr_dict)
            x = norm(x + x_res)
            x = F.relu(x)
        return x

    
if __name__ == "__main__":
    # 示例数据
    N, D = 10, 5  # 10个点，每个点5维特征
    x = torch.randn((N, D))  # 随机初始化点特征
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # 源点
                                [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]])  # 目标点

    data = Data(x=x, edge_index=edge_index)

    # 定义模型并前向传播
    # model = SimpleGNN(in_channels=D, hidden_channels=16, out_channels=8)
    model = GATNet(F_in=D, F_hidden=16, F_out=8)
    out = model(data.x, data.edge_index)
    print(out.shape)  # 输出的形状应为 (N, 8)


class TransformerEncoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers, dropout=0.2):
        super(TransformerEncoder, self).__init__()
        self.embedding = torch.nn.Linear(input_dim, output_dim)  
        self.transformer_encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=output_dim,  
            nhead=num_heads,  
            dropout=dropout  
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers 
        )
        self.output_projection = torch.nn.Linear(output_dim, output_dim)  # 投影到输出空间

    def forward(self, x):
        # 假设x的形状是(N, D)
        x = self.embedding(x)  # 变换到输出维度
        x = x.unsqueeze(1)  # 添加时间步维度，形状变为(N, 1, D')
        x = self.transformer_encoder(x)  # 通过Transformer编码
        x = self.output_projection(x.squeeze(1))  # 移除时间步维度并输出
        return x