# -*- coding:UTF-8 -*-

# author:Feixiang Wang
# software: PyCharm

"""
文件说明：
    
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATv2Conv, TransformerConv,GINEConv
from torch_scatter import scatter_sum
import time


# PNAConv(2020、Dynamic Graph CNN for Learning on Point Clouds)
# EdgeConv(2018)
# TransformerConv(2020)
class GCN_2D(nn.Module):
    """RNA 2D图神经网络"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(GCN_2D, self).__init__()
        # self.x_proj = nn.Linear(input_dim, hidden_dim)  # 节点特征投影
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index, batch):
        # x: 节点特征矩阵 [num_nodes, input_dim]
        # edge_index: 边索引 [2, num_edges]
        # batch: 批索引 [num_nodes]
        # x = self.x_proj(x)
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = global_mean_pool(x, batch)  # 全局平均池化 [batch_size, hidden_dim]
        x = self.fc(x)
        return x


class GATv2_2D(nn.Module):
    """药物2D图神经网络（处理边特征）"""
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim, dropout=0.5):
        super(GATv2_2D, self).__init__()
        # 使用GATv2Conv替换GCNConv，支持多维边特征
        self.conv1 = GATv2Conv(input_dim, hidden_dim, edge_dim=edge_dim)  # 边特征维度
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index, edge_attr, batch):
        # 图卷积，直接使用多维边特征
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.activation(x)
        x = global_mean_pool(x, batch)  # 全局平均池化
        x = self.fc(x)
        return x

class Trans_2D(nn.Module):
    """药物2D图神经网络（处理边特征）"""
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim, dropout=0.5):
        super(Trans_2D, self).__init__()
        # 使用GATv2Conv替换GCNConv，支持多维边特征
        self.conv1 = TransformerConv(input_dim, hidden_dim, edge_dim=edge_dim)  # 边特征维度
        self.conv2 = TransformerConv(hidden_dim, hidden_dim, edge_dim=edge_dim)
        # self.conv3 = TransformerConv(hidden_dim, hidden_dim, edge_dim=edge_dim, heads=4, concat=False)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index, edge_attr, batch):
        # 图卷积，直接使用多维边特征
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.activation(x)
        x = global_mean_pool(x, batch)  # 全局平均池化
        x = self.fc(x)
        return x


class GINE_2D(nn.Module):
    """使用GINEConv的2D图神经网络（高效处理边特征）"""

    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim, dropout=0.5):
        super(GINE_2D, self).__init__()
        # GINEConv需要MLP作为节点更新函数
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.conv1 = GINEConv(nn=self.mlp1, edge_dim=edge_dim)
        self.conv2 = GINEConv(nn=self.mlp2, edge_dim=edge_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index, edge_attr, batch):
        # 图卷积，直接使用多维边特征
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.activation(x)
        x = global_mean_pool(x, batch)  # 全局平均池化
        x = self.fc(x)
        return x

# class DrugGCN(nn.Module):
#     """药物2D图神经网络（处理边特征）"""
#     def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
#         super(DrugGCN, self).__init__()
#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         # self.edge_proj = nn.Linear(edge_attr_dim, hidden_dim)  # 边特征投影
#         self.fc = nn.Linear(hidden_dim, output_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = nn.ReLU()
#
#     def forward(self, x, edge_index, edge_attr, batch):
#         # 边特征处理
#         # edge_attr = self.edge_proj(edge_attr)
#
#         # 图卷积
#         x = self.conv1(x, edge_index, edge_weight=edge_attr)
#         x = self.activation(x)
#         x = self.dropout(x)
#         x = self.conv2(x, edge_index, edge_weight=edge_attr)
#         x = self.activation(x)
#         x = global_mean_pool(x, batch)  # 全局平均池化
#         x = self.fc(x)
#         return x

class Conv(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(Conv, self).__init__()
        self.pre_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU()
        )
        self.preffn_dropout = nn.Dropout(dropout_rate)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_attr, bases):
        """
        x: 节点特征 [num_nodes, hidden_size]
        edge_index: 边索引 [2, num_edges]
        edge_attr: 边特征 [num_edges, hidden_size]
        bases: 基函数权重 [num_edges, hidden_size]
        """
        # 获取源节点特征
        src_idx = edge_index[0]  # 源节点索引
        src_x = x[src_idx]  # [num_edges, hidden_size]

        # 计算边特征: (源节点特征 + 边特征) -> pre_ffn -> * bases
        edge_feat = src_x + edge_attr
        edge_feat = self.pre_ffn(edge_feat)
        edge_feat = edge_feat * bases

        # 聚合边特征到目标节点
        dst_idx = edge_index[1]  # 目标节点索引
        aggr = scatter_sum(edge_feat, dst_idx, dim=0, dim_size=x.size(0))
        aggr = self.preffn_dropout(aggr)

        # 残差连接
        y = x + aggr

        # FFN处理
        y_ffn = self.ffn(y)
        y_ffn = self.ffn_dropout(y_ffn)

        return y + y_ffn


class PDF(torch.nn.Module):
    def __init__(self,
                 input_dim: int,  # 节点特征维度
                 edge_attr_dim: int,  # 边特征维度
                 hidden_dim: int,  # 隐藏层维度
                 num_basis: int,  # 基函数数量
                 num_layers: int = 3):  # 卷积层数
        super(PDF, self).__init__()

        self.num_layers = num_layers

        # 基函数处理
        self.filter_encoder = nn.Sequential(
            nn.Linear(num_basis, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.filter_drop = nn.Dropout(0.5)

        # 节点特征投影
        self.node_proj = nn.Linear(input_dim, hidden_dim)

        # 边特征投影
        self.edge_proj = nn.Linear(edge_attr_dim, hidden_dim)

        # 创建多层卷积
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(Conv(hidden_size=hidden_dim, dropout_rate=0.5))

        # 池化层
        self.pool = global_mean_pool

    def forward(self, x, edge_index, edge_attr, bases, batch):
        """
        x: 节点特征 [num_nodes, input_dim]
        edge_index: 边索引 [2, num_edges]
        edge_attr: 边特征 [num_edges, edge_attr_dim]
        bases: 基函数 [num_edges, num_basis]
        batch: 批索引 [num_nodes]
        """
        # 处理基函数
        bases = self.filter_encoder(bases)
        bases = self.filter_drop(bases)
        bases = F.softmax(bases, dim=0)  # 沿边维度softmax

        # 投影节点和边特征
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)

        # 多层卷积
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr, bases)
            x = F.relu(x)

        # 全局平均池化
        graph_emb = self.pool(x, batch)
        return graph_emb

def main_Trans_2D():
    # 超参数设置
    input_dim = 10
    hidden_dim = 16
    output_dim = 2
    edge_dim = 5

    # 创建模拟数据
    # 6个节点，每个节点特征维度10
    x = torch.randn(6, input_dim)

    # 边索引（包含4条边）
    edge_index = torch.tensor([[0, 1, 2, 3],
                               [1, 0, 3, 2]], dtype=torch.long)

    # 边特征（4条边，每条边特征维度5）
    edge_attr = torch.randn(4, edge_dim)

    # 批索引（3个图，每个图2个节点）
    batch = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)

    # 实例化模型
    model = Trans_2D(input_dim, hidden_dim, output_dim, edge_dim)

    # 前向传播
    output = model(x, edge_index, edge_attr, batch)

    # 打印形状信息
    print(f"输入节点特征x shape: {x.shape}")  # torch.Size([6, 10])
    print(f"边索引edge_index shape: {edge_index.shape}")  # torch.Size([2, 4])
    print(f"边特征edge_attr shape: {edge_attr.shape}")  # torch.Size([4, 5])
    print(f"批索引batch shape: {batch.shape}")  # torch.Size([6])

    # 中间层输出
    # with torch.no_grad():
    #     # 测试第一层卷积输出
    #     x1 = model.activation(model.conv1(x, edge_index, edge_attr))
    #     print(f"第一层卷积后x shape: {x1.shape}")  # torch.Size([6, 16])
    #
    #     # 全局池化后
    #     pooled = global_mean_pool(x1, batch)
    #     print(f"全局池化后 shape: {pooled.shape}")  # torch.Size([3, 16])

    print(f"最终输出output shape: {output.shape}")  # torch.Size([3, 2])

def main_PDF():
    # 超参数设置
    input_dim = 10      # 节点特征维度
    hidden_dim = 16     # 隐藏层维度
    edge_attr_dim = 5   # 边特征维度
    num_basis = 8       # 基函数数量
    num_layers = 3      # 卷积层数

    # 创建模拟数据
    # 6个节点，每个节点特征维度10
    x = torch.randn(6, input_dim)

    # 边索引（包含4条边）
    edge_index = torch.tensor([[0, 1, 2, 3],
                               [1, 0, 3, 2]], dtype=torch.long)

    # 边特征（4条边，每条边特征维度5）
    edge_attr = torch.randn(4, edge_attr_dim)

    # 基函数（4条边，每个边有8个基函数权重）
    bases = torch.randn(4, num_basis)

    # 批索引（3个图，每个图2个节点）
    batch = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)

    # 实例化模型
    model = PDF(input_dim=input_dim,
                edge_attr_dim=edge_attr_dim,
                hidden_dim=hidden_dim,
                num_basis=num_basis,
                num_layers=num_layers)

    # 前向传播
    output = model(x, edge_index, edge_attr, bases, batch)

    # 打印形状信息
    print(f"输入节点特征x shape: {x.shape}")          # torch.Size([6, 10])
    print(f"边索引edge_index shape: {edge_index.shape}")   # torch.Size([2, 4])
    print(f"边特征edge_attr shape: {edge_attr.shape}")     # torch.Size([4, 5])
    print(f"基函数bases shape: {bases.shape}")            # torch.Size([4, 8])
    print(f"批索引batch shape: {batch.shape}")        # torch.Size([6])

    # 中间层输出验证
    # with torch.no_grad():
    #     # 获取卷积中间输出
    #     conv_features = x
    #     for i, conv in enumerate(model.convs):
    #         conv_features = conv(conv_features, edge_index, edge_attr, bases)
    #         print(f"第{i+1}层卷积后特征 shape: {conv_features.shape}")  # torch.Size([6, 16])

    print(f"最终输出output shape: {output.shape}")     # torch.Size([3, 16])

def benchmark_functions():
    # 运行50次测试
    num_runs = 50

    # 测试 main_Trans_2D
    start_time = time.time()
    for _ in range(num_runs):
        main_Trans_2D()
    trans_2d_total_time = time.time() - start_time
    trans_2d_avg_time = trans_2d_total_time / num_runs

    # 测试 main_PDF
    start_time = time.time()
    for _ in range(num_runs):
        main_PDF()
    pdf_total_time = time.time() - start_time
    pdf_avg_time = pdf_total_time / num_runs

    print(f"运行 {num_runs} 次的结果:")
    print(f"main_Trans_2D 总时间: {trans_2d_total_time:.6f} 秒")
    print(f"main_Trans_2D 平均时间: {trans_2d_avg_time:.6f} 秒")
    print(f"main_PDF 总时间: {pdf_total_time:.6f} 秒")
    print(f"main_PDF 平均时间: {pdf_avg_time:.6f} 秒")

    if trans_2d_avg_time < pdf_avg_time:
        print("main_Trans_2D 运行更快")
    else:
        print("main_PDF 运行更快")

# 如果你想运行这个基准测试，可以取消下面的注释
# benchmark_functions()



if __name__ == "__main__":
    # main_Trans_2D()
    main_PDF()
    # benchmark_functions()