# -*- coding:UTF-8 -*-

# author:Feixiang Wang
# software: PyCharm

"""
文件说明：
    
"""
from .pointmodel import PointNetEncoder
from .egnn import EGNN_Sparse
from .fusion import Fusion
import torch
import torch.nn as nn

hidden_dim = 512

class MyModel(nn.Module):
    def __init__(self,device):
        super(MyModel, self).__init__()
        self.rna_point_model = PointNetEncoder(global_feat=True, feature_transform=True, channel=8).to(device)
        self.device = device
        self.drug_graph_model = EGNN_Sparse(
            feats_dim=63,
            pos_dim=3,
            edge_attr_dim=8,
            m_dim=128,
            dropout=0.5,
            aggr="mean").to(device)

        # 维度对齐投影层
        self.drug_seq_proj = nn.Sequential(
            nn.Linear(1024, hidden_dim),
            nn.ReLU(True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.5))
        self.rna_seq_proj = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.5))
        self.drug_3d_proj = nn.Sequential(
            nn.Linear(262+27, hidden_dim),
            nn.ReLU(True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.5))
        self.rna_3d_proj = nn.Sequential(
            nn.Linear(1024, hidden_dim),
            nn.ReLU(True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.5))

        # self.drug_seq_proj = nn.Linear(1024, hidden_dim)
        # self.rna_seq_proj = nn.Linear(768, hidden_dim)
        # self.drug_3d_proj = nn.Linear(262, hidden_dim)
        # self.rna_3d_proj = nn.Linear(1024, hidden_dim)

        self.fusion = Fusion(hidden_dim)

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.5))

        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.5))

        self.fc3 = nn.Sequential(
            nn.Linear(256, 1),
            nn.LayerNorm(hidden_dim))
    def forward(self, batch_rna, batch_mole):
        """执行药物-RNA相互作用预测的前向传播

        Args:
            drug_seq (torch.Tensor): 药物序列特征，形状为[batch_size, 1024]
            drug_3d (torch.Tensor): 药物3D结构
            RNA_seq (torch.Tensor): RNA序列特征，形状为[batch_size, 768]
            RNA_3d (torch.Tensor): RNA 3D结构

        Returns:
            torch.Tensor: 预测输出，形状为[batch_size, 1]
        """
        # [batch_size, 132]

        # [batch_size, 1024]
        # 获取 batch_size 和 channel
        batch_size = batch_rna.y.shape[0]
        channel = self.rna_point_model.channel
        # 动态计算中间维度
        num_points = batch_rna.point.shape[0] // batch_size
        point = batch_rna.point.reshape(batch_size,num_points,channel)
        RNA_3d, trans, trans_feat = self.rna_point_model(point)

        drug_3d = self.drug_graph_model(batch_mole.mol_nc_feature, batch_mole.edge_index,
                                       batch_mole.mol_edges_feature, batch_mole.batch).double()

        # 维度对齐 [batch_size, 1024]
        buff = batch_mole.buff.reshape(batch_size,27)
        # 拼接 buff 和 drug_3d
        drug_3d = torch.cat([drug_3d, buff], dim=-1)  # 在最后一个维度拼接

        drug_seq = self.drug_seq_proj(batch_mole.emb.reshape(batch_size,1024))
        RNA_seq = self.rna_seq_proj(batch_rna.emb.reshape(batch_size,768))
        drug_3d = self.drug_3d_proj(drug_3d)
        RNA_3d = self.rna_3d_proj(RNA_3d)

        feature = self.fusion(drug_seq, drug_3d, RNA_seq, RNA_3d)
        # feature = self.fusion(drug_seq, drug_seq, RNA_seq, RNA_seq)
        # 计算均值
        # feature = torch.mean(torch.stack([drug_seq, drug_3d, RNA_seq, RNA_3d]), dim=0)
        # feature = torch.mean(torch.stack([drug_3d]), dim=0)

        # 全连接层输出预测结果
        out = self.fc1(feature)
        out = self.fc2(out)
        out = self.fc3(out)
        return out