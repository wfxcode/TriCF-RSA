# -*- coding:UTF-8 -*-

# author:Feixiang Wang
# software: PyCharm

"""
文件说明：
    
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# class Mol_Fusion(nn.Module):
#     def __init__(self, hidden_dim):
#         super(Mol_Fusion, self).__init__()
#         self.module_seq = Fusion_Module(hidden_dim)
#         self.module_2d = Fusion_Module(hidden_dim)
#         self.module_3d = Fusion_Module(hidden_dim)
#         self.drug_weights = nn.Parameter(torch.ones(6))  # 初始化4个权重为1
#         self.rna_weights = nn.Parameter(torch.ones(6))  # 初始化4个权重为1
#     def forward(self, drug_seq, drug_2d, drug_3d, RNA_seq, RNA_2d, RNA_3d):
#         f_drug_seq, f_RNA_seq = self.module_seq(drug_seq, drug_2d, drug_3d, RNA_seq, RNA_2d, RNA_3d)
#
#         f_drug_2d, f_RNA_2d = self.module_2d(drug_2d, drug_3d, drug_seq, RNA_2d, RNA_3d, RNA_seq)
#
#         f_drug_3d, f_RNA_3d = self.module_3d(drug_3d, drug_seq, drug_2d, RNA_3d, RNA_seq, RNA_2d)

class Mol_Fusion(nn.Module):
    def __init__(self, hidden_dim):
        super(Mol_Fusion, self).__init__()
        self.module_seq = Fusion_Module(hidden_dim)
        self.module_2d = Fusion_Module(hidden_dim)
        self.module_3d = Fusion_Module(hidden_dim)
        self.drug_weights = nn.Parameter(torch.ones(6))  # 初始化6个权重为1
        self.rna_weights = nn.Parameter(torch.ones(6))  # 初始化6个权重为1
    def forward(self, drug_seq, drug_2d, drug_3d, RNA_seq, RNA_2d, RNA_3d):
        f_drug_seq, f_RNA_seq = self.module_seq(drug_seq, drug_2d, drug_3d, RNA_seq, RNA_2d, RNA_3d)
        f_drug_2d, f_RNA_2d = self.module_2d(drug_2d, drug_3d, drug_seq, RNA_2d, RNA_3d, RNA_seq)
        f_drug_3d, f_RNA_3d = self.module_3d(drug_3d, drug_seq, drug_2d, RNA_3d, RNA_seq, RNA_2d)

        # 收集所有药物特征（融合前3种 + 融合后3种）
        drug_features = [drug_seq, drug_2d, drug_3d, f_drug_seq, f_drug_2d, f_drug_3d]
        # 收集所有RNA特征（融合前3种 + 融合后3种）
        rna_features = [RNA_seq, RNA_2d, RNA_3d, f_RNA_seq, f_RNA_2d, f_RNA_3d]

        # 应用门控机制（sigmoid激活）
        drug_weights = torch.sigmoid(self.drug_weights)
        rna_weights = torch.sigmoid(self.rna_weights)

        # 加权融合药物特征
        final_drug = torch.zeros_like(drug_seq)
        for i, feat in enumerate(drug_features):
            final_drug += drug_weights[i] * feat

        # 加权融合RNA特征
        final_rna = torch.zeros_like(RNA_seq)
        for i, feat in enumerate(rna_features):
            final_rna += rna_weights[i] * feat

        return final_drug, final_rna




class Fusion_Module(nn.Module):
    def __init__(self,hidden_dim):
        super(Fusion_Module, self).__init__()
        self.dim = hidden_dim
        self.gru1 = nn.GRUCell(hidden_dim, hidden_dim)

        self.gru2 = nn.GRUCell(hidden_dim, hidden_dim)

        self.softmax = nn.Softmax(dim=1)

        self.out_norm = nn.LayerNorm(hidden_dim*2)

    def forward(self, drug_seq, drug_2d, drug_3d, RNA_seq, RNA_2d, RNA_3d):
        """
        Args:
            drug_seq (torch.Tensor): 药物序列特征，形状为[batch_size, hidden_dim]
            drug_3d (torch.Tensor): 药物3D结构特征，形状为[batch_size, hidden_dim]
            RNA_seq (torch.Tensor): RNA序列特征，形状为[batch_size, hidden_dim]
            RNA_3d (torch.Tensor): RNA 3D结构特征，形状为[batch_size, hidden_dim]

        Returns:
            torch.Tensor: 预测输出，形状为[batch_size, output_dim]
        """

        # 特征融合：将药物和RNA的序列特征在维度1上堆叠
        c_seq = torch.stack((drug_seq, RNA_seq), 1)
        # 初始序列特征聚合（维度压缩）
        c_seq_0 = torch.sum(c_seq, dim=1)

        # 药物特征处理分支：通过两个GRU层融合序列和3D结构特征
        c_seq_1 = self.gru1(c_seq_0, drug_2d)  # 2D结构特征融合

        c_seq_2 = self.gru1(c_seq_1, drug_3d)   # 3D结构特征融合


        # RNA特征处理分支：通过两个GRU层融合序列和3D结构特征
        c_seq_3 = self.gru2(c_seq_0, RNA_2d)   # 2D结构特征融合

        c_seq_4 = self.gru2(c_seq_3, RNA_3d)    # 3D结构特征融合


        # 注意力机制：计算双模态特征的注意力权重
        c_seq3 = torch.stack((c_seq_2, c_seq_4), 1)
        attention = self.softmax(c_seq3)  # 注意力分数计算

        # 加权特征融合：根据注意力权重融合原始序列特征
        feature = c_seq * attention
        out = self.out_norm(torch.flatten(feature, start_dim=1))
        f_drug_seq = out[:, 0: self.dim]
        f_RNA_seq = out[:, self.dim:]
        return f_drug_seq, f_RNA_seq

class Fusion(nn.Module):
    def __init__(self,hidden_dim):
        super(Fusion, self).__init__()

        self.gru1 = nn.GRUCell(hidden_dim, hidden_dim)
        # self.norm_drug1 = nn.LayerNorm(hidden_dim)
        # self.norm_drug2 = nn.LayerNorm(hidden_dim)

        self.gru2 = nn.GRUCell(hidden_dim, hidden_dim)
        # self.norm_rna1 = nn.LayerNorm(hidden_dim)
        # self.norm_rna2 = nn.LayerNorm(hidden_dim)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, drug_seq, drug_3d, RNA_seq, RNA_3d):
        """
        Args:
            drug_seq (torch.Tensor): 药物序列特征，形状为[batch_size, hidden_dim]
            drug_3d (torch.Tensor): 药物3D结构特征，形状为[batch_size, hidden_dim]
            RNA_seq (torch.Tensor): RNA序列特征，形状为[batch_size, hidden_dim]
            RNA_3d (torch.Tensor): RNA 3D结构特征，形状为[batch_size, hidden_dim]

        Returns:
            torch.Tensor: 预测输出，形状为[batch_size, output_dim]
        """

        # 特征融合：将药物和RNA的序列特征在维度1上堆叠
        c_seq = torch.stack((drug_seq, RNA_seq), 1)
        c_3d = torch.stack((drug_3d, RNA_3d), 1)
        # 初始序列特征聚合（维度压缩）
        c_seq_0 = torch.sum(c_seq, dim=1)

        # 药物特征处理分支：通过两个GRU层融合序列和3D结构特征
        c_seq_1 = self.gru1(c_seq_0, drug_seq)  # 序列特征编码

        c_seq_2 = self.gru1(c_seq_1, drug_3d)   # 3D结构特征融合


        # RNA特征处理分支：通过两个GRU层融合序列和3D结构特征
        c_seq_3 = self.gru2(c_seq_0, RNA_seq)   # 序列特征编码

        c_seq_4 = self.gru2(c_seq_3, RNA_3d)    # 3D结构特征融合


        # 注意力机制：计算双模态特征的注意力权重
        c_seq3 = torch.stack((c_seq_2, c_seq_4), 1)
        attention = self.softmax(c_seq3)  # 注意力分数计算

        # 加权特征融合：根据注意力权重融合原始序列特征
        feature = c_3d * attention
        out = torch.flatten(feature, start_dim=1)

        return out


class Fusion_LSTM(nn.Module):
    def __init__(self,hidden_dim):
        super(Fusion_LSTM, self).__init__()

        self.lstm1 = nn.LSTMCell(
            input_size=hidden_dim,
            hidden_size=hidden_dim,  # 256
            # dropout=0.2
        )
        # self.norm_drug1 = nn.LayerNorm(hidden_dim)
        # self.norm_drug2 = nn.LayerNorm(hidden_dim)

        self.lstm2 = nn.LSTMCell(
            input_size=hidden_dim,
            hidden_size=hidden_dim,  # 256
            # dropout=0.2
        )
        # self.norm_rna1 = nn.LayerNorm(hidden_dim)
        # self.norm_rna2 = nn.LayerNorm(hidden_dim)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        """
        Args:
            drug_seq (torch.Tensor): 药物序列特征，形状为[batch_size, hidden_dim]
            drug_3d (torch.Tensor): 药物3D结构特征，形状为[batch_size, hidden_dim]
            RNA_seq (torch.Tensor): RNA序列特征，形状为[batch_size, hidden_dim]
            RNA_3d (torch.Tensor): RNA 3D结构特征，形状为[batch_size, hidden_dim]

        Returns:
            torch.Tensor: 预测输出，形状为[batch_size, output_dim]
        """

        # 特征融合：将药物和RNA的序列特征在维度1上堆叠
        c_x1 = torch.stack((x, y), 1)
        c_x2 = torch.stack((x, y), 1)
        # 初始序列特征聚合（维度压缩）
        c_seq_0 = torch.sum(c_x1, dim=1)

        # 药物特征处理分支：通过两个GRU层融合序列和3D结构特征
        c_seq_1 = self.lstm1(c_seq_0, x)  # 序列特征编码

        c_seq_2 = self.lstm1(c_seq_1, x)   # 3D结构特征融合


        # RNA特征处理分支：通过两个GRU层融合序列和3D结构特征
        c_seq_3 = self.lstm2(c_seq_0, y)   # 序列特征编码

        c_seq_4 = self.lstm2(c_seq_3, y)    # 3D结构特征融合


        # 注意力机制：计算双模态特征的注意力权重
        c_seq3 = torch.stack((c_seq_2, c_seq_4), 1)
        attention = self.softmax(c_seq3)  # 注意力分数计算

        # 加权特征融合：根据注意力权重融合原始序列特征
        feature = c_x2 * attention
        out = torch.flatten(feature, start_dim=1)

        return out


def main():
    hidden_dim = 800
    # 设置超参数
    batch_size = 32
    input_dim = hidden_dim
    num_epochs = 100

    # 初始化模型
    model = Fusion(hidden_dim)

    # 创建模拟数据
    drug_seq = torch.randn(batch_size, input_dim)
    drug_3d = torch.randn(batch_size, input_dim)
    RNA_seq = torch.randn(batch_size, input_dim)
    RNA_3d = torch.randn(batch_size, input_dim)

    # 创建模拟标签
    target = torch.randn(batch_size, hidden_dim)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(drug_seq, drug_3d, RNA_seq, RNA_3d)

        # 计算损失
        loss = criterion(outputs, target)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 测试前向传播
    with torch.no_grad():
        test_output = model(drug_seq, drug_3d, RNA_seq, RNA_3d)
        print("\n测试输出形状:", test_output.shape)
        print("最后一个样本的输出示例:")
        print(test_output[0])


if __name__ == "__main__":
    main()