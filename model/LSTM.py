# -*- coding:UTF-8 -*-

# author:Feixiang Wang
# software: PyCharm

"""
文件说明：
    
"""
import torch
import torch.nn as nn


class LSTM_extracter(nn.Module):
    def __init__(self, input, hidden_size, LSTM_hideen=128, dropout=0.5):
        super(LSTM_extracter, self).__init__()
        self.line_emb = nn.Linear(input, hidden_size)
        self.relu = nn.ReLU()
        # if input==1024:
        self.RNN = nn.LSTM(
            input_size=hidden_size,
            hidden_size=LSTM_hideen,  # 256
            num_layers=2,  # LSTM层数
            bidirectional=True,  # 启用双向
            batch_first=True,  # 输入形状为(batch, seq, feature)
            # dropout=0.2
        )
        self.seq_proj = nn.Sequential(
            nn.Linear(LSTM_hideen*2, LSTM_hideen*4),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(LSTM_hideen * 4, hidden_size),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size))
    def forward(self, atom_emb, mask):
        atom_emb = self.relu(self.line_emb(atom_emb))
        latm_out, _  = self.RNN(atom_emb)
        # 掩码池化
        mask = torch.as_tensor(mask, device=latm_out.device)  # shape: (batch_size, 1, 128)
        mask = mask.permute(0, 2, 1)
        valid_length = mask.sum(dim=1, keepdim=True)
        emb_seq = (latm_out * mask).sum(dim=1) / valid_length.squeeze(2)  # shape: (batch_size, 256)
        emb_seq = self.seq_proj(emb_seq)
        return emb_seq

class GRU_extracter(nn.Module):
    def __init__(self, input, hidden_size, GRU_hidden,dropout):
        super(GRU_extracter, self).__init__()

        self.line_emb = nn.Linear(input, hidden_size)
        self.relu = nn.ReLU()
        # if input==1024:
        self.RNN = nn.GRU(
            input_size=hidden_size,
            hidden_size=GRU_hidden,  # 256
            num_layers=2,  # LSTM层数
            bidirectional=True,  # 启用双向
            batch_first=True,  # 输入形状为(batch, seq, feature)
            # dropout=0.2
        )
        self.seq_proj = nn.Sequential(
            nn.Linear(GRU_hidden * 2, GRU_hidden * 4),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(GRU_hidden * 4, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(True),
            nn.Dropout(dropout))
        # elif input==768:
        #     self.RNN = BILSTM(hidden_size, LSTM_hideen)
        #     self.seq_proj = nn.Sequential(
        #         nn.Linear(500, hidden_size),
        #         nn.LayerNorm(hidden_size),
        #         nn.ReLU(True),
        #         nn.Dropout(dropout))

    def forward(self, atom_emb, mask):
        atom_emb = self.relu(self.line_emb(atom_emb))
        latm_out, _ = self.RNN(atom_emb)
        # 掩码池化
        mask = torch.as_tensor(mask, device=latm_out.device)  # shape: (batch_size, 1, 128)
        mask = mask.permute(0, 2, 1)
        valid_length = mask.sum(dim=1, keepdim=True)
        emb_seq = (latm_out * mask).sum(dim=1) / valid_length.squeeze(2)  # shape: (batch_size, 256)
        emb_seq = self.seq_proj(emb_seq)
        return emb_seq


if __name__ == "__main__":
    # 基础配置
    batch_size = 4
    seq_len = 128
    input_dim = 1024
    hidden_size = 256

    # 初始化模型
    model = LSTM_extracter(
        input=input_dim,
        hidden_size=hidden_size,
        LSTM_hideen=128
    )

    # 创建测试输入
    atom_emb = torch.randn(batch_size, seq_len, input_dim)  # [batch, seq, feature]
    mask = torch.ones(batch_size, 1, seq_len)  # 全有效序列

    # 前向传播
    output = model(atom_emb, mask)

    # 验证输出
    print(f"输入形状：{atom_emb.shape} → 输出形状：{output.shape}")
    print(f"输出值范围：[{output.min().item():.4f}, {output.max().item():.4f}]")
    print("NaN检查：", torch.isnan(output).any())
    print("Inf检查：", torch.isinf(output).any())