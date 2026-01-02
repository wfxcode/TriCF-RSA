import torch
from torch.utils.data import Dataset, DataLoader
from data import RNA_dataset, Molecule_dataset
from model import PointNetEncoder, LSTM_extracter, Trans_2D, Fusion_Module
from torch_geometric.loader import DataLoader
import torch.optim as optim
from scipy.stats import pearsonr,spearmanr
from torch.autograd import Variable
import numpy as np
import os
import torch.nn as nn
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random
import pandas as pd
import datetime
import logging
import shutil
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# hyperparameter
BATCH_SIZE = 32
EPOCH = 1500
seed_dataset = 2
seed = 1

LR = 1e-3
weight_decay = 5e-3

dropout = 0.5

RNA_type = 'All_sf'
rna_dataset = RNA_dataset(RNA_type)
molecule_dataset = Molecule_dataset(RNA_type)

# def validate_graph_data(data):
#     """验证图数据的完整性"""
#     if hasattr(data, 'edge_index') and data.edge_index is not None:
#         max_index = data.edge_index.max().item()
#         num_nodes = data.num_nodes
#         if max_index >= num_nodes:
#             print(f"警告: 边索引超出节点范围! 最大索引: {max_index}, 节点数: {num_nodes}")
#             return False
#     return True
#
# # 在数据加载前验证
# for i, data in enumerate(molecule_dataset):
#     if not validate_graph_data(data):
#         print(f"第 {i} 个分子数据有问题")
#         # 可以选择跳过或修复这个数据点
#
# for i, data in enumerate(rna_dataset):
#     if not validate_graph_data(data):
#         print(f"第 {i} 个RNA数据有问题")

hidden_dim = 256
n_splits = 10
num_workers = 2
prefetch_factor = 1

# mole、rna、rm
blind_type = "rna"
all_df = pd.read_csv('data/RSM_data/' + 'All_sf' + '_dataset_v1.csv', delimiter='\t')  

df = [pd.read_csv(f'data/RSM_data/blind_{blind_type}/fold_{i}.csv', delimiter='\t') for i in range(1, 11)]

# print(len(df))
# df_cold_all = pd.concat([df1, df2,df3, df4,df5], axis=0).reset_index()

# set random seed
def set_seed(seed):
    """
    设置随机种子以确保实验的可复现性。

    参数：
    - seed: int, 随机种子的值。
    """
    # 设置 Python 的随机种子
    random.seed(seed)
    # 设置环境变量 PYTHONHASHSEED 以确保 hash 值的可复现性
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 设置 NumPy 的随机种子
    np.random.seed(seed)
    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)
    # 设置 PyTorch 在 CUDA 上的随机种子
    torch.cuda.manual_seed(seed)
    # 设置所有 CUDA 设备的随机种子（适用于多 GPU 场景）
    torch.cuda.manual_seed_all(seed)
    # 禁用 cuDNN 的自动优化，以确保结果的可复现性
    torch.backends.cudnn.benchmark = False
    # 设置 cuDNN 的确定性操作模式
    torch.backends.cudnn.deterministic = True
    # 设置 PyTorch 打印选项，提高输出精度
    torch.set_printoptions(precision=20)
set_seed(seed)

# combine two pyg dataset
class CustomDualDataset(Dataset):
    """
    用于同时处理两个数据集。
    """

    def __init__(self, dataset1, dataset2):
        """
        初始化 CustomDualDataset 实例。

        Args:
            dataset1 (Dataset): 第一个数据集。
            dataset2 (Dataset): 第二个数据集。

        Raises:
            AssertionError: 如果两个数据集的长度不一致。
        """
        self.dataset1 = dataset1
        self.dataset2 = dataset2

        # 确保两个数据集的长度一致
        assert len(self.dataset1) == len(self.dataset2)

    # 修改 CustomDualDataset 的 __getitem__ 方法
    def __getitem__(self, index):
        return (
            self.dataset1[index].detach().clone(),
            self.dataset2[index].detach().clone()
        )

    def __len__(self):
        return len(self.dataset1)


class Mol_Fusion(nn.Module):
    def __init__(self, hidden_dim):
        super(Mol_Fusion, self).__init__()
        self.module_seq = Fusion_Module(hidden_dim)
        self.module_2d = Fusion_Module(hidden_dim)
        self.module_3d = Fusion_Module(hidden_dim)

    def forward(self, drug_seq, drug_2d, drug_3d, RNA_seq, RNA_2d, RNA_3d):
        f_drug_seq, f_RNA_seq = self.module_seq(drug_seq, drug_2d, drug_3d, RNA_seq, RNA_2d, RNA_3d)
        f_drug_2d, f_RNA_2d = self.module_2d(drug_2d, drug_3d, drug_seq, RNA_2d, RNA_3d, RNA_seq)
        f_drug_3d, f_RNA_3d = self.module_3d(drug_3d, drug_seq, drug_2d, RNA_3d, RNA_seq, RNA_2d)

        return f_drug_seq, f_RNA_seq,f_drug_2d, f_RNA_2d,f_drug_3d, f_RNA_3d

class MyModel(nn.Module):
    def __init__(self, device):
        super(MyModel, self).__init__()
        self.device = device
        # ================= RNA特征提取 =================
        # RNA 点云网络
        self.rna_point_model = PointNetEncoder(global_feat=True, feature_transform=True, channel=8).to(device)
        # RNA点云特征维度对齐
        self.rna_3d_proj = nn.Sequential(
            nn.Linear(1024, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout))
        # RNA图神经网络
        self.rna_gcn = Trans_2D(
            input_dim=768,  # RNA节点特征维度
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            edge_dim=5,
            dropout=dropout
        ).to(device)
        # RNA序列特征
        self.rna_seq = LSTM_extracter(768, hidden_dim, LSTM_hideen=hidden_dim, dropout=dropout)

        # ================= drug特征提取 =================
        # drug点云网络
        self.drug_point_model = PointNetEncoder(global_feat=True, feature_transform=True, channel=8).to(device)
        # drug点云特征维度对齐
        self.drug_3d_proj = nn.Sequential(
            nn.Linear(1024, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout))
        # drug融合药效团
        self.drug_3d_cross = nn.Sequential(
            nn.Linear(hidden_dim+27, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim))
        # 药物图神经网络
        self.drug_gcn = Trans_2D(
            input_dim=39,  # 药物节点特征维度
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            edge_dim=6,
            dropout=dropout
        ).to(device)
        # drug序列特征
        self.drug_seq = LSTM_extracter(1024, hidden_dim, LSTM_hideen=hidden_dim,dropout=dropout)

        # ================= 特征融合部分 =================
        self.fusion1 = Mol_Fusion(hidden_dim)
        self.fusion2 = Mol_Fusion(hidden_dim)
        self.fusion3 = Mol_Fusion(hidden_dim)

        self.drug_weights = nn.Parameter(torch.ones(6))  # 初始化6个权重为1
        self.rna_weights = nn.Parameter(torch.ones(6))  # 初始化6个权重为1

        self.rna_cross = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim))
        #
        self.drug_cross = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim))
        # 预测模块
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Dropout(dropout))

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 1))

    def forward(self, batch_rna, batch_mole):
        """执行药物-RNA相互作用预测的前向传播

        Args:
            batch_rna: RNA数据批次
            batch_mole: 药物数据批次

        Returns:
            torch.Tensor: 预测输出，形状为[batch_size, 1]
        """
        batch_size = batch_rna.y.shape[0]

        # ================= RNA特征提取 =================
        # RNA点云特征提取
        # 计算点云规模
        num_points = batch_rna.point.shape[0] // batch_size
        rna_point = batch_rna.point.reshape(batch_size, num_points, -1)
        RNA_3d, _, _ = self.rna_point_model(rna_point)
        RNA_3d = self.rna_3d_proj(RNA_3d)

        # RNA序列特征提取
        RNA_seq = self.rna_seq(
            batch_rna.atom_emb.reshape(batch_size, 500, 768),
            batch_rna.mask
        )

        # RNA图特征提取
        RNA_2d = self.rna_gcn(
            batch_rna.x,
            batch_rna.edge_index,
            batch_rna.edge_attr,
            batch_rna.batch
        )

        # ================= 药物特征提取 =================
        # 药物点云特征提取
        num_points_drug = batch_mole.point.shape[0] // batch_size
        drug_point = batch_mole.point.reshape(batch_size, num_points_drug, -1)
        drug_3d, _, _ = self.drug_point_model(drug_point)

        # 添加药效团特征
        buff = batch_mole.buff.reshape(batch_size, 27).float()
        drug_3d = torch.cat([self.drug_3d_proj(drug_3d), buff], dim=-1)
        drug_3d = self.drug_3d_cross(drug_3d)

        # 药物序列特征提取
        drug_seq = self.drug_seq(
            batch_mole.atom_emb.reshape(batch_size, 128, 1024),
            batch_mole.mask
        )

        # 药物图特征提取
        drug_2d = self.drug_gcn(
            batch_mole.x,
            batch_mole.edge_index,
            batch_mole.edge_attr,
            batch_mole.batch
        )

        # ================= 特征融合部分 =================
        f_drug_seq, f_RNA_seq,f_drug_2d, f_RNA_2d,f_drug_3d, f_RNA_3d = self.fusion1(drug_seq, drug_2d, drug_3d, RNA_seq, RNA_2d, RNA_3d)
        f_drug_seq, f_RNA_seq,f_drug_2d, f_RNA_2d,f_drug_3d, f_RNA_3d = self.fusion2(f_drug_seq, f_drug_2d, f_drug_3d, f_RNA_seq, f_RNA_2d, f_RNA_3d)
        f_drug_seq, f_RNA_seq,f_drug_2d, f_RNA_2d,f_drug_3d, f_RNA_3d = self.fusion3(f_drug_seq, f_drug_2d, f_drug_3d, f_RNA_seq, f_RNA_2d, f_RNA_3d)
        # 拼接

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

        drug_combined = self.drug_cross(final_drug)
        rna_combined = self.rna_cross(final_rna)
        feature = torch.cat((drug_combined, rna_combined), dim=-1)

        # 全连接层输出预测结果
        out = self.fc1(feature)
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    # 获取当前文件名（不包括路径）
    current_file_name = os.path.splitext(os.path.basename(__file__))[0]
    # 获取当前时间并格式化为字符串
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 创建保存目录
    save_dir = os.path.join('save', current_file_name, blind_type, current_time + f"_{n_splits}")

    os.makedirs(save_dir, exist_ok=True)
    # 复制当前文件到保存目录
    shutil.copy(__file__, os.path.join(save_dir, current_file_name + '.py'))

    # 配置日志记录器
    log_file = os.path.join(save_dir, f'{current_file_name}.txt')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)
    logging.getLogger('').addHandler(console_handler)
    logging.info(save_dir)
    fold = 0
    p_list = []  # 存储每个折叠的最大皮尔逊相关系数
    s_list = []  # 存储每个折叠的最大斯皮尔曼相关系数
    r_list = []  # 存储每个折叠的最大均方根误差
    m_list = []  # 存储每个折叠的MAE
    for i, df_f in enumerate(df):

        test_id = df_f['Entry_ID'].tolist()
        test_id = all_df[all_df['Entry_ID'].isin(test_id)].index.tolist()

        train_id = []
        for j, other_id in enumerate(df):
            if j != i:
                train_id.extend(other_id['Entry_ID'].tolist())

        train_id = all_df[all_df['Entry_ID'].isin(train_id)].index.tolist()

        logging.info(test_id)
        logging.info(f"{len(test_id)}, {len(train_id)}")

        max_p = -1  # 初始化最大皮尔逊相关系数
        max_s = -1  # 初始化最大斯皮尔曼相关系数
        max_rmse = 0  # 初始化最大均方根误差
        max_mae = 0  # 初始化最大平均绝对误差
        fold += 1
        # if fold != 5:
        #     continue
        logging.info(f"Fold {fold}")


        train_dataset = CustomDualDataset(rna_dataset[train_id], molecule_dataset[train_id])
        test_dataset = CustomDualDataset(rna_dataset[test_id], molecule_dataset[test_id])


        # 创建训练和测试数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, num_workers=num_workers, drop_last=False, shuffle=False,
            pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor  # 预取2个batch
        )
        # train_loader = DataLoader(
        #     train_dataset, batch_size=BATCH_SIZE, num_workers=8, drop_last=False, shuffle=False)
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, num_workers=num_workers, drop_last=False, shuffle=False,
            pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor  # 预取2个batch
        )


        model = MyModel(device).to(device)


        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)

        mse = torch.nn.MSELoss()
        for epo in range(EPOCH):
            # train
            train_loss = 0
            for step, (batch_rna,batch_mole) in enumerate(train_loader):
                optimizer.zero_grad()
                pre = model(batch_rna.to(device), batch_mole.to(device))
                loss = mse(pre.squeeze(dim=1).view(-1,1), batch_rna.y.view(-1,1))
                loss.backward()
                optimizer.step()
                train_loss = train_loss + loss
            # test
            with torch.set_grad_enabled(False):
                test_loss = 0
                model.eval()
                y_label = []
                y_pred = []
                for step, (batch_rna_test,batch_mole_test) in enumerate(test_loader):
                    label = Variable(torch.from_numpy(np.array(batch_rna_test.y))).float()
                    score = model(batch_rna_test.to(device), batch_mole_test.to(device))
                    n = torch.squeeze(score, 1)
                    logits = torch.squeeze(score).detach().cpu().numpy()
                    label_ids = label.to('cpu').numpy()
                    loss_t = mse(n.cpu().view(-1,1), label.view(-1,1))
                    y_label.extend(label_ids.flatten().tolist())
                    y_pred.extend(logits.flatten().tolist())
                    test_loss = test_loss + loss_t

            model.train()
            # 计算评估指标
            p = pearsonr(y_label, y_pred)
            s = spearmanr(y_label, y_pred)
            rmse = np.sqrt(mean_squared_error(y_label, y_pred))
            mae = mean_absolute_error(y_label, y_pred)  # 计算MAE
            if max_p < p[0]:
                max_p = p[0]
                max_s = s[0]
                max_rmse = rmse
                max_mae = mae  # 更新最佳MAE
                logging.info(f"epo: {epo}, pcc: {p[0]}, scc: {s[0]}, rmse: {rmse}, mae: {mae}")
                model_save_path = os.path.join(save_dir,
                                               'model_' + str(RNA_type) + str(seed_dataset) + '_' + str(
                                                   fold) + '_' + str(seed) + '.pth')
                torch.save(model.state_dict(), model_save_path)
                # torch.save(model.state_dict(), 'save/' + 'model_blind_'+str(blind_type)+str(seed_dataset)+'_'+str(fold)+'_'+str(seed)+'.pth')
            elif epo % 100 == 0:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{current_time} - tip: {epo}, pcc: {p[0]}, scc: {s[0]}, rmse: {rmse}, mae: {mae}")

        p_list.append(max_p)
        r_list.append(max_rmse)
        s_list.append(max_s)
        m_list.append(max_mae)  # 将最佳MAE添加到列表

        logging.info(f"Average p: {np.mean(p_list)}")
        logging.info(f"Average s: {np.mean(s_list)}")
        logging.info(f"Average rmse: {np.mean(r_list)}")
        logging.info(f"Average mae: {np.mean(m_list)}")  # 打印平均MAE

