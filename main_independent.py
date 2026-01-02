import torch
from torch.utils.data import Dataset, DataLoader
import os
import torch
from data import RNA_dataset, Molecule_dataset, RNA_dataset_independent, Molecule_dataset_independent
from model import PointNetEncoder, LSTM_extracter, Trans_2D, Fusion_Module
from torch_geometric.loader import DataLoader
import torch.optim as optim
from scipy.stats import pearsonr,spearmanr
from torch.autograd import Variable
import numpy as np
import os
import torch.nn as nn
from sklearn.metrics import mean_squared_error,mean_absolute_error
import random
import logging
import shutil
import datetime

torch.set_printoptions(profile="full")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# hyperparameter
BATCH_SIZE = 32
EPOCH = 1500
seed_dataset = 2
seed = 1
# seed = 3407
# LR = 1e-3
# weight_decay = 2e-4

# LR = 1e-3
# weight_decay = 5e-3
LR = 5e-4
weight_decay = 1e-5
# weight_decay = 1e-4

dropout = 0.5
# All RNA or 6 RNA subtype: All_sf; Aptamers; miRNA; Repeats; Ribosomal; Riboswitch; Viral_RNA;

RNA_type = 'Viral_RNA_independent'
rna_dataset = RNA_dataset(RNA_type)
molecule_dataset = Molecule_dataset(RNA_type)

rna_dataset_in = RNA_dataset_independent()
molecule_dataset_in = Molecule_dataset_independent()

hidden_dim = 256
num_workers = 4
prefetch_factor = 2



# set random seed
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(precision=20)
set_seed(seed)

# combine two pyg dataset
class CustomDualDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

        assert len(self.dataset1) == len(self.dataset2)

    def __getitem__(self, index):
        try:
            return self.dataset1[index], self.dataset2[index]
        except Exception as e:
            print(f"Error at index {index}: {e}")
            raise


    def __len__(self):
        return len(self.dataset1)  



def average_multiple_lists(lists):
    return [sum(item)/len(lists) for item in zip(*lists)]





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
        self.rna_seq = LSTM_extracter(768, hidden_dim, LSTM_hideen=hidden_dim,dropout=dropout)

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
    save_dir = os.path.join('save', current_file_name, current_time)

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

    # use viral RNA to train
    train_dataset = CustomDualDataset(rna_dataset, molecule_dataset)
    # independent test
    test_dataset = CustomDualDataset(rna_dataset_in, molecule_dataset_in)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, num_workers=num_workers, drop_last=False, shuffle=False,
        pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor  # 预取2个batch
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, num_workers=num_workers, drop_last=False, shuffle=False,
        pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor  # 预取2个batch
    )

    # 初始化模型、优化器和损失函数
    model = MyModel(device).to(device)
    # logging.info(model)
    # 创建优化器
    opt_class = optim.AdamW
    optimizer = opt_class(model.parameters(), lr=LR, weight_decay=weight_decay)

    mse = torch.nn.MSELoss()
    model.to(device)

    y_pred_all = []
    max_p = -1
    max_s = -1
    # optimizer = optim.Adam(model.parameters(), lr=6e-5 , weight_decay=1e-5)
    # optimal_loss = 1e10
    # loss_fct = torch.nn.MSELoss()
    loss_func = nn.L1Loss()
    for epoch in range(0,EPOCH):
        train_loss = 0

        for step, (batch_rna, batch_mole) in enumerate(train_loader):
            optimizer.zero_grad()
            pre = model(batch_rna=batch_rna.to(device),
                        batch_mole=batch_mole.to(device))

            loss = loss_func(pre.squeeze(dim=1).view(-1,1), batch_rna.y.view(-1,1))
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss
        with torch.set_grad_enabled(False):
            model.eval()
            y_label = []
            y_pred = []
            for step, (batch_rna_test, batch_mole_test) in enumerate(test_loader):
                label = Variable(torch.from_numpy(np.array(batch_rna_test.y))).float()
                score = model(batch_rna=batch_rna_test.to(device),
                        batch_mole=batch_mole_test.to(device))
                logits = torch.squeeze(score).detach().cpu().numpy()
                label_ids = label.to('cpu').numpy()

                y_label = y_label+label_ids.flatten().tolist()
                y_pred = y_pred+logits.flatten().tolist()

            p = pearsonr(y_label, y_pred)
            s = spearmanr(y_label, y_pred)
            rmse = np.sqrt(mean_squared_error(y_label, y_pred))
            mae = mean_absolute_error(y_label, y_pred)  # 计算MAE
            # print( 'epo:',epoch, 'pcc:',p[0],'scc: ',s[0], 'rmse:',rmse)

            if max_p < p[0]:
                max_p = p[0]
                max_s = s[0]
                max_rmse = rmse
                max_mae = mae  # 更新最佳MAE
                logging.info(f"epo: {epoch}, pcc: {p[0]}, scc: {s[0]}, rmse: {rmse}, mae: {mae}")

                torch.save(model.state_dict(), 'save/' + 'model_independent_'+str(seed)+'.pth')
            elif epoch % 100 == 0:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{current_time} - tip: {epoch}, pcc: {p[0]}, scc: {s[0]}, rmse: {rmse}, mae: {mae}")


            model.train()
