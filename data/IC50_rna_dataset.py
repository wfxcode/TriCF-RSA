import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
import numpy as np
from torch_geometric.data import Data
from .utils import KD_to_pKD

class IC50_RNA_dataset(InMemoryDataset):
    def __init__(self,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        
        root = "dataset/IC50/rna/"
            

        file_path = 'data/RSM_data/R_SIM_with_IC50.xlsx'
        # read xlsx
        self.df = pd.read_excel(file_path)
      

        # language model embedding floder
        # RSIM_RNA_seq = "data/R-SIM_RNA_seq.csv"
        # self.seq_fea = pd.read_csv(RSIM_RNA_seq, delimiter=',', index_col=0)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return "data_rna.pt"


    def process(self):
        # 数据标准化处理
        y = self.df['IC50 (M)']
        y = y.apply(lambda x: KD_to_pKD(x))
        min_y = min(y)
        max_y = max(y)
        data_list = []
        for index, row in self.df.iterrows():
            data = Data()
            data.sequence = row['RNA sequence']
            data.t_id = row["RNA ID"]
            data.e_id = row['Entry ID']
            data.y = (y[index] - min_y) / (max_y - min_y)
            # data.y = pkd_mapping[data.e_id]
            # data.emb = torch.tensor(self.seq_fea.loc[data.t_id])

            # RNA 3D数据
            file_3d = f"data/R-SIM/R-SIM_RNA_3D/{data.t_id}"
            data.point = torch.tensor(np.loadtxt(file_3d).astype(np.float32))
            # data.data_3d = data_3d # (2000,8)

            # RNA序列嵌入
            file_emb = f"data/R-SIM/R-SIM_RNA_seq/{data.t_id}.pt"
            atom_emb = torch.load(file_emb, map_location='cpu', weights_only=False).detach().clone().float()[0]
            atom_len = atom_emb.shape[0]
            temp = torch.zeros(500 - atom_len, 768)  # 填充部分
            data.atom_emb = torch.cat([atom_emb, temp])  # shape: (500, 768)
            mask = []
            mask.append([] + atom_len * [1] + (500 - atom_len) * [0])
            data.mask = mask

            # RNA 2D数据
            file_2d = f"data/R-SIM/R-SIM_RNA_2D/{data.t_id}.pt"
            pt_2d = torch.load(file_2d, weights_only=False)
            data.edge_index = pt_2d.edge_index  # (2,E)
            data.x = atom_emb                   # (N,768)
            data.edge_attr = pt_2d.edge_attr    # (E,5)

            data_list.append(data)


        data, slices = self.collate(data_list)
        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])
