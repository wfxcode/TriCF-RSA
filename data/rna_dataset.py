import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
import numpy as np
from torch_geometric.data import Data

class RNA_dataset(InMemoryDataset):
    def __init__(self,
                 RNA_type,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        
        root = "dataset/rna/" + RNA_type
            

        # All RNA or 6 RNA subtype: All_sf; Aptamers; miRNA; Repeats; Ribosomal; Riboswitch; Viral_RNA;
        csv_file_path = 'data/RSM_data/' + RNA_type + '_dataset_v1.csv'  
        self.df = pd.read_csv(csv_file_path, delimiter='\t')
      

        # language model embedding floder
        # RSIM_RNA_seq = "data/R-SIM_RNA_seq.csv"
        # self.seq_fea = pd.read_csv(RSIM_RNA_seq, delimiter=',', index_col=0)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return "data_rna.pt"


    def process(self):
        # 数据标准化处理
        path = "data/RSM_data/All_sf_dataset_v1.csv"
        df = pd.read_csv(path, delimiter='\t')
        df['pKd_normalized'] = (df['pKd'] - df['pKd'].min()) / (df['pKd'].max() - df['pKd'].min())
        pkd_mapping = dict(zip(df['Entry_ID'], df['pKd_normalized']))

        data_list = []
        for index, row in self.df.iterrows():
            data = Data()
            data.sequence = row['Target_RNA_sequence']
            data.t_id = row["Target_RNA_ID"]
            data.e_id = row['Entry_ID']
            data.y = pkd_mapping[data.e_id]
            # data.emb = torch.tensor(self.seq_fea.loc[data.t_id])

            # RNA 3D数据
            file_3d = f"data/R-SIM/R-SIM_RNA_3D/{data.t_id}"
            data.point = torch.tensor(np.loadtxt(file_3d).astype(np.float32))
            # data.data_3d = data_3d # (2000,8)

            # RNA序列嵌入
            file_emb = f"data/R-SIM/R-SIM_RNA_seq/{data.t_id}.pt"
            atom_emb = torch.load(file_emb, map_location='cpu').detach().clone().float()[0]
            atom_len = atom_emb.shape[0]
            temp = torch.zeros(500 - atom_len, 768)  # 填充部分
            data.atom_emb = torch.cat([atom_emb, temp])  # shape: (500, 768)
            mask = []
            mask.append([] + atom_len * [1] + (500 - atom_len) * [0])
            data.mask = mask

            # RNA 2D数据
            file_2d = f"data/R-SIM/R-SIM_RNA_2D/{data.t_id}.pt"
            pt_2d = torch.load(file_2d)
            data.edge_index = pt_2d.edge_index  # (2,E)
            data.x = atom_emb                   # (N,768)
            data.edge_attr = pt_2d.edge_attr    # (E,5)

            data_list.append(data)


        data, slices = self.collate(data_list)
        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])
