import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
import numpy as np
from torch_geometric.data import Data
import math
def KD_to_pKD(KD):
    return -math.log10(KD)

class RNA_dataset_independent(InMemoryDataset):
    def __init__(self,
                 # RNA_type,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        
        self.root = "dataset/rna_independent"
            

        # All RNA or 6 RNA subtype: All_sf; Aptamers; miRNA; Repeats; Ribosomal; Riboswitch; Viral_RNA;
        # csv_file_path = 'data/RSM_data/' + RNA_type + '_dataset_v1.csv'
        # self.df = pd.read_csv(csv_file_path, delimiter='\t')

        csv_file_path_qsar = 'data/independent_data.csv'
        # read csv
        self.df_qsar = pd.read_csv(csv_file_path_qsar, delimiter=',')

        # language model embedding floder
        # RSIM_RNA_seq = "data/R-SIM_RNA_seq.csv"
        # self.seq_fea = pd.read_csv(RSIM_RNA_seq, delimiter=',', index_col=0)
        super().__init__(self.root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return "data_rna.pt"


    def process(self):

        data_list = []
        min_pKd = -0.477121254719662
        max_pKd = 11.301029995664

        for index in range(0,48):
            data = Data()

            pKd = KD_to_pKD(self.df_qsar.iloc[index]['KD'])
            data.y = (pKd-min_pKd) / (max_pKd-min_pKd)

            sequence = 'GGCAGAUCUGAGCCUGGGAGCUCUCUGCC'
            data.sequence = sequence
            data.t_id = "hiv"
            # data.t_id = "independent"
            # data.e_id = row['Entry_ID']
            # data.emb = torch.tensor(self.seq_fea.loc[data.t_id])
            # RNA 3D数据
            file_path = f"data/independent/independent_RNA_3D"
            data.point = torch.tensor(np.loadtxt(file_path).astype(np.float32))

            # RNA序列嵌入
            file_emb = f"data/independent/independent_RNA_seq.pt"
            atom_emb = torch.load(file_emb, map_location='cpu').detach().clone().float()[0]  # 89
            atom_len = atom_emb.shape[0]

            temp = torch.zeros(500 - atom_len, 768)  # 填充部分
            data.atom_emb = torch.cat([atom_emb, temp])  # shape: (512, 128)
            mask = []
            mask.append([] + atom_len * [1] + (500 - atom_len) * [0])
            data.mask = mask

            # RNA 2D数据
            file_2d = f"data/independent/independent_RNA_2D.pt"
            pt_2d = torch.load(file_2d)
            data.edge_index = pt_2d.edge_index  # (2,E)
            data.x = atom_emb                   # (N,768)
            data.edge_attr = pt_2d.edge_attr    # (E,5)

            data_list.append(data)

        data, slices = self.collate(data_list)
        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])
