import os
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import numpy as np
from .utils import get_pharmacophore

class Molecule_dataset_independent(InMemoryDataset):
    def __init__(self,
                 # RNA_type,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        root = "dataset/small_molecule_independent/"

        csv_file_path_qsar = 'data/independent_data.csv'

        # read csv
        self.df_qsar = pd.read_csv(csv_file_path_qsar, delimiter=',')

        # RSIM_drug_seq = "data/R-SIM_drug_seq.csv"
        # self.seq_fea = pd.read_csv(RSIM_drug_seq, delimiter=',',index_col=0)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return "data_sm.pt"
    
    def process(self):
        # drug_vocab = WordVocab.load_vocab('data/smiles_vocab.pkl')
        data_list = []


        for index in range(0, 48):
        # for index, row in self.df.iterrows():
            data = Data()
            row = self.df_qsar.iloc[index]
            data.smiles = row['SMILES']
            # data.x = torch.from_numpy(x).to(torch.int64)
            # data.graph_len = len(data.x)

            # data.smiles_ori = row['SMILES']
            # pKD = KD_to_pKD(row['KD'])
            # data.e_id = row['Entry_ID']
            data.m_id = row['Name']
            data.smiles = row['SMILES']
            # 药物3D数据
            # data_3d = Data()
            file_3d = f"data/independent/independent_drug_3D/{data.m_id}"
            data.buff = torch.tensor(get_pharmacophore(data.smiles))
            data.point = torch.tensor(np.loadtxt(file_3d).astype(np.float32))


            # data.emb = torch.tensor(self.seq_fea.loc[data.m_id])

            file_2d = f"data/independent/independent_drug_2D/{data.m_id}.pt"
            pt_2d = torch.load(file_2d)
            data.edge_attr = pt_2d.edge_attr  # (E,6)
            data.edge_index = pt_2d.edge_index  # (2,E)
            data.x = pt_2d.x  # (N,39)

            file_emb = f"data/independent/independent_drug_seq/{data.m_id}.pt"
            atom_emb = torch.load(file_emb, map_location='cpu').detach().clone().float()[0]  # 89
            atom_len = atom_emb.shape[0]

            temp = torch.zeros(128 - atom_len, 1024)   # 填充部分
            data.atom_emb = torch.cat([atom_emb, temp])  # shape: (512, 128)
            mask = []
            mask.append([] + atom_len * [1] + (128 - atom_len) * [0])
            data.mask = mask
            # 确保所有属性存在且非空
            # assert data.smiles_ori is not None, f"Missing smiles_ori for index {data.m_id}"
            # assert data.y is not None, f"Missing y for index {data.m_id}"
            # assert data.e_id is not None, f"Missing e_id for index {data.m_id}"
            # assert data.m_id is not None, f"Missing m_id for index {data.m_id}"
            # assert data.x is not None, f"Missing x for index {data.m_id}"
            # assert data.mol_len is not None, f"Missing mol_len for index {data.m_id}"
            # assert data.mol_edges_feature is not None, f"Missing mol_edges_feature for index {data.m_id}"
            # assert data.edge_index is not None, f"Missing edge_index for index {data.m_id}"
            # assert data.mol_nc_feature is not None, f"Missing mol_nc_feature for index {data.m_id}"

            data_list.append(data)
        
        data, slices = self.collate(data_list)
        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])