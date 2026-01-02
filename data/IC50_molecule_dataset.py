
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

import numpy as np
from .utils import get_pharmacophore



# def get_pharmacophore(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     buff = [0] * 27
#     mol_feats = fdef.GetFeaturesForMol(mol)
#     for feat in mol_feats:
#         feat_FT = feat.GetFamily() + '.' + feat.GetType()
#         if feat_FT in keys_list:
#             index = keys_list.index(feat_FT)
#             buff[index] = buff[index] + 1
#     return buff

class IC50_Molecule_dataset(InMemoryDataset):
    def __init__(self,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        root = "dataset/IC50/small_molecule/"
       

        file_path = 'data/RSM_data/R_SIM_with_IC50.xlsx'
        # read xlsx
        self.df = pd.read_excel(file_path)

        # RSIM_drug_seq = "data/R-SIM_drug_seq.csv"
        # self.seq_fea = pd.read_csv(RSIM_drug_seq, delimiter=',',index_col=0)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return "data_sm.pt"
    
    def process(self):
        data_list = []

        for index, row in self.df.iterrows():
                data = Data()
                # data.y = row['pKd']
                data.e_id = row['Entry ID']
                data.m_id = row['Molecule ID']
                data.smiles = row['SMILES']

                # 药物3D数据
                # data_3d = Data()
                file_3d = f"data/R-SIM/R-SIM_drug_3D/{row['Molecule ID']}"
                data.buff = torch.tensor(get_pharmacophore(data.smiles))
                data.point = torch.tensor(np.loadtxt(file_3d).astype(np.float32))
                # data.data_3d = data_3d  # (300,8)

                # 药物序列emb
                file_seq = f"data/R-SIM/R-SIM_drug_seq/{row['Molecule ID']}.pt"
                atom_emb = torch.load(file_seq, map_location='cpu', weights_only=False).detach().clone().float()[0]  # 89
                atom_len = atom_emb.shape[0]

                temp = torch.zeros(128 - atom_len, 1024)   # 填充部分
                data.atom_emb = torch.cat([atom_emb, temp])  # shape: (128, 1024)
                mask = []
                mask.append([] + atom_len * [1] + (128 - atom_len) * [0])
                data.mask = mask


                # data_2d = Data()
                file_2d = f"data/R-SIM/R-SIM_drug_2D/{row['Molecule ID']}.pt"
                pt_2d = torch.load(file_2d, weights_only=False)
                data.edge_attr = pt_2d.edge_attr     # (E,6)
                data.edge_index = pt_2d.edge_index   # (2,E)
                data.x = pt_2d.x                     # (N,39)


                data_list.append(data)
        
        data, slices = self.collate(data_list)
        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])