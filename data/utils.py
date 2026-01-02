# -*- coding:UTF-8 -*-

# author:Feixiang Wang
# software: PyCharm

"""
文件说明：
    
"""
import os

import math

def get_pharmacophore(smiles):
    from rdkit import Chem
    from rdkit import RDConfig
    from rdkit.Chem import AllChem

    fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))
    keys_list = list(fdef.GetFeatureDefs().keys())

    mol = Chem.MolFromSmiles(smiles)
    buff = [0] * 27
    if mol is None:
        print("None")
        return buff
    mol_feats = fdef.GetFeaturesForMol(mol)
    for feat in mol_feats:
        # print(feat.GetFamily(), feat.GetType(), feat.GetAtomIds())
        # print(pos.x, pos.y, pos.z)
        feat_FT = feat.GetFamily() + '.' + feat.GetType()
        if feat_FT in keys_list:
            index = keys_list.index(feat_FT)
            buff[index] = buff[index] + 1
    return buff

def KD_to_pKD(KD):
    return -math.log10(KD)


