# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:01:05 2024

@author: junyao
"""
import warnings
warnings.filterwarnings('ignore')
import FateAxis.model.clf as clf
import scanpy as sc
import numpy as np
import pandas as pd
import FateAxis.tool.extractor as ext

adata1 = sc.read_h5ad('F:\\public\\LC_procssed.h5ad')
adata=sc.AnnData(adata1.X)
adata.obs = adata1.obs
adata.var = adata1.var
adata_use = adata[adata.obs.celltype.isin(['5','2'])]
sc.pp.highly_variable_genes(adata_use,subset=True, n_top_genes=2000)
fsn = clf.classification(adata_use.X, adata_use.obs.celltype,dl_epoch=2,
                              feature_name = adata_use.var_names)
fsn.run_hybrid(explain=True)

ite_num = 1
steady_state = False
outline_fea = []
outline_score = []
outline_ite = []
top100 = []


# while ite_num <10 or steady_state == True:
#     print(adata_use)
#     fsn = clf.classification(adata_use.X, adata_use.obs.celltype,dl_epoch=2,
#                              feature_name = adata_use.var_names)
#     fsn.run_gru(explain=True)
#     fsn.run_lgr(explain=True)
#     model_ext = ext.extractor(fsn.model_acc, 
#                               fsn.shap_value, 0.95)
#     outline = model_ext.extract_outline_fea(z_score_cutoff=5)
#     drop =  model_ext.extract_bottom_fea(0.05)
#     indexes_to_drop = np.concatenate((outline, drop))
#     z_score = model_ext.z_score
#     z_score = z_score.drop(indexes_to_drop)
#     top100_this_loop = adata_use.var_names[z_score.iloc[:100].index]
#     top100_intersect = np.intersect1d(top100_this_loop,top100)
#     print(len(top100_intersect))
#     if outline != 'No_outline':
#         for i in outline:
#             outline_fea.append(fsn.feature_name[i])
#             outline_score.append(model_ext.z_score[outline])
#             outline_ite.append(ite_num)
            
#     print(outline_fea)
#     if outline == 'No_outline' and ite_num>3 and len(top100_intersect)>=95:
#         steady_state = True
        
#     top100 = top100_this_loop
#     ite_num += 1
#     if ite_num!=1:
#         adata_use = adata_use[:,~adata_use.var_names.isin(adata_use.var_names[indexes_to_drop])]
        
# outline_df = pd.DataFrame({'outline_fea':outline_fea,
#                            'outline_score':outline_score,
#                            'outline_ite':outline_ite})