# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:34:23 2024

@author: JJY
"""
import warnings
warnings.filterwarnings('ignore')
import FateAxis.model.clf as clf
import scanpy as sc
import numpy as np
import pandas as pd
import FateAxis.tool.extractor as ext

### load data
data = pd.read_csv("F:/fa_result/pp_grn2/cluster_4.csv",index_col=0)
adata = sc.AnnData(data.T,)
label = [0 if 'Source state' in s else 1 for s in list(data.columns)]
adata.obs.celltype = label
adata.obs.feature_num = (adata.X > 0).sum(axis=1)
ite_num = 1
steady_state = False
outline_fea = []
outline_score = []
outline_ite = []
top500 = []

adata_use = adata
overlap_num = []
while ite_num <20 or steady_state == True:

     fsn = clf.classification(adata_use.X, label,dl_epoch=4,
                              feature_name = adata_use.var_names,
                              core_num=10)
     fsn.run_cnn1d(explain=True)
     fsn.run_gru(explain=True)
     fsn.run_lstm(explain=True)
     fsn.run_rnn(explain=True)
     fsn.run_xgb(explain=True)
     fsn.run_rf(explain=True)
     fsn.run_svm(explain=True)
     fsn.run_lgr(explain=True)
     model_ext = ext.extractor(fsn.model_acc, 
                               fsn.shap_value, 0.95)
     outline = model_ext.extract_outline_fea(z_score_cutoff=5)
     drop =  model_ext.extract_bottom_fea(0.05)
     indexes_to_drop = np.concatenate((outline, drop))
     z_score = model_ext.z_score
     z_score = z_score.drop(indexes_to_drop)
     top500_this_loop = adata.var_names[z_score.iloc[:500].index]
     top500_intersect = np.intersect1d(top500_this_loop,top500)
     overlap_num.append(len(top500_intersect))
     if outline != 'No_outline':
         for i in outline:
             outline_fea.append(fsn.feature_name[i])
             outline_score.append(model_ext.z_score[outline])
             outline_ite.append(ite_num)
            
     if outline == 'No_outline' and ite_num>3 and len(top500_intersect)>=450:
         steady_state = True
        
     top500 = top500_this_loop
     ite_num += 1
     if ite_num!=1:
         adata_use = adata_use[:,~adata_use.var_names.isin(adata_use.var_names[indexes_to_drop])]
print(overlap_num)
outline_df = pd.DataFrame({'outline_fea':outline_fea,
                            'outline_score':outline_score,
                            'outline_ite':outline_ite})