# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 01:00:24 2025

@author: peiweike
"""

import FateAxis.tool.preprocessor as pp
import scanpy as sc
import pandas as pd
import os 
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = [6, 4.5]
plt.rcParams["savefig.dpi"] = 300

adata = sc.read_h5ad('/data/jiangjunyao/easyGRN/processed_data/celltag_multi_iep.prcessed.h5ad')
adata_use = adata[adata.obs.celltype.isin(['early','reprogramming'])]
all_baseGRN = os.listdir('/data/jiangjunyao/easyGRN/processed_data/celltag_multi_iep/baseGRN/')
for i in all_baseGRN:
    for j in [10,20,50]:
        for k in [0.05,0.0005]:
            baseGRN = pd.read_csv('/data/jiangjunyao/easyGRN/processed_data/celltag_multi_iep/baseGRN/'+i)
            processor=pp.pper(adata_use, 'celltype',species='mm',
                              baseGRN=baseGRN,ncores=40)
            processor.extract_fea('hvg',2000,tf_exp_thr=0.05)
            processor.construct_metacell_grn(use_rep='X_scVI',metacell_method='knn',metacell_num=j)
            processor.construct_full_grn(use_rep='X_scVI')
            grp_mt = processor.filter_grp(k)
            grp_name = str(i)+'-metacell_num_'+str(j)+'-sig_grp_pval_'+str(k)+'.csv'
            grp_mt.to_csv('/data/jiangjunyao/easyGRN/processed_data/celltag_multi_iep/grp_mt/'+grp_name)