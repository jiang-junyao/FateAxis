# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:01:05 2024

@author: junyao
"""

import FateAxis.model.clf as clf
import scanpy as sc

adata1 = sc.read_h5ad('F:\\public\\LC_procssed.h5ad')
adata=sc.AnnData(adata1.X)
adata.obs = adata1.obs
adata.var = adata1.var
adata_use = adata[adata.obs.celltype.isin(['5','2'])]
sc.pp.highly_variable_genes(adata_use,subset=True, n_top_genes=2000)

fsn = clf.classification(adata_use.X, adata_use.obs.celltype,dl_epoch=10)
print(fsn.config.keys())
fsn.run_cnn1d(explain=False)