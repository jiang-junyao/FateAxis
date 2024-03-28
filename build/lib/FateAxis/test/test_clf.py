# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:01:05 2024

@author: junyao
"""

import FateAxis.model.clf as clf
import scanpy as sc

adata = sc.read_h5ad('F:\\public\\LC_procssed.h5ad')
sc.pp.normalize_total(adata)
adata_use = adata[adata.obs.celltype.isin(['5','0'])]

fsn = clf.classification(adata_use.X, adata_use.obs.celltype)
print(fsn.config.keys())
fsn.run_gbm()