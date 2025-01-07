# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 16:00:15 2024

@author: junyao
"""

import FateAxis.tool.preprocessor as pp
import scanpy as sc
adata1 = sc.read_h5ad('F:\\public\\LC_procssed.h5ad')
adata=sc.AnnData(adata1.X)
adata.obs = adata1.obs
adata.var = adata1.var
adata_use = adata[adata.obs.celltype.isin(['5','2'])]
sc.pp.normalize_total(adata_use)
sc.pp.highly_variable_genes(adata_use)
sc.pp.pca(adata_use)
processor=pp.pper(adata_use, 'celltype','hs')
processor.extract_fea()
print(adata_use)
print(processor.adata)