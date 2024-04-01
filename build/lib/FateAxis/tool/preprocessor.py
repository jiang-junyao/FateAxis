# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:32:57 2024

@author: junyao
"""
import sys
import scanpy as sc
import pandas as pd
import pkg_resources
import numpy as np

class pper:
    def __init__(self,
                 obj,
                 group,
                 species='hs',):
        
        self.adata = obj
        self.group=group

        
        ### set tf db
        if species == 'hs':
            self.tf_path = pkg_resources.resource_filename('FateAxis', 
                                                      'tfdb/hs_tf.txt')
        elif species == 'mm':
            self.tf_path = pkg_resources.resource_filename('FateAxis', 
                                                      'tfdb/mm_tf.txt')
        else:
            sys.exit("please input correct species name: mm or hs")
        
    def extract_fea(self,pval_thr=0.05,tf_exp_thr=0.05):
        
        ### extract deg
        sc.tl.rank_genes_groups(self.adata,self.group, method='wilcoxon')
        result = self.adata.uns['rank_genes_groups']

        groups = result['names'].dtype.names
        differential_genes = []
        for group in groups:
            genes = result['names'][group]
            pvals_adj = result['pvals_adj'][group]
            
            for gene, pval_adj in zip(genes, pvals_adj):
                if pval_adj < pval_thr:
                    differential_genes.append(gene)
        differential_genes = list(set(differential_genes))
        
        ### extract expressed tf
        tf_name = pd.read_table(self.tf_path,header=None)[0].tolist()
        tf_name = np.intersect1d(tf_name, self.adata.var_names)
        cell_counts = {}
        
        for gene in tf_name:
            
            expression = self.adata[:, self.adata.var_names == gene].X
            if isinstance(expression, np.ndarray):
                count = np.sum(expression > 0)
            else:
                count = np.sum(expression.toarray() > 0)
            cell_counts[gene] = count
    
        total_cells = self.adata.shape[0]
        tf_exp_thr = total_cells * tf_exp_thr
        selected_tf = [gene for gene, count in cell_counts.items() if count >
                       tf_exp_thr]
        
        ### subset the adata
        final_fea = list(set(differential_genes + selected_tf))
        self.adata = self.adata[:,final_fea]
        
        
        
        