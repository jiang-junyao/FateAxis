"""
Generate GRP-cell matrix using celloracle

author: junyao
"""
import sys
import scanpy as sc
import pandas as pd
import pkg_resources
import numpy as np
import celloracle as co
class pper:
    def __init__(self,
                 obj,
                 group,
                 baseGRN=None,
                 species='hs',
                 ncores=10):
        
        self.adata = obj
        self.group = group
        self.species = species
        self.ncores = ncores

        
        ### set tf db
        if species == 'hs':
            self.tf_path = pkg_resources.resource_filename('FateAxis', 
                                                      'tfdb/hs_tf.txt')
        elif species == 'mm':
            self.tf_path = pkg_resources.resource_filename('FateAxis', 
                                                      'tfdb/mm_tf.txt')
        else:
            sys.exit("please input correct species name: mm or hs")

        ### use celloracle defualt grn
        if (baseGRN is None) & (self.species=='hs'):
            self.baseGRN = co.data.load_human_promoter_base_GRN()
        elif (baseGRN is None) & (self.species=='mm'):
            baseGRN = co.data.load_mouse_scATAC_atlas_base_GRN()
        else:
            self.baseGRN = baseGRN
            
    def extract_fea(self,pval_thr=0.05,tf_exp_thr=0.05):
        
        ### extract deg
        sc.pp.log1p(self.adata)
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
        self.grn_fea = list(set(differential_genes + selected_tf))
        print('sig deg num:'+str(len(differential_genes)))
        print('expressed tf num:'+str(len(selected_tf)))
        print('total feature num:'+str(len(self.grn_fea)))
        
    def construct_full_grn(self,embedding_name='X_umap'):
        
        adata_input = self.adata[:,self.grn_fea]
        grn = self.run_celloracle(adata_input,self.baseGRN,
                                  self.group,embedding_name,self.ncores)
        full_grn = {}
        full_index = {}
        for i in grn.links_dict.keys():
            raw_grn = grn.links_dict[i]
            filtered_grn = raw_grn[raw_grn['p']<0.05]
            filtered_grn = filtered_grn[filtered_grn['coef_abs']>0]
            full_grn[i] = filtered_grn
            full_index[i] = filtered_grn['source']+'#'+filtered_grn['target']
            print('celltype '+str(i) + '. raw pair '+str(raw_grn.shape[0])+\
                  ' filter pair '+str(filtered_grn.shape[0]))
        self.full_grn = full_grn
        self.full_index = full_index

    def construct_metacell_grn(self,embedding_name='X_umap',metacell_num=20,
                               cell_number=100):
        
        adata_input = self.adata[:,self.grn_fea]
        group1 = list(set(adata_input.obs[self.group]))[0]
        group2 = list(set(adata_input.obs[self.group]))[1]
        group1_cell = adata_input.obs_names[adata_input.obs[self.group]==group1]
        group2_cell = adata_input.obs_names[adata_input.obs[self.group]==group2]

        meta_grn_dict = {}
        for i in range(metacell_num):
           print(i)
           group1_meta = np.random.choice(group1_cell, size=cell_number, 
                                          replace=False).tolist()
           group2_meta = np.random.choice(group2_cell, size=cell_number, 
                                          replace=False).tolist()
           adata_meta = adata_input[group1_meta+group2_meta]
           sc.pp.filter_genes(adata_meta,min_cells=3)
           print(adata_meta)
           meta_grn = self.run_celloracle(adata_meta,self.baseGRN,
                                          self.group,embedding_name,
                                          self.ncores)
           #meta_grn_dict[i] = self.__filter_grp(meta_grn.links_dict)
           meta_grn_dict[i] = meta_grn.links_dict
        self.meta_grn = meta_grn_dict
    
    def run_celloracle(self,adata,baseGRN=None,group='celltype',
                       embedding_name='X_umap',ncores=10,filter=None):
        
        print('build grn based on celloracle')
        oracle = co.Oracle()
        oracle.import_anndata_as_raw_count(adata=adata,
                                   cluster_column_name=group,
                                   embedding_name=embedding_name)
        oracle.import_TF_data(TF_info_matrix=baseGRN)
        oracle.perform_PCA()
        n_comps = np.where(np.diff(np.diff(np.cumsum(oracle.pca.explained_variance_ratio_))>0.002))[0][0]
        n_cell = oracle.adata.shape[0]
        k = int(0.025*n_cell)
        oracle.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k*8,
                              b_maxl=k*4, n_jobs=ncores)
        links = oracle.get_links(cluster_name_for_GRN_unit=group, alpha=10,
                         verbose_level=10)
        return links

    def filter_grp(self,grn):
        filtered_grn = self.meta_grn
        for i in self.meta_grn.keys():
            grn_meta = self.meta_grn[i]
            for j in grn_meta.keys():
                grn_use = grn_meta[j]
                grn_use.index = grn_use['source']+'#'+grn_use['target']
                grn_use = grn_use[grn_use['p']<0.05]
                valid_grp = np.intersect1d(list(grn_use.index),self.full_index[j])
                grn_use = grn_use.loc[valid_grp]
                filtered_grn[i][j] = grn_use
        all_indexes = set(index for d in filtered_grn.values() for sd in d.values() for index in sd.index)
        
        result = pd.concat({(str(i)+'#'+str(j)): \
                            filtered_grn[i][j]['coef_mean'].reindex(all_indexes, 
                                                                    fill_value=0)
                            for i in filtered_grn.keys() 
                            for j in filtered_grn[i].keys()}, 
                           axis=1)
        return result
