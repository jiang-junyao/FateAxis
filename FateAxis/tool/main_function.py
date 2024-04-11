# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 17:03:56 2024

This script is the main function of the FateAxis

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
data = pd.read_csv("/data/jiangjunyao/fa_result/pp_grn2/cluster4_m.csv",index_col=0)
data = data.abs()
adata = sc.AnnData(data.T,)
label = [0 if 'Source' in s else 1 for s in list(data.columns)]
config_path = '/data/jiangjunyao/FateAxis/FateAxis/config/config1.js'
def calculate_grp_importance(data,
                            label,
                            config_path = None,
                            model_use = ['cnn','gru','lstm','rnn','gbm'
                                         ,'rf','svm','lgr'],
                            core_use = 10,
                            max_loop_number = 10,
                            model_acc_threshold = 0.9,
                            outlier_z_score_threshold = 3,
                            outlier_quantile = 90):
    ### standlize data
    adata.obs.celltype = label
    adata.obs.feature_num = (adata.X > 0).sum(axis=1)
    ite_num = 1
    steady_state = False
    outline_fea_all = []
    outline_score_all = []
    outline_ite_all = []
    outline_percent_all = []
    top500 = []
    
    adata_use = adata
    overlap_num = []
    while ite_num < max_loop_number+1 and steady_state == False:
         print('Classify grp at iteration: '+str(ite_num))
         fsn = clf.classification(adata_use.X, label,config_path,dl_epoch=5,
                                  feature_name = adata_use.var_names,
                                  core_num=core_use,acc_cut=model_acc_threshold)
         ### lanuch model
         if 'cnn' in model_use:
             fsn.run_cnn1d(explain=True)
         if 'gru' in model_use:
             fsn.run_gru(explain=True)
         if 'lstm' in model_use:
             fsn.run_lstm(explain=True)
         if 'rnn' in model_use:
             fsn.run_rnn(explain=True)
         if 'gbm' in model_use:
             fsn.run_gbm(explain=True)
         if 'rf' in model_use:
             fsn.run_rf(explain=True)
         if 'svm' in model_use:
             fsn.run_svm(explain=True)
         if 'lgr' in model_use:
             fsn.run_lgr(explain=True)
         ### extract feature
         model_ext = ext.extractor(fsn.model_acc, 
                                   fsn.shap_value, model_acc_threshold)
         outline = model_ext.extract_outline_fea(z_score_cutoff=outlier_z_score_threshold,
                                                 quantile_cutoff=outlier_quantile)
         drop =  model_ext.extract_bottom_fea(0.1)
         indexes_to_drop = np.concatenate((outline, drop))
         z_score = model_ext.z_score
         z_score = z_score.drop(indexes_to_drop)
         total_score = model_ext.full_score.sum()
    
        
         top500_this_loop = adata.var_names[z_score.iloc[:100].index]
         top500_intersect = np.intersect1d(top500_this_loop,top500)
         overlap_num.append(len(top500_intersect))

         ### summary result
         if outline != 'No_outline':   
             outline_fea = list(fsn.feature_name[z_score[:len(outline)].index])
             outline_score = list(z_score[:len(outline)].values)
             outline_ite = [ite_num] * len(outline)
             outline_score_sum = z_score.iloc[:len(outline)].sum()
             outline_percent = [outline_score_sum/total_score] * len(outline)
             
         outline_fea_all = outline_fea_all+outline_fea
         outline_score_all = outline_score_all+outline_score
         outline_ite_all = outline_ite_all+outline_ite
         outline_percent_all = outline_percent_all+outline_percent
         print('overlap num: '+str(len(top500_intersect)))
         if outline == 'No_outline' and ite_num > 3 and len(top500_intersect)>=90:
             steady_state = True
         top500 = top500_this_loop
         ite_num += 1
         if ite_num!=1:
             adata_use = adata_use[:,~adata_use.var_names.isin(adata_use.var_names[indexes_to_drop])]
             
    print(overlap_num)

    outline_df = pd.DataFrame({'outline_fea':outline_fea_all,
                                'outline_score':outline_score_all,
                                'outline_ite':outline_ite_all,
                                'outline_percent':outline_percent_all})
    outline_df[['Source', 'Target']] = outline_df['outline_fea'].str.split('#', expand=True)
    return fsn,model_ext,outline_df


def cal_tf_score(grp_importance):
    all_propotion = []
    for i in set(list(grp_importance['outline_ite'])):
        propotion = grp_importance[grp_importance['outline_ite']==i]['outline_percent'].values[0]
        if i ==1:
            all_propotion.append(propotion)
        else:
            all_propotion.append((1-sum(all_propotion))*propotion)
            
    all_weight = []
    for i in range(grp_importance.shape[0]):
        index = grp_importance['outline_ite'][i]
        all_weight.append(all_propotion[index-1])
    grp_importance['weight'] = all_weight
    grp_importance['grp_score'] = grp_importance['outline_score']*grp_importance['weight']
    ### cal tf score
    sum_scores = grp_importance.groupby('Source')['grp_score'].sum()
    sum_scores_df = pd.DataFrame(sum_scores).reset_index()
    sorted_df = sum_scores_df.sort_values('grp_score', ascending=False)
    sorted_df.index = range(sorted_df.shape[0])
    return sorted_df