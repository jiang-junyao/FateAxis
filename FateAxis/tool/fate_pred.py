# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 13:58:56 2025

@author: peiweike
"""

import warnings
warnings.filterwarnings('ignore')
import FateAxis.model.clf as clf
import scanpy as sc
import numpy as np
import pandas as pd
import FateAxis.tool.preditor as preditor

class fateaxis:
    def __init__(self,
                 mtx,
                 label,
                 config_path = None,
                 model_use = ['cnn','gru','lstm','rnn','gbm'
                             ,'rf','svm','lgr'],
                 core_use = 10,
                 dl_epoch=5,
                 model_out_dir = None,
                 device='cuda:0'):
        self.mtx = mtx
        self.label = label
        self.model_use = model_use
        self.model_out_dir = model_out_dir
        self.fsn = clf.classification(mtx, label, config_path,dl_epoch=dl_epoch,
                                          model_outdir=model_out_dir,
                                          core_num=core_use,
                                          device = device)
        self.weighted_probabilities = 'no weight'
        
    def run_clf(self):
        if 'cnn' in self.model_use:
            self.fsn.run_cnn1d(explain=False)
        if 'gru' in self.model_use:
            self.fsn.run_gru(explain=False)
        if 'lstm' in self.model_use:
            self.fsn.run_lstm(explain=False)
        if 'rnn' in self.model_use:
            self.fsn.run_rnn(explain=False)
        if 'gbm' in self.model_use:
            self.fsn.run_gbm(explain=False)
        if 'rf' in self.model_use:
            self.fsn.run_rf(explain=False)
        if 'svm' in self.model_use:
            self.fsn.run_svm(explain=False)
        if 'lgr' in self.model_use:
            self.fsn.run_lgr(explain=False)
        


    def predict(self,mtx,model_acc_cut=0.95):

        weights = preditor.get_model_weights(self.fsn)
        model_name, model_probabilities,label = preditor.get_model_predictions(self.fsn, mtx,
                                                      model_acc_cutoff=model_acc_cut)
        prob_dict = {}
        for name, prob in zip(model_name, model_probabilities):
            prob_dict[name] = prob
            
        label_dict = {}
        for name, label1 in zip(model_name, label):
            label_dict[name] = label1
            
        weighted_probabilities = sum(
            weight * prob for weight, prob in zip(weights, model_probabilities)
        )
        self.weighted_probabilities = weighted_probabilities
        return prob_dict,label_dict
    
    def __normalize_label(self,label):

        parts = label.split('+')
        parts.sort()
        return '+'.join(parts)
    
    def ensemble_predict_with_fuzzy_labels(self, test_loader, method='fixed', 
                                           fixed_threshold=0.1, 
                              alpha=0.5, relative_threshold=0.8, 
                              beta=0.5,model_acc_cutoff = 0.95):
    


        ### generate fuzzy labels
        if self.weighted_probabilities == 'no weight':
            print('Please run predict first!')
        else:
            
            fuzzy_labels_fixed = preditor.generate_fuzzy_labels(self.weighted_probabilities, 
                                                   method=method, 
                                                   threshold=fixed_threshold,
                                                   alpha=alpha,
                                                   relative_threshold=relative_threshold,
                                                   beta=beta)
            normalized_labels = [self.__normalize_label(lbl) for lbl in fuzzy_labels_fixed]
            
            return normalized_labels
