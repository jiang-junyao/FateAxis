
import numpy as np
import pandas as pd
import sys

class extractor:
    def __init__(self,
                 model_acc,
                 shap_value,
                 acc_cutoff,):

        ### get survive model
        self.survive_model = []
        for key,value in model_acc.items():
            if value >= acc_cutoff:
                self.survive_model.append(key)
        if self.survive_model == []:
            print('no model pass the acc threshold please decrease acc threshold')
            sys.exit()
        ### cal full score
        self.full_score = None
        for i in self.survive_model:
            model_score = np.array(shap_value[i])
            if not np.isnan(model_score.mean()):
                if self.full_score is None:
                    self.full_score = model_score
                else:
                    self.full_score = self.full_score + model_score
        self.full_score = pd.Series(self.full_score, 
                                    index=np.arange(len(self.full_score)))
        self.full_score = self.full_score.sort_values(ascending=False)
        ### z_score normalization
        
        z_score = (self.full_score - np.mean(self.full_score)) / \
            np.std(self.full_score)
        self.z_score = pd.Series(z_score, index=np.arange(len(z_score)))
        self.z_score = self.z_score.sort_values(ascending=False)
            
    def extract_outline_fea(self,z_score_cutoff = 5, outline_foldchange = 3,
                            quantile_cutoff = 95):
        quantile_score = np.percentile(self.z_score, quantile_cutoff)*3
        outline_fea = []
        for i in range(1, len(self.z_score)-1):
            current_value = self.z_score[i]
            pre_value_cutoff = self.z_score[i-1]/outline_foldchange 
            cutoff = max(quantile_score,pre_value_cutoff,z_score_cutoff)
            if current_value > cutoff:
                outline_fea.append(self.z_score.index[i])
        if outline_fea == []:
            return 'No_outline'
        else:
            return outline_fea
        
    def extract_bottom_fea(self,bottom_percent=0.1):
        
        num_percent = int(len(self.z_score) * bottom_percent)
        last_percent_indices = self.z_score.iloc[-num_percent:].index

        return last_percent_indices
