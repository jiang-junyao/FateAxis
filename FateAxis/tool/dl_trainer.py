# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 00:24:58 2024

@author: junyao
"""


import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="shap")
import torch.nn as nn
import torch.nn.functional as F
import shap
import numpy as np

class Error(Exception):
    pass



class torch_trainer():
    def __init__(self,
                 model,
                 device = 'gpu'):
        
        self.model = model
        if 'cuda' in device and torch.cuda.is_available():
            self.device = torch.device(device)
            #torch.set_default_tensor_type(torch.cuda.FloatTensor)
            #print('GPU lanuch!')
        else:
            self.device = torch.device('cpu')
            print('only using CPU to train DL model~~~')
        self.model = self.model.to(self.device)
        
    def fit(self,train_loader, total_epoch=4,transfer_data=True):

        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for epoch in range(total_epoch):
            
            #print('-----epoch:'+str(epoch))
            epoch_loss = 0
            
            for data, target in train_loader:
                if transfer_data:
                    data = data.unsqueeze(1)
                data = data.float()
                data = data.to(self.device)
                target = target.long()
                target = target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            scheduler.step()
            epoch_loss = epoch_loss/len(train_loader.dataset)
            #print('epoch loss:'+str(epoch_loss))
            
    def evaluate(self, test_loader,transfer_data=True):
        
        self.model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                if transfer_data:
                    data = data.unsqueeze(1)
                data = data.float()
                data = data.to(self.device)
                target = target.long()
                target = target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, 
                                             reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
        return test_loss / len(test_loader.dataset),correct / len(test_loader.dataset)
    
    def explain(self,shap_loader,X,transfer_data=True,device=None,
                acc=None,acc_thr=0.9):
        
        self.model.eval()
        correct_pred_index = []
        test_loss = 0
        with torch.no_grad():
            for data, target in shap_loader:
                if transfer_data:
                    data = data.unsqueeze(1)
                data = data.float()
                data = data.to(self.device)
                target = target.long()
                target = target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct_pred = pred.eq(target.view_as(pred))
                correct_pred_index.append(correct_pred)
        correct_pred_index = torch.cat(correct_pred_index).nonzero(as_tuple=True)[0].cpu().numpy()
        data_for_shap = X[correct_pred_index[:10]]
        data_for_shap = data_for_shap.to(torch.device('cpu'))
        test_loss = test_loss / len(shap_loader.dataset)
        if transfer_data:
            data_for_shap = data_for_shap.unsqueeze(1)
        self.model.to(torch.device('cpu'))
        if device == 'cpu':
            self.model.set_hidden_device(device)
        torch.cuda.empty_cache()
        explainer = shap.DeepExplainer(self.model, data_for_shap)
        shap_values = explainer.shap_values(data_for_shap,check_additivity=False)
        np.squeeze(np.abs(shap_values))
        score = np.abs(shap_values)
        score = (1-test_loss)*(self.__min_max_scaling(np.mean(np.mean(score,axis=0),axis=2)[0]))
        return score


            
    def __min_max_scaling(self,arr):

        min_val = np.min(arr)
        max_val = np.max(arr)
        scaled_arr = (arr - min_val) / (max_val - min_val)

        return scaled_arr