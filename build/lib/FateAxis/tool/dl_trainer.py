# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 00:24:58 2024

@author: junyao
"""


import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
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
        if device=='gpu' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            #torch.set_default_tensor_type(torch.cuda.FloatTensor)
            #print('GPU lanuch!')
        else:
            self.device = torch.device('cpu')
            #print('only using CPU to train DL model~~~')
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
                data = data.to(self.device)
                target = target.long()
                target = target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, 
                                             reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
        return test_loss / len(test_loader.dataset),correct / len(test_loader.dataset)
    
    def explain(self,shap_loader,X,transfer_data=True,device=None):
        
        self.model.eval()
        correct_pred_index = []
        test_loss = 0
        with torch.no_grad():
            for data, target in shap_loader:
                if transfer_data:
                    data = data.unsqueeze(1)
                
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
        if transfer_data:
            data_for_shap = data_for_shap.unsqueeze(1)
        self.model.to(torch.device('cpu'))
        if device == 'cpu':
            self.model.set_hidden_device(device)
        explainer = shap.DeepExplainer(self.model, data_for_shap)
        shap_values = explainer.shap_values(data_for_shap)
        score = test_loss*(self.__min_max_scaling(np.abs(shap_values)))
        
        return score

            
    def __min_max_scaling(self,arr):

        min_val = np.min(arr)
        max_val = np.max(arr)
        scaled_arr = (arr - min_val) / (max_val - min_val)

        return scaled_arr