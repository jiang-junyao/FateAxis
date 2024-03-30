#!/usr/bin/env python3

import io
import sys
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
            device = torch.device('cuda')
            print('GPU lanuch!')
        else:
            device = torch.device('cpu')
            print('only using CPU to train DL model~~~')

        
    def fit(self,train_loader, total_epoch=4):
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(total_epoch):
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.squeeze()
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
    def evaluate(self, test_loader):
        
        self.model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, 
                                             reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
        return test_loss,correct / len(test_loader.dataset)
    
    def explain(self,shap_loader,X):
        
        self.model.eval()
        correct_pred_index = []
        test_loss = 0
        
        with torch.no_grad():
            for i, (data, target) in enumerate(shap_loader):
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct_pred = pred.eq(target.view_as(pred))
                correct_pred_index.append(correct_pred)
        data_for_shap = X[correct_pred_index[:50]]
        correct_pred_index = torch.cat(correct_pred_index).nonzero(as_tuple=True)[0].numpy()
        explainer = shap.DeepExplainer(self.model, data_for_shap)
        shap_values = explainer.shap_values(data_for_shap)
        score = test_loss*(self.__min_max_scaling(np.abs(shap_values)))
        
        return score
