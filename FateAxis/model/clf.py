from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
import scanpy as sc
import warnings
import FateAxis.tool.io_tool as io_use
from sklearn.exceptions import ConvergenceWarning
import pkg_resources
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings('ignore')
import torch
from scipy.special import softmax
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader

### DL module
from FateAxis.tool import dl_trainer
from FateAxis.model import cnn_1d
from FateAxis.model import cnn_hybrid
from FateAxis.model import gru
from FateAxis.model import lstm
from FateAxis.model import rnn

class MyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)
    
class classification:
    def __init__(self,
                 input_mt,
                 label,
                 config_path=None,
                 split_size = 0.3,
                 dl_epoch = 4,
                 device='gpu',
                 feature_name = [],
                 acc_cut = 0.9,
                 core_num = 30):
        le = preprocessing.LabelEncoder()
        self.org_label = label
        self.acc_cut = acc_cut
        self.label = le.fit_transform(label)
        self.input_mt = input_mt
        self.dl_epoch = dl_epoch
        self.device = device
        self.core_num = core_num
        self.feature_name = feature_name
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(input_mt, 
                                                                                self.label,
                                                                                test_size=split_size, 
                                                                                random_state=114514)
        self.model_acc = {}
        self.model_loss = {}
        self.shap_value = {}
        self.pred = {}
        if config_path==None:
            config_path = pkg_resources.resource_filename('FateAxis', 
                                                          'config/config1.js')
        self.config = io_use.read_json(config_path)
        
        # dataset used for torch
        self.train_data = MyDataset(self.x_train, self.y_train)
        self.test_data = MyDataset(self.x_test, self.y_test)
        
        # data for explain
        

    
    ### xgb
    def run_gbm(self, explain = True,para=None):
        
        for config_use in self.config['GBM'].keys():
            
            try:
                para = self.config['GBM'][config_use]
                self.gbm = XGBClassifier(**para['config'])
                self.gbm.fit(self.x_train, self.y_train)
        
                gbm_test_acc = accuracy_score(self.y_test, self.gbm.predict(self.x_test))
                self.model_acc[config_use] = gbm_test_acc
                loss = torch.nn.CrossEntropyLoss()(torch.tensor(self.gbm.predict_proba(self.x_test)),
                                                       torch.tensor(self.y_test,dtype=torch.long))
                
                self.model_loss[config_use] = loss
                if explain:
                    pred = self.gbm.predict(self.input_mt)
                    idx = pred==self.label
                    explainer_gbm = shap.TreeExplainer(self.gbm)
                    shap_vals = explainer_gbm.shap_values(self.input_mt[idx])
                    self.shap_value[config_use] = (1-loss)*(self.__min_max_scaling(np.abs(shap_vals).mean(axis=0)))
                    
            except Exception:
                pass
            
    ### linear SVM
    def run_svm(self, explain = True,para=None):

        for config_use in self.config['SVM'].keys():
            
            para = self.config['SVM'][config_use]
            self.svm = SVC(**para['config'])
            self.svm.fit(self.x_train, self.y_train)
    
            pred = self.svm.predict_proba(self.x_test)
            svm_test_acc = accuracy_score(self.y_test,self.svm.predict(self.x_test))
            self.model_acc[config_use] = svm_test_acc
            loss = torch.nn.CrossEntropyLoss()(torch.tensor(pred),
                                                   torch.tensor(self.y_test,dtype=torch.long))
            self.model_loss[config_use] = loss
            if explain:
                ### evaluate
                self.shap_value[config_use] = (1-loss)*(self.__min_max_scaling(np.abs(self.svm.coef_[0])))
            
    ### Logistic Regression
    def run_lgr(self, explain = True):

        for config_use in self.config['logistic regression'].keys():
            
            para = self.config['logistic regression'][config_use]
            self.lgr = LogisticRegression(**para['config'])
            self.lgr.fit(self.x_train, self.y_train)
    
            test_acc = accuracy_score(self.y_test, self.lgr.predict(self.x_test))
            self.model_acc[config_use] = test_acc
            loss = torch.nn.CrossEntropyLoss()(torch.tensor(self.lgr.predict_proba(self.x_test)),
                                                   torch.tensor(self.y_test,dtype=torch.long))
            self.model_loss[config_use] = loss
            if explain:
                self.shap_value[config_use] = (1-loss)*(self.__min_max_scaling(np.abs(self.lgr.coef_[0])))
            
    ### Random Forest
    def run_rf(self, explain = True,para=None):

        for config_use in self.config['RFC'].keys():
            
            para = self.config['RFC'][config_use]
            self.rf = RandomForestClassifier(**para['config'])
            self.rf.fit(self.x_train, self.y_train)
    
            test_acc = accuracy_score(self.y_test, self.rf.predict(self.x_test))
            loss = torch.nn.CrossEntropyLoss()(torch.tensor(self.rf.predict_proba(self.x_test)),
                                                   torch.tensor(self.y_test,dtype=torch.long))
            self.model_acc[config_use] = test_acc
            self.model_loss[config_use] = loss
            if test_acc >= self.acc_cut:
                if explain:
                    pred = self.rf.predict(self.input_mt)
                    idx = pred==self.label
                    explainer_rf = shap.TreeExplainer(self.rf)
                    shap_val = explainer_rf.shap_values(self.input_mt[idx])
                    if len(shap_val)>2:
                        shap_val = np.array(shap_val).sum(axis=2)
                    self.shap_value[config_use] = (1-loss)*(self.__min_max_scaling(np.abs(shap_val).mean(axis=0)))
                
    ### 1D CNN
    def run_cnn1d(self,explain=True):
        

        
        for config_use in self.config['CNN_1D'].keys():
         
            if self.config['CNN_1D'][config_use]['config']['num_layers'] < 3:
                model = cnn_1d.Limited(config_use, 
                                self.config['CNN_1D'][config_use]['config'])
            else:
                model = cnn_1d.Unlimited(config_use, 
                                self.config['CNN_1D'][config_use]['config'])

            batch_size = self.config['CNN_1D'][config_use]['batch_size']
            
            train_loader = DataLoader(self.train_data, batch_size=batch_size, 
                                      shuffle=True)
            test_loader = DataLoader(self.test_data, batch_size=batch_size, 
                                     shuffle=False)

            Trainer = dl_trainer.torch_trainer(model,device=self.device)
            Trainer.fit(train_loader,self.dl_epoch)
            loss,acc = Trainer.evaluate(test_loader)
            self.model_acc[config_use] = acc
            self.model_loss[config_use] = loss
            torch.cuda.empty_cache()
            if acc >= self.acc_cut:
                if explain:
                    X = torch.from_numpy(self.input_mt).float()
                    y = torch.from_numpy(self.label).long()
                    data = MyDataset(X, y)
                    shap_loader = DataLoader(data, batch_size=1, shuffle=False)
                    shap_score = Trainer.explain(shap_loader,X,
                                                 acc=acc,acc_thr=self.acc_cut)
                    self.shap_value[config_use] = shap_score
                    
    def run_hybrid(self,explain=True):
        

        
        for config_use in self.config['CNN_Hybrid'].keys():
            print(config_use)
            model = cnn_hybrid.Limited(config_use, 
                                self.config['CNN_Hybrid'][config_use]['config'])


            batch_size = self.config['CNN_Hybrid'][config_use]['batch_size']
            
            train_loader = DataLoader(self.train_data, batch_size=batch_size, 
                                      shuffle=True)
            test_loader = DataLoader(self.test_data, batch_size=batch_size, 
                                     shuffle=False)

            Trainer = dl_trainer.torch_trainer(model,device=self.device)
            Trainer.fit(train_loader,self.dl_epoch,transfer_data=False)
            loss,acc = Trainer.evaluate(test_loader)
            self.model_acc[config_use] = acc
            self.model_loss[config_use] = loss
            print('acc:'+str(acc))
            print('loss:'+str(loss))
            if explain:
                X = torch.from_numpy(self.input_mt).float()
                y = torch.from_numpy(self.label).long()
                data = MyDataset(X, y)
                shap_loader = DataLoader(data, batch_size=1, shuffle=False)
                shap_score = Trainer.explain(shap_loader,X,acc,self.acc_cut)
                self.shap_value[config_use] = shap_score
                
    def run_gru(self,explain=True):
        

        for config_use in self.config['GRU'].keys():
            n_fea = self.input_mt.shape[1]
            model = gru.GRU(self.device,config_use, n_fea,
                                **self.config['GRU'][config_use]['config'])
            model.set_hidden_device('gpu')
            
            batch_size = self.config['GRU'][config_use]['batch_size']
            
            train_loader = DataLoader(self.train_data, batch_size=batch_size, 
                                      shuffle=True)
            test_loader = DataLoader(self.test_data, batch_size=batch_size, 
                                     shuffle=False)

            Trainer = dl_trainer.torch_trainer(model,device=self.device)
            Trainer.fit(train_loader,self.dl_epoch,transfer_data=True)
            loss,acc = Trainer.evaluate(test_loader)
            self.model_acc[config_use] = acc
            self.model_loss[config_use] = loss
            if acc >= self.acc_cut:
                if explain:
                    X = torch.from_numpy(self.input_mt).float()
                    y = torch.from_numpy(self.label).long()
                    data = MyDataset(X, y)
                    shap_loader = DataLoader(data, batch_size=1, shuffle=False)
                    shap_score = Trainer.explain(shap_loader,X,device='cpu')
                    self.shap_value[config_use] = shap_score

    def run_lstm(self,explain=True):


        for config_use in self.config['LSTM'].keys():

            n_fea = self.input_mt.shape[1]
            model = lstm.LSTM(self.device,config_use,n_fea,
                                **self.config['LSTM'][config_use]['config'])
            model.set_hidden_device('gpu')
            batch_size = self.config['LSTM'][config_use]['batch_size']
            
            train_loader = DataLoader(self.train_data, batch_size=batch_size, 
                                      shuffle=True)
            test_loader = DataLoader(self.test_data, batch_size=batch_size, 
                                     shuffle=False)

            Trainer = dl_trainer.torch_trainer(model,device=self.device)
            Trainer.fit(train_loader,self.dl_epoch,transfer_data=True)
            loss,acc = Trainer.evaluate(test_loader)
            self.model_acc[config_use] = acc
            self.model_loss[config_use] = loss
            if acc >= self.acc_cut:
                if explain:
                    X = torch.from_numpy(self.input_mt).float()
                    y = torch.from_numpy(self.label).long()
                    data = MyDataset(X, y)
                    shap_loader = DataLoader(data, batch_size=1, shuffle=False)
                    shap_score = Trainer.explain(shap_loader,X,device='cpu')
                    self.shap_value[config_use] = shap_score
                
    def run_rnn(self,explain=True):
        

        for config_use in self.config['RNN'].keys():

            n_fea = self.input_mt.shape[1]
            model = rnn.RNN(self.device,config_use,n_fea,
                                **self.config['RNN'][config_use]['config'])
            model.set_hidden_device('gpu')
            batch_size = self.config['RNN'][config_use]['batch_size']
            
            train_loader = DataLoader(self.train_data, batch_size=batch_size, 
                                      shuffle=True)
            test_loader = DataLoader(self.test_data, batch_size=batch_size, 
                                     shuffle=False)

            Trainer = dl_trainer.torch_trainer(model,device=self.device)
            Trainer.fit(train_loader,self.dl_epoch,transfer_data=True)
            loss,acc = Trainer.evaluate(test_loader)
            self.model_acc[config_use] = acc
            self.model_loss[config_use] = loss
            if acc >= self.acc_cut:
                if explain:
                    X = torch.from_numpy(self.input_mt).float()
                    y = torch.from_numpy(self.label).long()
                    data = MyDataset(X, y)
                    shap_loader = DataLoader(data, batch_size=1, shuffle=False)
                    shap_score = Trainer.explain(shap_loader,X,device='cpu')
                    self.shap_value[config_use] = shap_score     
                
    def __min_max_scaling(self,arr):

        min_val = np.min(arr)
        max_val = np.max(arr)
        scaled_arr = (arr - min_val) / (max_val - min_val)

        return scaled_arr
