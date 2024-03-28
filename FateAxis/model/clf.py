from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import shap
import scanpy as sc
import warnings
import FateAxis.tool.io_tool as io_use
from sklearn.exceptions import ConvergenceWarning
import pkg_resources
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import torch
from scipy.special import softmax
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

class classification:
    def __init__(self,
                 input_mt,
                 label,
                 split_size = 0.3,
                 core_num = 30):
        le = preprocessing.LabelEncoder()
        self.org_label = label
        self.label = le.fit_transform(label)
        self.input_mt = input_mt
        self.core_num = core_num
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(input_mt, 
                                                                                self.label,
                                                                                test_size=split_size, 
                                                                                random_state=114514)
        self.model_acc = {}
        self.model_loss = {}
        self.shap_value = {}
        self.pred = {}
        config_path = pkg_resources.resource_filename('FateAxis', 
                                                      'config/config1.js')
        self.config = io_use.read_json(config_path)
    ### xgb
    def run_gbm(self, explain = True,para=None):
        print('---runing xgboost---')
        
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
                print('acc:'+str(gbm_test_acc))
                print('loss:'+str(loss))
                if explain:
                    pred = self.gbm.predict(self.input_mt)
                    idx = pred==self.label
                    explainer_gbm = shap.TreeExplainer(self.gbm,feature_perturbation = 'interventional',
                                                       data = self.input_mt[idx])
                    shap_vals = explainer_gbm.shap_values(self.input_mt[idx])
                    self.shap_value[config_use] = loss*(self.__min_max_scaling(np.abs(shap_vals).mean(axis=0)))
                    
            except Exception:
                pass
            
    ### linear SVM
    def run_svm(self, explain = True,para=None):
        print('---runing svm---')
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
            print('acc:'+str(svm_test_acc))
            print('loss:'+str(loss))
            if explain:
                ### evaluate
                self.shap_value[config_use] = loss*(self.__min_max_scaling(np.abs(self.svm.coef_[0])))
            
    ### Logistic Regression
    def run_lgr(self, explain = True):
        print('---runing logistic regression---')
        for config_use in self.config['logistic regression'].keys():
            
            para = self.config['logistic regression'][config_use]
            self.lgr = LogisticRegression(**para['config'])
            self.lgr.fit(self.x_train, self.y_train)
    
            test_acc = accuracy_score(self.y_test, self.lgr.predict(self.x_test))
            self.model_acc[config_use] = test_acc
            loss = torch.nn.CrossEntropyLoss()(torch.tensor(self.lgr.predict_proba(self.x_test)),
                                                   torch.tensor(self.y_test,dtype=torch.long))
            self.model_loss[config_use] = loss
            print('acc:'+str(test_acc))
            print('loss:'+str(loss))
            if explain:
                self.shap_value[config_use] = loss*(self.__min_max_scaling(np.abs(self.lgr.coef_[0])))
            
    ### Random Forest
    def run_rf(self, explain = True,para=None):
        print('---runing Random forest---')
        for config_use in self.config['RFC'].keys():
            
            para = self.config['RFC'][config_use]
            self.rf = RandomForestClassifier(**para['config'])
            self.rf.fit(self.x_train, self.y_train)
    
            test_acc = accuracy_score(self.y_test, self.rf.predict(self.x_test))
            loss = torch.nn.CrossEntropyLoss()(torch.tensor(self.rf.predict_proba(self.x_test)),
                                                   torch.tensor(self.y_test,dtype=torch.long))
            self.model_acc[config_use] = test_acc
            self.model_loss[config_use] = loss
            print(test_acc)
            if explain:
                pred = self.rf.predict(self.input_mt)
                idx = pred==self.label
                explainer_rf = shap.TreeExplainer(self.rf)
                shap_val = explainer_rf.shap_values(self.input_mt[idx])
                if len(shap_val)>2:
                    shap_val = np.array(shap_val).sum(axis=2)
                self.shap_value[config_use] = loss*(self.__min_max_scaling(shap_val.mean(axis=0)))

        
    def __min_max_scaling(self,arr):

        min_val = np.min(arr)
        max_val = np.max(arr)
        scaled_arr = (arr - min_val) / (max_val - min_val)

        return scaled_arr
