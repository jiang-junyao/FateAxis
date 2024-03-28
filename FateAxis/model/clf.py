from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
from sklearn.exceptions import ConvergenceWarning

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
    ### xgb
    def run_gbm(self, explain = True,para=None):
        print('---runing xgboost---')
        self.gbm = XGBClassifier()
        grid_search = GridSearchCV(self.gbm, param_grid=para, cv=3, scoring='accuracy', n_jobs=self.core_num)
        grid_search.fit(self.x_train, self.y_train)
        self.gbm_best_params = grid_search.best_params_
        print(self.gbm_best_params)
        self.gbm = XGBClassifier(**self.gbm_best_params)
        self.gbm.fit(self.x_train, self.y_train)

        gbm_test_acc = accuracy_score(self.y_test, self.gbm.predict(self.x_test))
        self.model_acc['xgboost'] = gbm_test_acc
        loss = torch.nn.CrossEntropyLoss()(torch.tensor(self.gbm.predict_proba(self.x_test)),
                                               torch.tensor(self.y_test))
        
        self.model_loss['xgboost'] = loss
        print('acc:'+str(gbm_test_acc))
        print('loss:'+str(loss))
        if explain:
            pred = self.gbm.predict(self.input_mt)
            idx = pred==self.label
            explainer_gbm = shap.TreeExplainer(self.gbm,feature_perturbation = 'interventional',data = self.input_mt[idx])
            shap_vals = explainer_gbm.shap_values(self.input_mt[idx])
            self.shap_value['xgboost'] = loss*(self.__min_max_scaling(np.abs(shap_vals).mean(axis=0)))
            
    ### linear SVM
    def run_svm(self, explain = True,para=None):
        print('---runing svm---')
        self.svm = SVC(probability=True)
        grid_search = GridSearchCV(self.svm, param_grid=para, cv=3, scoring='accuracy', n_jobs=self.core_num)
        grid_search.fit(self.x_train, self.y_train)
        self.svm_best_params = grid_search.best_params_
        print(self.svm_best_params)
        self.svm = SVC(probability=True,**self.svm_best_params)
        self.svm.fit(self.x_train, self.y_train)

        pred = self.svm.predict_proba(self.x_test)
        svm_test_acc = accuracy_score(self.y_test,self.svm.predict(self.x_test))
        self.model_acc['svm'] = svm_test_acc
        loss = torch.nn.CrossEntropyLoss()(torch.tensor(pred),
                                               torch.tensor(self.y_test))
        self.model_loss['svm'] = loss
        print('acc:'+str(svm_test_acc))
        print('loss:'+str(loss))
        if explain:
            ### evaluate
            self.shap_value['svm'] = loss*(self.__min_max_scaling(np.abs(self.svm.coef_[0])))
            
    ### Logistic Regression
    def run_lgr(self, explain = True,para=None):
        print('---runing logistic regression---')
        self.lgr = LogisticRegression()
        grid_search = GridSearchCV(self.lgr, param_grid=para, cv=3, scoring='accuracy', n_jobs=self.core_num)
        grid_search.fit(self.x_train, self.y_train)
        self.lgr_best_params = grid_search.best_params_
        print(self.lgr_best_params)
        self.lgr = LogisticRegression(**self.lgr_best_params)
        self.lgr.fit(self.x_train, self.y_train)

        test_acc = accuracy_score(self.y_test, self.lgr.predict(self.x_test))
        self.model_acc['logistic regression'] = test_acc
        loss = torch.nn.CrossEntropyLoss()(torch.tensor(self.lgr.predict_proba(self.x_test)),
                                               torch.tensor(self.y_test))
        self.model_loss['logistic regression'] = loss
        print('acc:'+str(test_acc))
        print('loss:'+str(loss))
        if explain:
            self.shap_value['logistic regression'] = loss*(self.__min_max_scaling(np.abs(self.lgr.coef_[0])))
            
    ### Random Forest
    def run_rf(self, explain = True,para=None):
        print('---runing Random forest---')
        self.rf = RandomForestClassifier()
        grid_search = GridSearchCV(self.rf, param_grid=para, cv=3, scoring='accuracy', n_jobs=self.core_num)
        grid_search.fit(self.x_train, self.y_train)
        self.rf_best_params = grid_search.best_params_
        print(self.rf_best_params)
        self.rf = RandomForestClassifier(**self.rf_best_params)
        self.rf.fit(self.x_train, self.y_train)

        test_acc = accuracy_score(self.y_test, self.rf.predict(self.x_test))
        loss = torch.nn.CrossEntropyLoss()(torch.tensor(self.rf.predict_proba(self.x_test)),
                                               torch.tensor(self.y_test))
        self.model_acc['random forest'] = test_acc
        self.model_loss['random forest'] = loss
        print(test_acc)
        if explain:
            pred = self.rf.predict(self.input_mt)
            idx = pred==self.label
            explainer_rf = shap.TreeExplainer(self.rf)
            shap_val = explainer_rf.shap_values(self.input_mt[idx])
            if len(shap_val)>2:
                shap_val = np.array(shap_val).sum(axis=2)
            self.shap_value['random forest'] = loss*(self.__min_max_scaling(shap_val.mean(axis=0)))

    ### DT
    def run_dt(self, explain = True,para=None):
        print('---runing Decision Tree---')
        self.dt = DecisionTreeClassifier()
        grid_search = GridSearchCV(self.dt, param_grid=para, cv=3, scoring='accuracy', n_jobs=self.core_num)
        grid_search.fit(self.x_train, self.y_train)
        self.dt_best_params = grid_search.best_params_
        print(self.dt_best_params)
        self.dt = DecisionTreeClassifier(**self.dt_best_params)
        self.dt.fit(self.x_train, self.y_train)

        gbm_test_acc = accuracy_score(self.y_test, self.dt.predict(self.x_test))
        self.model_acc['decision tree'] = gbm_test_acc
        print(gbm_test_acc)
        if explain:
            pred = self.dt.predict(self.input_mt)
            idx = pred==self.label
            explainer_dt = shap.TreeExplainer(self.dt)
            normalized_importance = softmax(np.abs(explainer_dt.shap_values(self.input_mt[idx])).mean(0))
            if len(normalized_importance.shape)>1:
                normalized_importance = np.sum(normalized_importance,axis=1)
            self.shap_value['decision tree'] = normalized_importance

