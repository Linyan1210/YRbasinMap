# -*- coding: utf-8 -*-
"""
This code is used to select optimal hyperparameter combinations for each RF variant in the cross-validation.
You need to change the input data of each RF variant.
@author: Linyan, Zhang
"""
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from pdpbox import pdp
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from collections import Counter
from imblearn.over_sampling import BorderlineSMOTE
from openpyxl import Workbook
from numpy import mean
from numpy import std
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score, accuracy_score, confusion_matrix, cohen_kappa_score
import re

data=pd.read_excel('E:\\NestPaper\\Remodifying_paper\\Python_code\\RF_all_variant.xlsx','Sheet1') # changing the input data of each variant
y=data.targets_2
x=data.iloc[:,3:28]
#################### three times three folds cross-validation
rekfold = RepeatedKFold(n_splits=3, n_repeats=3, random_state=42) # n_splits is the number of folds
model = RandomForestClassifier(random_state=12, n_estimators=500, class_weight='balanced')
#################### original hyperparameters setting
param_grid = {
    'max_features': ['sqrt','log2'] + list(range(2,10,2)), 
    'max_samples': [0.2, 0.4, 0.6, 0.8, 1],
    'min_samples_split': list(range(15, 30, 3)),
}
gsearch = GridSearchCV(model, param_grid, scoring='balanced_accuracy', cv=rekfold, n_jobs=10)
cd = gsearch.fit(x, y)
print(cd.best_params_)
#################### extracting all test data after three times three folds
all_true_test_labels = []
all_proba_class_test = []
all_predicted_test_labels = []
all_index =[]
bacc_scores = []
auc_scores = []
for i, (train_index, test_index) in enumerate(rekfold.split(x)):
     x_train, x_test = x.iloc[train_index], x.iloc[test_index]  
     y_train, y_test = y.iloc[train_index], y.iloc[test_index]     
     best_model = cd.best_estimator_
     y_proba_test = best_model.predict_proba(x_test)[:, 1]
     predicted_test_labels = (y_proba_test > 0.5).astype(int)   
#################### Evaluate the model's performance
     index_values = y_test.index
     all_index.extend(index_values)
     all_true_test_labels.extend(y_test)
     all_proba_class_test.extend(y_proba_test)
     all_predicted_test_labels.extend(predicted_test_labels)
     bacc = balanced_accuracy_score(y_test, predicted_test_labels)
     bacc_scores.append(bacc)  
     Auc = roc_auc_score(y_test, y_proba_test) 
     auc_scores.append(Auc)
#################### Calculate the average BACC across nine rounds
average_bacc = sum(bacc_scores) / len(bacc_scores)
print("Average bacc:", average_bacc)     
average_auc = sum(auc_scores) / len(auc_scores)
print("Average auc:", average_auc)