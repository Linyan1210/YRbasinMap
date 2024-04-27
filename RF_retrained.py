# -*- coding: utf-8 -*-
"""
This code is used to retrain the four RF model variants based on their optimal hyperparameter combinations and 
generate the predictors importance and partial dependence plots. 
You need to change the input data of each RF variant.
@author: Linyan Zhang
"""
import numpy as np
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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
from sklearn.inspection import partial_dependence
from openpyxl import Workbook
from numpy import mean
from numpy import std
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score, accuracy_score, confusion_matrix, cohen_kappa_score
################ Data input
data=pd.read_excel('E:\\NestPaper\\Remodifying_paper\\QHI\\RF_all_variant.xlsx','Sheet1') # changing the input data
y=data.targets_2
x=data.iloc[:,3:28]
################ The following is the optimal hyperparameter combinations of four RF model variant, please correspond to the input data
################ all_variant
opmodel = RandomForestClassifier(class_weight='balanced',
                                 n_estimators=500,
                                 max_features='sqrt',
                                 max_samples=0.8,                    
                                 min_samples_split=18
                                )  
################ noQ_variant
opmodel = RandomForestClassifier(class_weight='balanced',
                                 n_estimators=500,
                                 max_features=2,
                                 max_samples=0.8,                      
                                 min_samples_split=21
                                )  
################ noHI_variant
opmodel = RandomForestClassifier(class_weight='balanced',
                                 n_estimators=500,
                                 max_features='sqrt',
                                 max_samples=0.6,                      
                                 min_samples_split=27
                                )  
################ QHI_variant
opmodel = RandomForestClassifier(class_weight='balanced',
                                 n_estimators=500,
                                 max_features=6,
                                 max_samples=0.8,                      
                                 min_samples_split=24
                                )   
opmodel.fit(x, y)
################ generating BACC and AUC of retrained RF model
ypred224_retrained = opmodel.predict(x) 
proba224_retrained = opmodel.predict_proba(x)[:,1]
Auc=roc_auc_score(y, opmodel.predict_proba(x)[:,1]) 
print("roc_auc_score:", Auc)
bacc = balanced_accuracy_score(y, ypred224_retrained)
print("Balanced Accuracy:", bacc)
#################  Applying the model to each reaches to predict the POSI, please generate the predictors for each reaches 
reaches=pd.read_excel('E:\\NestPaper\\Remodifying_paper\\Python_code\\RF_all_variant_reaches.xlsx','reaches_38871')
reaches_38871=reaches.iloc[:,5::]
ypred_38871 = opmodel.predict(reaches_38871)  # predict classes value
proba_38871 = opmodel.predict_proba(reaches_38871)[:,1] # probability of each classes 

#################  Generating predictors importance
print ('Importance of each target: %s' % opmodel.feature_importances_)
importances = opmodel.feature_importances_
indices = np.argsort(importances)
features = x.columns
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='g', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance (%)')
plt.show()

#################  Generating partial dependence plots
feature_names = x.columns.tolist()  # Assuming x is your DataFrame
# Loop over each predictor,plot one by one, because x range is different for each feature.
for i, feature_name in enumerate(feature_names):
    # Replace special characters in the feature name with underscores
    # safe_feature_name = re.sub(r'[^\w\s]', '_', feature_name)
    # Calculate partial dependence for the current predictor
    pdp_results = partial_dependence(opmodel, X=x, features=[i])
    grid_values = pdp_results['grid_values'][0]
    partial_dependence_values = pdp_results['average'][0]
    # x_min = x.iloc[:, i].min()
    # x_max = x.iloc[:, i].max()
    # Plot partial dependence for the current predictor
    plt.figure(figsize=(8, 6))
    plt.plot(grid_values, partial_dependence_values)    
    # Add rug plot
    plt.plot(x.iloc[:, i], [0.36] * len(x), '|', color='black', markersize=10, label='Rug Plot')      
    # plt.xlim(45, 70)
    plt.ylim(0.36, 0.5)     
    plt.xlabel('Feature Values')
    plt.ylabel('Partial Dependence')
    plt.title(f'Partial Dependence Plot for {feature_name}')
    # Save plot as PNG
    plt.savefig(f'E:\\NestPaper\\Remodifying_paper\\Figures\\PDP\\{feature_name}.png', dpi=300)
    # Close the current plot to free memory
    plt.close()