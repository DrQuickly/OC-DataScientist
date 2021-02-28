#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 23:46:38 2021

@author: famien
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import make_scorer,f1_score

from imblearn.over_sampling import SMOTE
#from imblearn.under_sampling import RandomUnderSampler
#from imblearn.pipeline import Pipeline


app_train = pd.read_csv('data/app_train.csv',index_col=0)
#app_test = pd.read_csv('data/app_test.csv',index_col=0)

app_train.set_index('SK_ID_CURR',inplace=True)
#app_test.set_index('SK_ID_CURR',inplace=True)


y_train = app_train['TARGET']
X_train = app_train.drop(['TARGET'],axis=1)

#over = SMOTE(sampling_strategy=0.1)
#under = RandomUnderSampler(sampling_strategy=0.5)
#steps = [('o', over), ('u', under)]
#pipeline = Pipeline(steps=steps)

#X_train_new, y_train_new = pipeline.fit_resample(X_train, y_train)

over = SMOTE(random_state=42)
X_train_new, y_train_new = over.fit_resample(X_train, y_train)

del y_train,X_train

fig,ax = plt.subplots(figsize=(20, 10), dpi= 80, facecolor='w', edgecolor='k')

def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%".format(pct,absolute)

colors = ['red', 'blue']

wedges, texts, autotexts = plt.pie(y_train_new.value_counts(),
                                   autopct=lambda pct: func(pct, y_train_new.value_counts())
                                   ,colors=colors,textprops=dict(color="w"))

ax.legend(wedges, y_train_new.value_counts().index,
          title="TARGET",
          title_fontsize=22,
          loc="center",
          fontsize = 20,
          bbox_to_anchor=(1.0, 0, 0.5, 1))

plt.setp(autotexts, size=15, weight="bold")
plt.savefig('target_resampling.png',bbox_inches="tight")

X_train_set,X_valid_set, y_train_set,y_valid_set = train_test_split(
    X_train_new,y_train_new, test_size=0.2, random_state=42)


# Entrainement d'un modèle XGBoost

parameters = {"learning_rate":[0.1], "n_estimators":[100],"max_depth":[5],
              "min_child_weight":[2,4],"gamma":[i/10.0 for i in range(2,4)],
              "subsample":[i/10.0 for i in range(7,9)],"colsample_bytree":[0.8],
              "objective":['binary:logistic'],"nthread":[4],"scale_pos_weight":[1],"seed":[27]}

xgb_grid = GridSearchCV(XGBClassifier(),
                        param_grid = parameters,verbose=2,cv=5,
                        scoring=make_scorer(f1_score))

xgb_grid.fit(X_train_set, y_train_set)

filename = 'xgboost_classifier_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(xgb_grid,file)
    
models_score = pd.DataFrame({})
models_score = models_score.append(pd.DataFrame(
    {'Models' : ['XGBoost Classifier'],
     'Training set' : [xgb_grid.best_score_],
     'Validation set' : [f1_score(xgb_grid.predict(X_valid_set), y_valid_set)]}),
                                   ignore_index=True)

# Entrainement d'un modèle  de regression logistique

parameters = {"C":np.logspace(-3,3,5),'solver':['liblinear']}

rl_grid = GridSearchCV(LogisticRegression(random_state=4),
                        param_grid = parameters,verbose=2,cv=5,
                        scoring=make_scorer(f1_score))

rl_grid.fit(X_train_set, y_train_set)

filename = 'logistic_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(rl_grid,file)
    
models_score = models_score.append(pd.DataFrame(
    {'Models' : ['Logistic Regression'],
     'Training set' : [rl_grid.best_score_],
     'Validation set' : [f1_score(rl_grid.predict(X_valid_set), y_valid_set)]}),
                                   ignore_index=True)
    
# Entrainement d'un modèle Random Forest

parameters = {'n_estimators' : [100], 'min_samples_leaf' : [1,2,3,4,5], 
              'max_features': [int(x) for x in np.linspace(1,10,5)],'n_jobs': [-1]}

rfc_grid = GridSearchCV(RandomForestClassifier(random_state=4),
                        param_grid = parameters,verbose=2,cv=5,
                        scoring=make_scorer(f1_score))

rfc_grid.fit(X_train_set, y_train_set)

filename = 'random_forest_classifier_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(rfc_grid,file)
    
models_score = models_score.append(pd.DataFrame(
    {'Models' : ['Random Forest Classifier'],
     'Training set' : [rfc_grid.best_score_],
     'Validation set' : [f1_score(rfc_grid.predict(X_valid_set), y_valid_set)]}),
                                   ignore_index=True)

models_score.to_csv('models_score.csv')