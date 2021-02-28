#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:20:51 2021

@author: famien
"""
import pickle
import pandas as pd
from flask import Flask, jsonify

DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

def read_dataframe(path,filename):
    data = pd.read_csv(path+filename)
    data.drop(['Unnamed: 0'],axis=1,inplace=True)
    
    return data

def load_model(modelname):
    
    modelpath = 'models/'
    modelname = modelname.replace(" ","")
    
    if modelname == 'RandomForest':
        return pickle.load(open(modelpath+"random_forest_classifier_model.pkl",'rb'))
    elif modelname == 'XGBoost':
        return pickle.load(open(modelpath+"xgboost_classifier_model.pkl",'rb'))
    elif modelname == 'LogisticRegression':
        return pickle.load(open(modelpath+"logistic_regression_model.pkl",'rb'))
    else:
        return pickle.load(open(modelpath+"random_forest_classifier_model.pkl",'rb'))

        
@app.route('/client/<ID_client>/<modelname>')
def predict_target(ID_client,modelname):

    model = load_model(modelname)

    path_read = 'data/'
    data = read_dataframe(path_read, 'data.csv')    
        
    X = data[data['SK_ID_CURR'] == int(ID_client)]
    X.drop(['SK_ID_CURR','TARGET'],axis=1,inplace=True)
      
    prediction = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0][prediction]
    
    dict_final = {
        'prediction' : int(prediction),
        'proba' : float(proba)
        }
    
    return jsonify(dict_final)
    
if __name__ == "__main__":
    app.run(debug=True)