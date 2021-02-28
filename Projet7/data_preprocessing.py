#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 18:51:26 2021

@author: famien
"""
import os
import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, StandardScaler


def get_files_stats(filename):
    """
    Ce code calcule les statistiques sur des fichiers passés en argument 
    et les retourne en sortie.

    Parameters
    ----------
    filename : fichiers de données formatés en lignes et colonnes.
        DESCRIPTION.

    Returns
    -------
    "Taille": représente le nombre de célulle;
    "Nbre ligne": représente le nombre de ligne;
    "Nbre colonne": représente le nombre de colonne;
    "Nbre NaN": représente le nombre de valeurs manquantes;
    "Pourcentage de NaN": représente le pourcentage de valeurs manquantes.

    """
    
    data0 = pd.read_csv(filename,encoding='ISO-8859-1')
    data0 = data0.loc[:,~data0.columns.str.contains('^Unnamed')]
    
    nbRows = data0.shape[0]
    nbCols = data0.shape[1]
    nbNaN = data0.isna().sum().sum()
    pNaN = round(100.0 * nbNaN/data0.size,2)
    
    statsValues = [data0.size,nbRows,nbCols,nbNaN,pNaN]
    
    return statsValues

def agg_numeric(df, group_var, df_name):
    """Aggregates the numeric values in a dataframe. This can
    be used to create features for each instance of the grouping variable.
    
    Parameters
    --------
        df (dataframe): 
            the dataframe to calculate the statistics on
        group_var (string): 
            the variable by which to group df
        df_name (string): 
            the variable used to rename the columns
        
    Return
    --------
        agg (dataframe): 
            a dataframe with the statistics aggregated for 
            all numeric columns. Each instance of the grouping variable will have 
            the statistics (mean, min, max, sum; currently supported) calculated. 
            The columns are also renamed to keep track of features created.
    
    """
    # Remove id variables other than grouping variable
    for col in df:
        if col != group_var and 'SK_ID' in col:
            df = df.drop(columns = col)
            
    group_ids = df[group_var]
    numeric_df = df.select_dtypes('number')
    numeric_df[group_var] = group_ids
    
    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()

    # Need to create new column names
    columns = [group_var]

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var != group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))

    agg.columns = columns
    return agg

def count_categorical(df, group_var, df_name):
    """Computes counts and normalized counts for each observation
    of `group_var` of each unique category in every categorical variable
    
    Parameters
    --------
    df : dataframe 
        The dataframe to calculate the value counts for.
        
    group_var : string
        The variable by which to group the dataframe. For each unique
        value of this variable, the final dataframe will have one row
        
    df_name : string
        Variable added to the front of column names to keep track of columns

    
    Return
    --------
    categorical : dataframe
        A dataframe with counts and normalized counts of each unique category in every categorical variable
        with one row for every unique value of the `group_var`.
        
    """
    
    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('object'))

    # Make sure to put the identifying id on the column
    categorical[group_var] = df[group_var]

    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])
    
    column_names = []
    
    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['count', 'count_norm']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))
    
    categorical.columns = column_names
    
    return categorical

def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

def encode_and_clean_features(features,test_features):
    
    # Extract the ids
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']
    
    # Extract the labels for training
    labels = features['TARGET']
    
    # Remove the ids and target
    features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns = ['SK_ID_CURR'])
    
    # Create a label encoder
    label_encoder = LabelEncoder()
       
    # List for storing categorical indices
    cat_indices = []
        
    # Iterate through each column
    for i, col in enumerate(features):
        if features[col].dtype == 'object':
            # Map the categorical features to integers
            features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
            test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

            # Record the categorical indices
            cat_indices.append(i)
    
    features_names = list(features.columns)
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    features = imputer.fit_transform(features)
    test_features = imputer.transform(test_features)
    
    features = scaler.fit_transform(features)
    test_features = scaler.transform(test_features)
    
    features = pd.DataFrame(features, columns = features_names)
    test_features = pd.DataFrame(test_features, columns = features_names)
    
    features['SK_ID_CURR'] = train_ids
    features['TARGET'] = labels
    test_features['SK_ID_CURR'] = test_ids
    
    return features, test_features 
    
####### Code principal

## Récupération des fichiers de données csv

file_list = ["application_train.csv","application_test.csv","bureau.csv","bureau_balance.csv",
             "credit_card_balance.csv","HomeCredit_columns_description.csv","installments_payments.csv",
             "POS_CASH_balance.csv","previous_application.csv","sample_submission.csv"]


## Recuperation des stats par fichiers de données

files_stats_colname = ['Taille','Nbre ligne','Nbre colonne','Nbre NaN',
                       'Pourcentage de NaN']

files_stats = pd.DataFrame(columns=(files_stats_colname),index=(file_list))
for files in file_list:
    print(files)
    files_stats.loc[files] =get_files_stats('data/'+files)
    
    data = pd.read_csv('data/'+files,encoding='ISO-8859-1')
    print(data.isna().sum())
 

## Lecture des données
app_train = pd.read_csv('data/application_train.csv',encoding='ISO-8859-1')
app_test = pd.read_csv('data/application_test.csv',encoding='ISO-8859-1')
#bureau = pd.read_csv('data/bureau.csv',encoding='ISO-8859-1')
#bureau_balance = pd.read_csv('data/bureau_balance.csv',encoding='ISO-8859-1')
#previous_app = pd.read_csv('data/previous_application.csv',encoding='ISO-8859-1')
#credit = pd.read_csv('data/credit_card_balance.csv',encoding='ISO-8859-1')
#cash = pd.read_csv('data/POS_CASH_balance.csv',encoding='ISO-8859-1')
#payments = pd.read_csv('data/installments_payments.csv',encoding='ISO-8859-1')

## Représentation de la variable 'TARGET' (jeu d'entrainement)

fig,ax = plt.subplots(figsize=(20, 10), dpi= 80, facecolor='w', edgecolor='k')

def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%".format(pct)

colors = ['red', 'blue']

wedges, texts, autotexts = plt.pie(app_train['TARGET'].value_counts(),
                                   autopct=lambda pct: func(pct, app_train['TARGET'].value_counts())
                                   ,colors=colors,textprops=dict(color="w"))

ax.legend(wedges, app_train['TARGET'].value_counts().index,
          title="TARGET",
          title_fontsize=22,
          loc="center",
          fontsize = 20,
          bbox_to_anchor=(1.0, 0, 0.5, 1))

plt.setp(autotexts, size=15, weight="bold")
plt.savefig('target.png',bbox_inches="tight")
plt.show()

    
## Recueil des infos sur les anciens crédits des clients

#previous_loan_counts = bureau.groupby(
#    'SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(
#        columns = {'SK_ID_BUREAU': 'PREVIOUS_LOAN_COUNTS'})
        
#previous_loan_counts.head()

#app_train = app_train.merge(previous_loan_counts, on = 'SK_ID_CURR', how = 'left')
#app_train['PREVIOUS_LOAN_COUNTS'] = app_train['PREVIOUS_LOAN_COUNTS'].fillna(0)


#bureau_counts = count_categorical(bureau, group_var = 'SK_ID_CURR', df_name = 'bureau')
#bureau_agg = agg_numeric(bureau.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'bureau')
#bureau_balance_counts = count_categorical(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
#bureau_balance_agg = agg_numeric(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')


# Dataframe grouped by the loan
#bureau_by_loan = bureau_balance_agg.merge(bureau_balance_counts, right_index = True, left_on = 'SK_ID_BUREAU', how = 'outer')

# Merge to include the SK_ID_CURR
#bureau_by_loan = bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].merge(bureau_by_loan, on = 'SK_ID_BUREAU', how = 'left')

# Aggregate the stats for each client
#bureau_balance_by_client = agg_numeric(bureau_by_loan.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'client')


# Merge with the value counts of bureau
#app_train = app_train.merge(bureau_counts, on = 'SK_ID_CURR', how = 'left')

# Merge with the stats of bureau
#app_train = app_train.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')

# Merge with the monthly information grouped by client
#app_train = app_train.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')


# Merge with the value counts of bureau
#app_test = app_test.merge(bureau_counts, on = 'SK_ID_CURR', how = 'left')

# Merge with the stats of bureau
#app_test = app_test.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')

# Merge with the value counts of bureau balance
#app_test = app_test.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')


missing_train = missing_values_table(app_train)
missing_train.head(100)

train_labels = app_train['TARGET']

# Align the dataframes, this will remove the 'TARGET' column
app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)

app_train['TARGET'] = train_labels

train_features, test_features = encode_and_clean_features(app_train,app_test)

missing_train = missing_values_table(train_features)
missing_train.head(100)

train_features.to_csv('app_train.csv')
test_features.to_csv('app_test.csv')


test_features['TARGET'] = np.nan
data = train_features.append(test_features,ignore_index=True,sort=False)
data.to_csv("data/data.csv")

app_test['TARGET'] = np.nan
app_data = app_train.append(app_test,ignore_index=True,sort=False)
app_data.to_csv("data/app_data.csv")