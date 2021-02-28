#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 17:46:33 2021

@author: famien
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("models_score.csv")

width = 0.2
x = np.arange(len(data.index))

fig = plt.figure(figsize=(20, 10), dpi= 80, facecolor='w', edgecolor='k')

ax = plt.subplot()
ax.bar(x - width/2,data['Training set'].values,width,label='Training set')
ax.bar(x + width/2,data['Validation set'].values,width,label='Validation set')

ax.set_ylabel('f1_score',fontsize=30)
ax.set_xticks(x)
ax.set_xticklabels(data['Models'].values,rotation=90,fontsize=30)
ax.legend(fontsize=30)