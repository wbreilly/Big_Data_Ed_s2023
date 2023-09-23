#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 12:07:34 2023

@author: WBR
"""

#%% 
import pandas as pd

#%%

hdir = '/Users/WBR/walter/local_professional/Big_Data_Ed_s2023'
reg = pd.read_csv(hdir + "/week2/reg.csv")
cla = pd.read_csv(hdir + "/week2/class.csv")

#%% Q1
reg.corr()

#%% Q2
import numpy as np

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

rmse(reg['data'],reg['predicted (model)'])

#%% Q3

def MAE(predictions, targets):
    return (predictions - targets).abs().mean()

MAE(reg['data'],reg['predicted (model)'])

#%% Q4

cla['pred'] = cla['Predicted (Model)'] > .5
cla['correct'] = np.where(((cla['Data'] == 'Y') & (cla['pred'] == True)) | ((cla['Data'] == 'N') & (cla['pred'] == False)), 1,0)

cla['correct'].mean()

#%% Q5

cla['pred'].astype(int).sum() # most common class is 0
cla['pred2'] = False
cla['correct2'] = np.where(((cla['Data'] == 'Y') & (cla['pred2'] == True)) | ((cla['Data'] == 'N') & (cla['pred2'] == False)), 1,0)

cla['correct2'].mean()

#%% Q7
from sklearn.metrics import cohen_kappa_score

cla['bin_Data'] = np.where(cla['Data'] == 'Y',1,0)
cohen_kappa_score(cla['bin_Data'],cla['pred'])

#%% Q8
from sklearn.metrics import precision_score

precision_score(cla['bin_Data'],cla['pred'])
#fewer than 5% false positives

#%% Q9
from sklearn.metrics import recall_score

recall_score(cla['bin_Data'],cla['pred'])
# 47% false negatives



#%% Q11
from sklearn.metrics import  roc_auc_score
roc
roc_auc_score(cla['bin_Data'],cla['pred'])
#%%
