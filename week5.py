#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 13:34:45 2023

@author: WBR
"""

#%% 
import pandas as pd

#%%

hdir = '/Users/WBR/walter/local_professional/Big_Data_Ed_s2023'
set1 = pd.read_csv(hdir + "/week5/set1.csv")
set2 = pd.read_csv(hdir + "/week5/set2.csv")

#%% Q1

(set1['p'] < .05).sum()

#%% Q2

set1['bonf'] = set1['p'] * len(set1)
(set1['bonf'] < .05).sum()

#%% Q3
i= 0
while i < 16:
    i +=1
    print(i, ' : ', .05/16*i)
    

# scipy.stats.false_discovery_control(set1['p'], axis=0, method='bh')

#%% Q4

(set2['p'] > .05).sum()

#%% Q5

set2['bonf'] = set2['p'] * len(set2)
(set2['bonf'] > .05).sum()

#%% Q6
from scipy.stats import false_discovery_control

set2['fdr'] = false_discovery_control(set2['p'], axis=0, method='bh')
(set2['fdr'] > .05).sum()
#%%
#%%
#%%
