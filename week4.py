#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 11:52:20 2023

@author: WBR
"""

#%% 
import pandas as pd
#%%

# data drawn from https://courses.edx.org/assets/courseware/v1/e1cb481d6f82484f874e9f03449bb9ff/asset-v1:PennX+BDE1x+1T2021+type@asset+block/USER475.pdf

hdir = '/Users/WBR/walter/local_professional/Big_Data_Ed_s2023'
df = pd.read_csv(hdir + "/week4/Asgn4-dataset.csv")

#%% Q1
df2 = df[df['KC'] == 'VALUING-CAT-FEATURES'].copy()
#%% Q3

df3 = df2[df2['firstattempt'] == 1].copy()
df3["P(Ln-1)"] = 0 
df3["P(Ln-1|RESULT)"] = 0
df3["P(P(Ln))"] = 0

df3.to_csv(hdir + '/week4/df3.csv',index=False)
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
