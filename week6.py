#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 17:48:42 2023

@author: WBR
"""


#%% 
import pandas as pd
import numpy as np
#%%

hdir = '/Users/WBR/walter/local_professional/Big_Data_Ed_s2023'
df = pd.read_csv(hdir + "/week6/a6-in-assoc.csv")

#%% Q3 Association Rule Mining

# write a function that computes support for a given if_var and then_var combo
# support is the number of datapoints (rows) that fit the rule, divided by the total (nrows)

def support(if_var,then_var):
    nfit = np.where(((df[if_var] == 1) & (df[then_var] == 1)) | ((df[if_var] == 0) & (df[then_var] == 0)), 1,0).sum()
    nrow = len(df)
    return nfit/nrow

support('resources','calendars_clocks')
support('resources','content_specific') #1
support('resources','rules')
support('resources','decorations')

#%% Q6

def support2(if_var,then_var,then_var2):
    nfit = np.where(((df[if_var] == 1) & (df[then_var] == 1) & (df[then_var2] == 1)) | ((df[if_var] == 0) & (df[then_var] == 0) & (df[then_var2] == 0)), 1,0).sum()
    nrow = len(df)
    return nfit/nrow

support2('yearly_schedule','resources','decorations')

#%% Q7

def confidence2(if_var,then_var,then_var2):
    nfit = np.where(((df[if_var] == 1) & (df[then_var] == 1) & (df[then_var2] == 1)) | ((df[if_var] == 0) & (df[then_var] == 0) & (df[then_var2] == 0)), 1,0).sum()
    nrule = np.where((df[if_var] == 1),1,0).sum()
    return nfit/nrule

confidence2('yearly_schedule','resources','decorations')

#%% Q8

def lift2(if_var,then_var,then_var2):
    nfit = np.where(((df[if_var] == 1) & (df[then_var] == 1) & (df[then_var2] == 1)) | ((df[if_var] == 0) & (df[then_var] == 0) & (df[then_var2] == 0)), 1,0).sum()
    ncomb = (np.where((df[if_var] == 1),1,0).sum() / len(df)) * (np.where(((df[then_var] == 1) & (df[then_var2] == 1)),1,0).sum() / len(df))
    return nfit/ncomb

lift2('yearly_schedule','resources','decorations')

# conf / prob_then
def prob_then(then_var, then_var2):
    return np.where(((df[then_var] == 1) & (df[then_var2] == 1)),1,0).sum() / len(df)
    
1 / prob_then('resources','decorations')

#%% Q10

np.where(((df['content_specific'] == 1)),1,0).sum() / len(df)

#%%
#%%
                        