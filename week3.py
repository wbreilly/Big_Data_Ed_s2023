#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:47:59 2023

@author: WBR
"""

#%% 
import pandas as pd
#%%

# assistments dataset from https://learning-analytics.info/index.php/JLA/article/view/3536/4014

hdir = '/Users/WBR/walter/local_professional/Big_Data_Ed_s2023'
df = pd.read_csv(hdir + "/week3/actions.csv")
df2 = pd.read_csv(hdir + "/week3/observations.csv")

#%%
df['double-anon-userID'].nunique()
# 236 participants
df.groupby('double-anon-userID')['problemId'].nunique().describe()
# about 5 unique problems per participant
#%%
df2.groupby('double-anon-userID')['problemId'].nunique().describe()
#%% Q1
df2['n_problem'] = df2.groupby('double-anon-userID')['problemId'].transform('nunique')
df2['n_gaming'] = df2.groupby('double-anon-userID')['IS-GAMING'].transform('sum')
df2['pct_gaming'] = df2['n_gaming'] / df2['n_problem']

#%% Q2
df2['bored'] = df2['AFFECT'].str.contains('BORED')
df2['ever_bored'] = df2.groupby('double-anon-userID')['bored'].transform('sum') 
df2[df2['ever_bored'] == 0]['double-anon-userID'].nunique()

#%% Q3
q3 = df2[df2['double-anon-userID'] == 30314880]

#%% Q4
df.groupby(['double-anon-userID','problemId'])
df.groupby('ObservationID-fullanon')['double-anon-userID'].count().mean()
df['observation_count'] = df.groupby('ObservationID-fullanon')['double-anon-userID'].transform('count')
# eliminate two observations with most actions (anything over 100 works), then find mean
df[df['observation_count'] < 100].groupby('ObservationID-fullanon')['double-anon-userID'].count().mean()
#%% Q7
timeTaken
df['mean_obs_timeTaken'] = df.groupby('ObservationID-fullanon')['timeTaken'].transform('mean')
df[df['ObservationID-fullanon'] == 'MFDTT-mathasst-9-at_12:58:03-79']['mean_obs_timeTaken']

#%% Q8
df['max_obs_timeTaken'] = df.groupby('ObservationID-fullanon')['timeTaken'].transform('max')
df[df['ObservationID-fullanon'] == 'EGMDH-math_assistments-4-at_10:34:30-9']['max_obs_timeTaken']
#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%


