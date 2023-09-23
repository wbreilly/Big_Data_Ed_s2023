#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:44:36 2023

@author: WBR
"""


#%% 
import pandas as pd
import numpy as np

#%%

hdir = '/Users/WBR/walter/local_professional/Big_Data_Ed_s2023'
df = pd.read_csv(hdir + "/week8/BDEPennW8.csv")

#%% Q1

n = df.poster.nunique()
(n *  (n - 1)) / 2

#%% Q2

responses_df = df[df['response-to'].notna()]

# # Create pairs of posters (responder, original poster)
# connections = list(zip(responses_df['poster'], responses_df['response-to']))

# # Count unique pairs
# unique_connections = set(connections)
# num_connections = len(unique_connections)

# Create a mapping of posters to threads
poster_to_thread = df.set_index('ID')['thread'].to_dict()

# Initialize a set to store unique connections
unique_connections = set()

# Iterate through each row in the dataset
for index, row in df.iterrows():
    poster = row['poster']
    response_to = row['response-to']
    thread = row['thread']
    
    # Check if the response is within the same thread and if the poster is different
    if not pd.isna(response_to) and poster_to_thread.get(response_to) == thread and poster != response_to:
        # Add the pair of posters to the set
        unique_connections.add(tuple(sorted([poster, response_to])))

# Count the unique pairs
num_connections = len(unique_connections)



#%%
#%%
#%%
#%%
#%%