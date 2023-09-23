#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 17:41:39 2023

@author: WBR
"""

#%% 
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#%%

hdir = '/Users/WBR/walter/local_professional/Big_Data_Ed_s2023'
df = pd.read_csv(hdir + "/week7/clustering.csv")

#%% Q2

kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(df)

cent = kmeans.cluster_centers_
cent = cent.transpose()
cent = pd.DataFrame(cent)
cent['dist'] =  cent[0] - cent[1]

#%% Q2

label = kmeans.predict(df)
filt0 = df[label==0][['a','f']]
filt1 = df[label==1][['a','f']]

plt.scatter(filt0['a'] , filt0['f'] , color = 'red')
plt.scatter(filt1['a'] , filt1['f'] , color = 'blue')

#%% Q4 # plot 7 clusters and cluster centers

def cluster_plot(df,n_clusters,n_vars):  
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(df)
    cent = kmeans.cluster_centers_
    label = kmeans.predict(df)

    colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
    
    ilab = 0
    slices = {}
    for ilab in range(n_clusters):
        vname = f"var_{ilab}"
        islice = df[label == ilab][['a','f']]
        slices[vname] = islice
        plt.scatter(slices[vname]["a"],slices[vname]["f"], color = colors[ilab])
        plt.scatter(cent[:,0],cent[:, n_vars - 1],marker = 'x')
    
cluster_plot(df,7,6)

#%% Q6

df_af = df[['a','f']]

cluster_plot(df_af,7,2)
cluster_plot(df_af,11,2)


#%% Q9

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster 

agg_clustering = AgglomerativeClustering(n_clusters=7)
agg_clustering.fit(df_af)

plt.scatter(df_af["a"], df_af["f"], c=agg_clustering.labels_, cmap='viridis', s=50)

# Create linkage matrix
linkage_matrix = linkage(df_af, method='ward')

# Visualize the clustering result
plt.figure(figsize=(12, 6))

# Scatter plot of the data points colored by their cluster labels
plt.subplot(121)
plt.scatter(df_af["a"], df_af["f"], c=agg_clustering.labels_, cmap='viridis', s=50)
plt.title('Agglomerative Clustering Result')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')


# Dendrogram plot
# plt.subplot(122)
dendrogram(linkage_matrix)
plt.title('Dendrogram')
plt.xlabel('Data Point Index')
plt.ylabel('Distance')

# Get cluster assignments for each data point
clusters = fcluster(linkage_matrix, t=4, criterion='maxclust')  # You can adjust the threshold 't'
fcluster
# Count the number of data points in the two main branches
num_points_branch1 = len(np.where(clusters == 1)[0])
num_points_branch2 = len(np.where(clusters == 2)[0])

# Add labels to the dendrogram plot
plt.text(200, 160, f'Branch 1: {num_points_branch1} points', fontsize=10, color='blue')
plt.text(200, 140, f'Branch 2: {num_points_branch2} points', fontsize=10, color='red')


# figure scaling and dpi scaling needs work


#%%
#%%
#%%
#%%
#%%
