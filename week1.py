#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 14:15:18 2023

@author: WBR
"""
#%%
import pandas as pd
from sklearn import tree
from sklearn.metrics import cohen_kappa_score
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
import statistics
import numpy as np
#%%

hdir = '/Users/WBR/walter/local_professional/Big_Data_Ed_s2023'
df = pd.read_csv(hdir + "/week1/a1-in.csv")
# =============================================================================
# dataset from paper (https://learninganalytics.upenn.edu/ryanbaker/Godwinetal_v12-2.pdf)
# =============================================================================
#%% dataset basics

df.columns

df.nunique(dropna=True)

df.head()
df.dtypes

outcome = 'ONTASK'
# id_vars = ['UNIQUEID', 'SCHOOL', 'Class', 'STUDENTID']
# interest_vars = ['Activity','TRANSITIONS','NumACTIVITIES', 'FORMATchanges', 'NumFORMATS',]
# other_vars = ['GRADE','CODER','Obsv/act','Transitions/Durations', 'Total Time',]


#%%
df = pd.get_dummies(df, columns=['SCHOOL', 'Class', 'CODER', 'Activity'])

X = df.drop('ONTASK',axis=1)
Y = df['ONTASK'].copy()

# produces slightly different answer than OneHot encoding with pd.get_dummies
# from sklearn import preprocessing
# enc = preprocessing.OrdinalEncoder()
# X = enc.fit_transform(X)

clf = tree.DecisionTreeClassifier(min_samples_split=10)

def fit_pred_kappa(X,Y,clf):
    clf = clf.fit(X, Y)
    pred = clf.predict(X)
    return cohen_kappa_score(Y, pred)
    

#%%
fit_pred_kappa(X,Y,clf)
#%% 
X2 = X.drop('STUDENTID',axis=1)
fit_pred_kappa(X2, Y,clf)
#%% 
df = pd.read_csv(hdir + "/week1/a1-in.csv")
df = pd.get_dummies(df, columns=[ 'Activity'])
df = df.drop(['ONTASK','STUDENTID','SCHOOL', 'Class', 'CODER', 'UNIQUEID'],axis=1)
fit_pred_kappa(df, Y,clf)
#%%
clf = GaussianNB()
fit_pred_kappa(df, Y,clf)

#%%
clf = XGBClassifier(learning_rate=0.5, n_estimators=200, random_state=5)
Y = pd.get_dummies(Y)
Y = Y.drop('N',axis = 1)
fit_pred_kappa(df, Y,clf)
clf = clf.fit(df,Y)

#%% Q7, Q8, Q9
df = pd.read_csv(hdir + "/week1/a1-in.csv")

#define groups so that each student will only appear in one batch
group_dict = {}
groups = np.array([])
for index, row in df.iterrows():
    student_id = row['STUDENTID']
    if student_id not in group_dict:
        group_dict[student_id] = index
    groups = np.append(groups, group_dict[student_id])

df = df.set_index(groups)

Y = df['ONTASK'].copy()
df = pd.get_dummies(df, columns=[ 'Activity'])
X = df.drop(['ONTASK','STUDENTID','SCHOOL', 'Class', 'CODER', 'UNIQUEID'],axis=1)

# Q7 # clf = tree.DecisionTreeClassifier(min_samples_split=10)
# Q8 # clf = GaussianNB()
clf = XGBClassifier(learning_rate=0.5, n_estimators=200, random_state=5)
Y = pd.get_dummies(Y)
Y = Y.drop('N',axis = 1)

# define splits
splits = GroupKFold(n_splits=10)


# train and test splits
accuracy = []
count = 0
for train_index, test_index in splits.split(X,Y,groups):
    count += 1
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    # breakpoint()
    print("Training model ",count)
    # acc_score = fit_pred_kappa(X_test,y_test,clf)
    # maybe problem is in fit_pred_kappa function usage, therefore dismantle func
    # Yes!! fit_pred_kappa was training and testing on same splits.
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc_score = cohen_kappa_score(y_test, pred)
    accuracy.append(acc_score)
print("Finished training.\n")

#Mean accuracy with regard to y for each model
index = 0
for a in accuracy: 
    index += 1
    print("Accuracy score for model", index, " ", a)

#Report the average accuracy for all models 
print("\nAverage accuracy score for all models: ", statistics.mean(accuracy))
print("Maximum accuracy score of all models: ", max(accuracy))
print("Minimum accuracy score of all models: ", min(accuracy))

# correct answer .1 
    
#%% Q8


#%%
#%%
#%%
#%%#%%
#%%
#%%
#%%




















