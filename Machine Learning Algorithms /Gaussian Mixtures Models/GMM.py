#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:15:54 2021

@author: omertas
"""

#import libraries 
from sklearn.preprocessing import StandardScaler, LabelEncoder

import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance

#load data
seed = pd.read_csv("seeds_dataset.txt", sep="\t", names = 
                     ["f1","f2", "f3", "f4", "f5", "f6", "f7", "class"])

seed = seed.dropna()
#separate data into features and target/labels
X = seed[["f1","f2", "f3","f4","f5","f6", "f7"]].values
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform (X)
le = LabelEncoder ()
Y = le.fit_transform(seed["class"].values)
Y = seed["class"].values

#split train and test no need for
#X_train,X_test,Y_train,Y_test=train_test_split(X, Y,
                             #    test_size=0.5,random_state=10)


#define the model
gmm = GaussianMixture(n_components=4)

#fit the model
gmm.fit(X)


#predict clusters
y_gmm = gmm.predict(X)


#predict accuracy
accuracy = 1- np.mean(y_gmm!=Y)
