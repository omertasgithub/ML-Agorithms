#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:28:04 2021

@author: omertas
"""


#import libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, confusion_matrix




#Load data
data = pd.read_csv("data_banknote_authentication.csv")

 
#Seperate futures and target
#no need for decision tree
x = data.iloc[:,0:4].values
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

le = LabelEncoder()
y = le.fit_transform(data['class'].values)

#Train test split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.5,
                                                    random_state=3)

#Define Gradient boost
gradient_boost = GradientBoostingClassifier()


#Fit KNN model
gradient_boost.fit(x_train, y_train)

#Predict with the data
pred_k = gradient_boost.predict(x_test)

#Find accuracy 
accuracy = 1-np.mean(pred_k != y_test)
#or
accuracy_score = gradient_boost.score(x_test, y_test)
