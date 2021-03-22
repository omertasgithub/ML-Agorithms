#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression




banknote = pd.read_csv("data_banknote_authentication.csv")

col = banknote.columns

#seperate futures
x = banknote[col[0:-1]]
y = banknote[col[-1]]

#split into train test 
x_train, x_test, y_train, y_test = \
    train_test_split(x,y,train_size = 0.5, stratify = y)
    
#define the model   
log_reg_classifier = LogisticRegression()

#Fit multiple logistic regression model 
log_reg_classifier.fit(x_train, y_train)

#predcit with the data
prediction = log_reg_classifier.predict(x_test)

#find accuracy
accuracy_logistic = np.mean(prediction == y_test)
    
#Or use 
log_reg_classifier.score(x_test, y_test)