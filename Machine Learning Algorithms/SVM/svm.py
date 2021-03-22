#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 12:37:27 2021

@author: omertas
"""

#import libraries 
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random 
from scipy.spatial import distance



#load data and prepare
seed = pd.read_csv("seeds_dataset.txt", sep="\t", names = 
                     ["f1","f2", "f3", "f4", "f5", "f6", "f7", "class"])

seed.dropna(axis=0, inplace=True)
#exclude label 3
seed = seed[seed["class"]!=3]
#convert them into - and + labels
seed["class"] = ["-" if x == 1 else "+" for x in seed["class"]]

#Separate futures and target
X = seed[["f1","f2", "f3", "f4", "f5", "f6", "f7"]].values
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform (X)
le = LabelEncoder ()
Y = le.fit_transform(seed["class"].values)
Y = seed["class"].values


#Train and test split
X_train,X_test,Y_train,Y_test=train_test_split(X, Y,
                                 test_size=0.5,random_state=10)


#Fit svm model and chose function
svm_classifier = svm.SVC(kernel='linear')

#Fit svm model
svm_classifier.fit(X_train,Y_train)

#predict y values by using input test data
predicted_linear_svm = svm_classifier.predict(X_test)

#find error
error_rate = np.mean(predicted_linear_svm!=Y_test)

#or


accuracy = metrics.accuracy_score(Y_test,predicted_linear_svm)
cfm_svm_linear = confusion_matrix(Y_test,predicted_linear_svm)
print("Accuracy is", str(round(accuracy*100,2)) + "%")
print("CFM of Linear SVM\n", cfm_svm_linear)


svm_classifier = svm.SVC(kernel='rbf')
svm_classifier.fit(X_train,Y_train)
predicted = svm_classifier.predict(X_test)
error_rate = np.mean(predicted!=Y_test)
accuracy = metrics.accuracy_score(Y_test,predicted)
cfm_svm_gauss = confusion_matrix(Y_test,predicted)
print("Accuracy is", str(round(accuracy*100,2)) + "%")
print("CFM of Gauss SVM\n", cfm_svm_gauss)



svm_classifier = svm.SVC(kernel='poly', degree = 3)
svm_classifier.fit(X_train,Y_train)
predicted = svm_classifier.predict(X_test)
error_rate = np.mean(predicted!=Y_test)
accuracy = metrics.accuracy_score(Y_test,predicted)
cfm_svm_cubic = confusion_matrix(Y_test,predicted)
print("Accuracy is", str(round(accuracy*100,2)) + "%")
print("CFM of Cubic SVM\n", cfm_svm_cubic)


