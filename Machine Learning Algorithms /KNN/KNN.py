#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 12:39:09 2021

@author: omertas
"""

#import libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier



#Load data
data = pd.read_csv("data_banknote_authentication.csv")

 
#Seperate futures and target
#Also need normalization since requires distances
x = data.iloc[:,0:4].values
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

le = LabelEncoder()
y = le.fit_transform(data['class'].values)

#Train test split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.5,
                                                    random_state=3)

#Define KNN model
k=3
knn_classifier = KNeighborsClassifier(n_neighbors=k)

#Fit KNN model
knn_classifier.fit(x_train, y_train)

#Predict with the data
pred_k = knn_classifier.predict(x_test)

#Find accuracy 
accuracy = 1-np.mean(pred_k != y_test)
#or
accuracy_score = knn_classifier.score(x_test, y_test)



"""
error_rate = []
for k in range(3,24,3):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(x_train, y_train)
    pred_k = knn_classifier.predict(x_test)
    error_rate.append(np.mean(pred_k != y_test))
print("Print 3.1")
accuracy =  1-np.array(error_rate)
print("Knn accuracy are", accuracy)


#Question 3.2
  
figure(figsize=(10,4))
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True)) 

plt.plot(range(3,24,3), error_rate , color='red', linestyle='dashed',
marker='o', markerfacecolor='black', markersize=10)
plt.title('Error Rate vs. k for banknote')
plt.xlabel('number of neighbors: k') 
plt.ylabel('Error Rate')
"""

accuracy_list = []
for k in range(3,100,3):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(x_train, y_train)
    pred_k = knn_classifier.predict(x_test)
    accuracy_list.append(knn_classifier.score(x_test, y_test))
    
plt.plot(range(3,100,3), accuracy_list , color='red', linestyle='dashed',
marker='o', markerfacecolor='black', markersize=10)
plt.title('Accuracy vs. k for banknote')
plt.xlabel('number of neighbors: k') 
plt.ylabel('Accuracy Rate')