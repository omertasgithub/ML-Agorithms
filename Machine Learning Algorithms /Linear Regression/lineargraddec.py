#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 20:01:02 2021

@author: omertas
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 9.0)

# Preprocessing Input data
data = pd.read_csv('data.csv')
X = data.iloc[:,0]
Y = data.iloc[:,1]
plt.scatter(X, Y)
plt.show()



#Bulinding model

m = 0   #slope
c = 0   #y-intercept

L = 0.00001 #The learning rate


epochs= 1000 #the number of iterations

n = len(X)

for i in range(epochs):
    Y_pred = m*X + c #The current predicted valiue of Y
    D_m = (-2/n) * sum(X * (Y-Y_pred)) #Derivative of m
    D_c = (-2/n) * sum(Y-Y_pred) #Derivative of c
    
    m = m - L * D_m #update m
    c = c - L * D_c #update c
print(m,c)


Y_pred = m*X + c
plt.scatter(X, Y) 
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
plt.show()

