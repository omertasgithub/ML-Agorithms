#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:45:53 2021

@author: omertas
"""

#Import libraries 
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#load data
df = pd.read_csv('multiple-lr-data.csv')
df.head()

#seperate futures
y = df['loan']
x = df[['age','credit-rating','children']]

# define the multiple Linear regression model
linear_regress = LinearRegression()


# Fit the multiple Linear regression model
linear_regress.fit(x,y)


# predict with the data
y_pred = linear_regress.predict(x)

print(y_pred)


print(mean_squared_error(y, y_pred))