
#Import libraries 
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

#load data
ctg = pd.read_csv('CTG.csv')

group_2 = ctg[['ASTV', 'MLTV', 'Max', 'Median', 'NSP']]

#Seperate futures and label
X = group_2[['ASTV', 'MLTV', 'Max', 'Median']].values
le = LabelEncoder()
Y = le.fit_transform(group_2["NSP"].values)

#Split into train and test
X_train,X_test,Y_train,Y_test=train_test_split(X, Y,
                                 test_size=0.5,random_state=3)



#Define Naive Bayes Model
naive_bayes = MultinomialNB()

#Fit the model
naive_bayes.fit(X_train, Y_train)


#predict with data

predict_naive = naive_bayes.predict(X_test)


#Find accuracy

accuracy_1 = 1 - np.mean(predict_naive!=Y_test)


#or

accuracy_2 = naive_bayes.score(X_test, Y_test)