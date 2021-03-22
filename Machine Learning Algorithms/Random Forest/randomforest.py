#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 13:15:20 2021

@author: omertas
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:23:39 2020

@author: omertas
"""
#hw5
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

#Load the data
data = pd.read_csv('CTG.csv')
#Prepare the data
update =[1 if x==1 else 0 for x in data["NSP"]]

data["NSP"] = update

columns = len(data.columns)
#Separate futures and labels
#I will just work with some of the futures
x = data[['ASTV', 'MLTV', 'Max', 'Median']].values
y = data.iloc[:, -1].values

"""
#Normalizaion may be required fr some algorithm
#Seperate futures and target
#no need for decision tree
x = data.iloc[:,0:4].values
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

le = LabelEncoder()
y = le.fit_transform(data['class'].values)
"""

#Train and tst split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.5,
                                                    random_state = 3)


#Define Random forest model
random_forest = RandomForestClassifier()

#Fit RM into model

random_forest.fit(x_train, y_train)


#Preedict with test data
pred_k = random_forest.predict(x_test)


#Find accuracy
accuracy = 1-np.mean(pred_k!=y_test)

#or
accuracy_score = random_forest.score(x_test, y_test)



#this sample is to measure how depth and number of estimators affect accuracy
print("Question 4 output")
lst_error_rate = []
cfm = []
for i in range(1,6):
    
    for j in range(1,11):
        group_2 = data[['ASTV', 'MLTV', 'Max', 'Median', 'NSP']]
        X = group_2[['ASTV', 'MLTV', 'Max', 'Median']].values
        le = LabelEncoder()
        Y = le.fit_transform(group_2["NSP"].values)

        X_train,X_test,Y_train,Y_test=train_test_split(X, Y,
                                                       test_size=0.5,random_state=3)
    
        model = RandomForestClassifier(n_estimators=j, max_depth=i,
                               criterion='entropy')
        model.fit(X_train, Y_train)
        prediction = model.predict(X_test)
        error_rate = np.mean(prediction != Y_test)
        print("N is", j, "d is", i, "error rate", 
              str(round(error_rate*100,2)) + "%")
        lst_error_rate.append(error_rate)
        #will pick the best later keep changing
        cfm.append(confusion_matrix(Y_test, prediction))

#need this to grpah acrros the depths     
lst = lst_error_rate
y = [lst[i:i + 10] for i in range(0, len(lst), 10)]
x =  np.arange(1,11)
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
#Question 4.2
plt.plot(x, y[0], label = "max depth 1",marker='o')
plt.plot(x, y[1], label = "max depth 2", marker='o')
plt.plot(x, y[2], label = "max depth 3", marker='o')
plt.plot(x, y[3], label = "max depth 4", marker='o')
plt.plot(x, y[4], label = "max depth 5", marker='o')
plt.legend(loc="lower left")
plt.xlabel("Number of Estimators")
plt.ylabel("Error Rate")
plt.xticks(x)
plt.title("Random Forest Error rate & Estimators")
plt.show()

#Question 4.3
print("\n")
min_error_index = np.argmin(lst_error_rate)
print("Best combination when N is at" ,min_error_index%10 + 1,
      "and d is at" ,min_error_index//10 +1, "with acuaracy of",
      str(round((1-lst_error_rate[min_error_index])*100,2)) + "%")




