# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 19:59:33 2018

@author: sony
"""

 #importing libraries
 import numpy as np
 import matplotlib.pyplot as plt
 import pandas as pd
 #importing the daset
 dataset = pd.read_csv('Position_Salaries.csv')
 X = dataset.iloc[:,1:2]
 Y = dataset.iloc[:,2]
 
 
 #Fitting the random forest regresion model to dataset
 from sklearn.ensemble import RandomForestRegressor
 regressor= RandomForestRegressor(n_estimators=100,random_state=0)
 regressor.fit(X,Y)
 
 #predicting the rersult
 t = [[6]]
 t = np.array(t)
 t = t.reshape(1,-1)
 Y_pred = regressor.predict(t)
 
 #visualization of results
 plt.scatter(X,Y,color='red')
 plt.plot(X,regressor.predict(X),color='blue')
 plt.title('truth or dare bluff')
 plt.xlabel('Position')
 plt.ylabel('salary')
 
 
 
 
 
 
 
 