#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

Input_File_Path = 'Salary_Data.csv'
T1 = pd.read_csv(Input_File_Path)

X = T1.iloc[:, :-1].values    #Take All Rows But Only Take X variables, Not the Y Variable
Y = T1.iloc[:, -1]            #Take only Last Column i.e. Y Variable

#Splitting Dataset into Training and Test Set
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, random_state=1)

#Training the Simple Linear Regression Model on Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_Train, Y_Train)

# Save Model
pickle.dump(regressor, open('model.pkl','wb'))

#Predicting The Test (Validation) Set
Y_pred = regressor.predict(X_Test)     #Just use X_Test to Predict

#Visualising Training Set Results
plt.scatter(X_Train, Y_Train, color = 'red')
plt.plot(X_Train, regressor.predict(X_Train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


#Visualising Test Set Results
plt.scatter(X_Test, Y_Test, color = 'red')
plt.plot(X_Train, regressor.predict(X_Train), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


#To predict Salary for a person with 12 Years of Experience
print(regressor.predict([[12]]))       #Used [[]] to make it into 2D Array as predict accepts 2D Array aleays


#Get Final Equation
print(regressor.coef_)
print(regressor.intercept_)


# In[6]:


# load model
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[14]]))

