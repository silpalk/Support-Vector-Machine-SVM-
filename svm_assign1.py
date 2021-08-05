# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 13:10:49 2021

@author: asilp
"""


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
salary_test = pd.read_csv(r"C:\Users\asilp\Desktop\datascience\assign12\SalaryData_Test (1).csv")
salary_train=pd.read_csv(r"C:\Users\asilp\Desktop\datascience\assign12\SalaryData_Train (1).csv")
salary=pd.concat([salary_train,salary_test],axis=0)
salary.columns
salary.drop(['relationship'],axis=1,inplace=True)
salary.drop(['race'],axis=1,inplace=True)
salary.drop(['native'],axis=1,inplace=True)
#dummy variable creation
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
salary['workclass']=le.fit_transform(salary['workclass'])
salary['education']=le.fit_transform(salary['education'])
salary['maritalstatus']=le.fit_transform(salary['maritalstatus'])
salary['occupation']=le.fit_transform(salary['occupation'])
salary['sex']=pd.get_dummies(salary['sex'],drop_first=True)


salary['Salary'].value_counts()
salary['Salary']=pd.get_dummies(salary['Salary'],drop_first=True)

salary_sample=salary.iloc[:700,:]
salary.dtypes  
salary.head()   
salary.describe()    
salary.columns                                    
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
train,test=train_test_split(salary_sample,test_size=0.2)

train_X = train.iloc[:, :10]
train_y = train.iloc[:, [10]]
test_X  = test.iloc[:, :10]
test_y  = test.iloc[:, [10]]


# kernel = linear

model_linear = SVC(kernel = "linear")
model_linear.fit(train_X, train_y)
predict_test=model_linear.predict(test_X)
accuracy_score(predict_test,test_y)

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X, train_y)
predict_test_rbf=model_rbf.predict(test_X)
accuracy_score(predict_test_rbf,test_y)


