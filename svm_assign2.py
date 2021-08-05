# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 18:17:53 2021

@author: asilp
"""



import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
forest_fire = pd.read_csv(r"C:\Users\asilp\Desktop\datascience\assign12\forestfires.csv")


forest_fire.columns
forest_fire.dtypes

#dummy variable creation
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
forest_fire['month']=le.fit_transform(forest_fire['month'])
forest_fire['day']=le.fit_transform(forest_fire['day'])
forest_fire['size_category']=le.fit_transform(forest_fire['size_category'])



  
forest_fire.describe()    
forest_fire.head()
                                   
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
train,test=train_test_split(forest_fire,test_size=0.2)

train_X = train.iloc[:, :30]
train_y = train.iloc[:, [30]]
test_X  = test.iloc[:, :30]
test_y  = test.iloc[:, [30]]


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


