# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas, numpy and matplotlib.pyplot.
2. Read the file containing the marks using read_csv.
3. Use scatter and label functions accordingly.
4. from sklearn.model_selecuon import train_test_split and use test spilt accordingly. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sri hari R
RegisterNumber:  212223040202
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/machine.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(X_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='red')
*/
```

## Output:
![image](https://github.com/srrihaari/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145550674/c57b9dfc-4d4d-4929-b5a3-8c9450d99c42)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
