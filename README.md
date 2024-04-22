# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries (e.g., pandas, numpy,matplotlib).
2. Load the dataset and then split the dataset into training and testing sets using sklearn library.
3. Create a Linear Regression model and train the model using the training data (study hours as input, marks scored as output).
4. Use the trained model to predict marks based on study hours in the test dataset.
5. Plot the regression line on a scatter plot to visualize the relationship between study hours and marks scored.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sri hari R
RegisterNumber:  212223040202
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/MLSET.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='orange')
lr.coef_
lr.intercept_
```

## Output:
## 1) Head:
![image](https://github.com/srrihaari/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145550674/e5330877-562a-4789-9926-02fe72adecd2)

## 2) Graph Of Plotted Data:
![image](https://github.com/srrihaari/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145550674/a2915013-c834-4111-8067-303d71536442)

## 3) Trained Data:
![image](https://github.com/srrihaari/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145550674/47e7cbe1-692e-43d9-90f7-c93bd1fb0f5d)

## 4) Line Of Regression:
![image](https://github.com/srrihaari/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145550674/8e71ad3d-4324-49d4-9139-35c79c1e10b1)

## 5) Coefficient And Intercept Values:
![image](https://github.com/srrihaari/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145550674/eb9188dc-ae51-4798-b338-7c0b3a6218ff)


 ## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
