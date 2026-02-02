# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Rithik M
RegisterNumber: 212225040342
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print("First five row of dataset")
print(df.head())
print("Last five row of dataset")
print(df.tail())
x = df.iloc[:,:-1].values

y = df.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print("Predicted values")
print(y_pred)
print("Actual value")
print(y_test)

plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='orange')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='pink')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
<img width="851" height="936" alt="Screenshot 2026-02-02 123238" src="https://github.com/user-attachments/assets/902bdc7c-bce7-413e-9ec2-0e0fdaa26a50" />

<img width="765" height="681" alt="Screenshot 2026-02-02 123302" src="https://github.com/user-attachments/assets/c69f9fcf-b7df-4a88-add2-08797e8ea149" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
