# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: sanjai M
RegisterNumber:  24901269
*/import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


df = pd.read_csv('student_scores.csv')


print(df)


df.head(0)
df.tail(0)

print(df.head())
print(df.tail())
x = df.iloc[:, :-1].values
print(x)

y = df.iloc[:, 1].values
print(y)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)


plt.scatter(x_train, y_train, color='black')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


plt.scatter(x_test, y_test, color='black')
plt.plot(x_train, regressor.predict(x_train), color='red')
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse = mean_absolute_error(y_test, y_pred)
print('MSE=', mse)

mae = mean_absolute_error(y_test, y_pred)
print('MAE = ', mae)

rmse = np.sqrt(mse)
print("RMSE= ", rmse)
```

## Output:
![simple linear regression model for predicti[exp 2==.pdf](https://github.com/user-attachments/files/17427941/exp.2.pdf)
ng the marks scored](sam.png)
## Data set
![image](https://github.com/user-attachments/assets/3f2a47a9-1b4e-4982-af9f-42d6fc1a23f4)
## Head value
![image](https://github.com/user-attachments/assets/8873e853-c947-4153-b592-0ed26035f0f2)
## Tail value
![image](https://github.com/user-attachments/assets/86d55488-a3eb-478b-95d6-dadaa0c29be2)

## X and Y value
![image](https://github.com/user-attachments/assets/dd1c86c3-06a1-4532-9e03-8dc3a9880afe)

![image](https://github.com/user-attachments/assets/f2a7510a-7926-43b7-b039-52a16433daf0)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
