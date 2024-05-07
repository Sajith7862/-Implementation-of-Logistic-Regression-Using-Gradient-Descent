# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages required.
2. Read the dataset.
3. Define X and Y array.
4. Define a function for sigmoid, loss, gradient and predict and perform operations.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: MOHAMED HAMEEM SAJITH J
RegisterNumber:  212223240090

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Placement_Data.csv")

dataset

dataset=dataset.drop("sl_no",axis=1)
dataset=dataset.drop("salary",axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')

dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

dataset

X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values

Y

theta=np.random.randn(X.shape[1])

y=Y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta
    
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
    
y_pred = predict(theta,X)

accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)

print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)

print(y_prednew)
*/
```

## Output:

# 1. Read the file and display:
![image](https://github.com/Sajith7862/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145972360/382d3ef6-d990-40a1-81fb-252828dc25e8)

# 2. Categorizing columns:
![image](https://github.com/Sajith7862/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145972360/0d6926f9-9e3c-41be-8ea0-e22c5295f625)

# 3. Labelling columns and displaying dataset:
![image](https://github.com/Sajith7862/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145972360/53010a9d-ed31-4964-9836-b4df5a095a13)

# 4. Display dependent variable:
![image](https://github.com/Sajith7862/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145972360/1ccba75e-c999-4bdb-8fd9-e43b9407b561)

# 5.Printing accuracy:
![image](https://github.com/Sajith7862/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145972360/c02a9e6b-1039-4d9d-8c94-be09fe4157d9)

# 6. Printing Y:
![image](https://github.com/Sajith7862/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145972360/3fea14d7-503a-4580-b245-5810e454cb13)

# 7. Printing y_prednew:
![image](https://github.com/Sajith7862/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145972360/18992f66-1f3f-455e-8e58-ffbfa2a60aaa)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

