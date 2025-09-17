# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values
4. Using logistic regression find the predicted values of accuracy , confusion matrices
5. Display the results.

## Program:
```python
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: RITHIKA L 

RegisterNumber: 212224230231  


import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```


## Output:
## Placement data


<img width="1017" height="853" alt="image" src="https://github.com/user-attachments/assets/9b9e2569-c2b7-4f2b-9788-f2a8cc72bad0" />



## Salary data


<img width="1035" height="330" alt="image" src="https://github.com/user-attachments/assets/6e9177d4-8a79-407f-ac21-be36a24a8943" />



## Checking the null() function


<img width="666" height="472" alt="image" src="https://github.com/user-attachments/assets/61831735-bcb0-45a5-8ed5-687c4e900a32" />


## Data duplicate


<img width="595" height="282" alt="image" src="https://github.com/user-attachments/assets/188f62ca-03ec-4975-9b0e-409f57e4c987" />


## Print data


<img width="1031" height="421" alt="image" src="https://github.com/user-attachments/assets/ff018a94-4175-43df-bfa2-6d5c4ef547bb" />



## Data status


<img width="445" height="718" alt="image" src="https://github.com/user-attachments/assets/560705b4-dadd-4a83-ba87-7cc070e20554" />


## Y-prediction array


<img width="1021" height="360" alt="image" src="https://github.com/user-attachments/assets/7140c333-d414-41b9-93f0-2d6cd8b72f42" />


## Accuracy value


<img width="383" height="75" alt="image" src="https://github.com/user-attachments/assets/80b3ff5a-75e8-4414-acd0-06acc663076b" />


## Confusion array


<img width="594" height="88" alt="image" src="https://github.com/user-attachments/assets/a7f7c815-3695-4c7f-a7f3-c8300c425412" />


## Classification report


<img width="963" height="393" alt="image" src="https://github.com/user-attachments/assets/7076d967-4105-4eb0-93ef-a327eb0a56d8" />


## Prediction of LR


<img width="822" height="306" alt="image" src="https://github.com/user-attachments/assets/5c7228e7-9629-49ba-9161-8b5b122da25f" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
