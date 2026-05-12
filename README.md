# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Hardware – PCs

2.Load Iris dataset and separate features (X) and target (Y) 

3.Split the dataset into training and testing sets 

4.Initialize the SGD Classifier model 5.Train the model using training data

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Kervin.S
RegisterNumber:212225220051

from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data
Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

model = SGDClassifier(max_iter=1000, learning_rate='optimal')
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(Y_test, Y_pred))

sample = [X[0]]
prediction = model.predict(sample)

print("Predicted Species:", iris.target_names[prediction][0]) 
*/
```

## Output:
<img width="291" height="49" alt="image" src="https://github.com/user-attachments/assets/94d8fd6f-7e40-47a9-8b30-538bd8c6261f" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
