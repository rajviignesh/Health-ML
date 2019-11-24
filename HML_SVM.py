from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


df = pd.read_csv("default2.csv")

feature_cols = ['a', 'b', 'c']
X = df[feature_cols]

y = df['d']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

C = 1.0
model = svm.SVC(kernel='linear', C=C, gamma=1)
#accuracy_result = cross_val_score(model, X, y, cv=4)
#print("Cross Validation Result for ", str(4), " -fold")
#print(accuracy_result * 100)

model.fit(x_train, y_train)
model.score(x_train, y_train)

pred = model.predict(x_test)
accuracy = accuracy_score(y_test, pred)
print(pred, x_test, accuracy)


D = pd.read_csv('test.csv')
feature_cols = ['a', 'b', 'c']
nigga = D[feature_cols]
print(nigga)
print(model.predict(nigga))

# print(Z)
