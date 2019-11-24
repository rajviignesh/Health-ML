import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

Data = pd.read_csv('default2.csv')
feature_cols = ['a', 'b', 'c']
X = Data[feature_cols]

y = Data['d']

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#accuracy_result = cross_val_score(logreg, X, y, cv=4)
#print("Cross Validation Result for ", str(4), " -fold")
#print(accuracy_result * 100)

gnb.fit(X_train, y_train)

pred = gnb.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print(pred, X_test, accuracy)


# rez = [[pred[j][i] for j in range(len(pred))] for i in range(len(pred[0]))]
D = pd.read_csv('test.csv')
nigga = D[feature_cols]
print(nigga)
print(gnb.predict(nigga))
