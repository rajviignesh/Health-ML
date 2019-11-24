import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

Data = pd.read_csv('default2.csv')
feature_cols = ['a', 'b', 'c']
X = Data[feature_cols]

y = Data['d']

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#accuracy_result = cross_val_score(logreg, X, y, cv=4)
#print("Cross Validation Result for ", str(4), " -fold")
#print(accuracy_result * 100)

logreg.fit(X_train, y_train)

pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print(pred, X_test, accuracy)
parameters = logreg.coef_

# rez = [[pred[j][i] for j in range(len(pred))] for i in range(len(pred[0]))]
D = pd.read_csv('test.csv')
nigga = D[feature_cols]
print(nigga)
print(logreg.predict(nigga))

# x_values = [0, 150]
# y_values = - (parameters[1][0] + np.dot(parameters[1][1], x_values)) / parameters[1][2]

# plt.scatter(X_test['a'], X_test['c'], X_test['b'], c='g')
# plt.scatter(y_test, s=10, c='g')
# plt.plot(x_values, y_values, label='Decision Boundary', c='r')
# plt.xlabel('Marks in 1st Exam')
# plt.ylabel('Marks in 2nd Exam')
# plt.legend()
# plt.show()
plt.show()
