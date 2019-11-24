import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


Data = pd.read_csv('default2.csv')
feature_cols = ['a', 'b', 'c']
X = Data[feature_cols]
y = Data['d']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = RandomForestClassifier(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)

pred = regressor.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print(accuracy * 100)

D = pd.read_csv('test.csv')
nigga = D[feature_cols]
print(nigga)
print(regressor.predict(nigga))
