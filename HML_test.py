from sklearn import tree
import numpy as np
import pandas as pd

Data = pd.read_csv('default2.csv')
feature_cols = ['a', 'b', 'c']
X = Data[feature_cols]
y = Data['d']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

from IPython.display import Image
from sklearn.externals.six import StringIO
import pydotplus

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=feature_cols)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
graph.write_png("iris.png")
