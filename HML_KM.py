import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

data = pd.read_csv("default2.csv")

df = data[data.columns[0:3]]
kmeans = KMeans(n_clusters=3)


y = data.d
X = df

accuracy_result = cross_val_score(kmeans, X, y, cv=4)
print("Cross Validation Result for ", str(4), " -fold")
print(accuracy_result * 100)

kmeans.fit(df)

labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_

fig = plt.figure(figsize=(5, 5))
colmap = {1: 'r', 2: 'g', 3: 'b'}

colors = map(lambda x: colmap[x + 1], labels)

plt.scatter(df['a'], df['b'], df['c'], color=list(colors), alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx + 1])

plt.show()
