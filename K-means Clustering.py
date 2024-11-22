import matplotlib.pyplot as plt from sklearn import datasets
from sklearn.cluster import KMeans iris = datasets.load_iris()
X = iris.data y = iris.target
plt.scatter(X[:,1], X[:,3], color='white', marker='o', edgecolor='red', s=50) plt.grid()
plt.show()
 
 



kmc = KMeans(n_clusters=3) y_kmc = kmc.fit_predict(X)
plt.scatter(X[y_kmc == 0, 1], X[y_kmc == 0, 3], s=50,c='lightgreen', marker='s',
edgecolor='black', label='Cluster 1')
plt.scatter(X[y_kmc == 1, 1], X[y_kmc == 1, 3], s=50,c='orange', marker='o',
edgecolor='black', label='Cluster 2')
plt.scatter(X[y_kmc == 2, 1], X[y_kmc == 2, 3], s=50,c='blue', marker='P',
edgecolor='black', label='Cluster 3')
plt.scatter(kmc.cluster_centers_[:, 1], kmc.cluster_centers_[:, 3],s=250, marker='*', c='red', edgecolor='black', label='Centroids')
plt.legend() plt.grid() plt.show()
 
