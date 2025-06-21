# this is an implementation of the K-Means algorithm using Python from scratch

import numpy as np
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('Qt5Agg')
from matplotlib import  pyplot as plt
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import  make_blobs
from sklearn.metrics import silhouette_score,davies_bouldin_score,calinski_harabasz_score
import pandas as pd

# Generate synthetic data
X,_ = make_blobs(n_samples=200,centers=3,cluster_std=1,random_state=42)

class Kmeans:
    def __init__(self,k=3,max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        np.random.seed(42)
        idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[idx]

        for _ in range(self.max_iters):
            labels = self._assign_labels(X)

            new_centroids = []
            for i in range(self.k):
                points = X[labels == i]
                if len(points) == 0:
                    new_centroids.append(self.centroids[i])  # Keep previous if empty cluster
                else:
                    new_centroids.append(points.mean(axis=0))
            new_centroids = np.array(new_centroids)

            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        self.labels = labels

    def _assign_labels(self,X):
        distances = np.linalg.norm(X[:,np.newaxis]- self.centroids,axis=2)
        return np.argmin(distances,axis=1)

#Run kmeans
kmeans = Kmeans(k=3)
kmeans.fit(X)


plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels, cmap='viridis')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='x', s=100)
plt.title("K-Means Clustering (from scratch)")
plt.savefig("kmeans_clusters.png")
print("Plot saved as kmeans_clusters.png")

silhouette = silhouette_score(X, kmeans.labels)
db_score = davies_bouldin_score(X, kmeans.labels)
ch_score = calinski_harabasz_score(X, kmeans.labels)

# Create metrics summary
metrics_df = pd.DataFrame({
    'Metric': ['Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score'],
    'Value': [silhouette, db_score, ch_score]
})

print(metrics_df)