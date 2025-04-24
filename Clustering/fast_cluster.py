# Clustering/fast_cluster.py
from sklearn.cluster import DBSCAN
import numpy as np

class FastCluster:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='cosine')

    def fit(self, embeddings):
        return self.model.fit_predict(embeddings)