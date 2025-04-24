import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

class ClusterVisualizer:
    def __init__(self, method="pca", n_components=2):
        self.method = method
        self.n_components = n_components

    def reduce_dimensions(self, vectors):
        if self.method == "tsne":
            reducer = TSNE(n_components=self.n_components, random_state=42, perplexity=30)
        else:
            reducer = PCA(n_components=self.n_components, random_state=42)

        return reducer.fit_transform(vectors)

    def plot_clusters(self, embeddings, labels, title="Cluster Visualization"):
        reduced = self.reduce_dimensions(embeddings)

        plt.figure(figsize=(10, 8))
        unique_labels = set(labels)

        for label in unique_labels:
            indices = np.where(labels == label)
            plt.scatter(reduced[indices, 0], reduced[indices, 1], label=f"Cluster {label}", s=30, alpha=0.6)

        plt.title(title)
        plt.legend()
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
