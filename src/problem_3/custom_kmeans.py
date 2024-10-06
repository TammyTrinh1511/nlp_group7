from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


class CustomKMeans(KMeans):
    """
    A Custom KMeans class that supports cosine similarity as a distance metric.

    This class extends the sklearn KMeans class to allow for clustering using cosine distance
    in addition to the standard Euclidean distance. Users can specify the distance metric when
    initializing the class.

    Attributes:
        metric (str): The distance metric to use ('euclidean' or 'cosine').
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: str = "random",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: bool = 0,
        random_state: int = None,
        copy_x: bool = True,
        metric: str = "euclidean",
    ):
        """
        Initialize the CustomKMeans instance.

        Args:
            n_clusters (int): The number of clusters to form.
            init (str): Method for initialization ('random' or 'k-means++').
            n_init (int): Number of time run with different centroid seeds.
            max_iter (int): Maximum number of iterations of algorithm.
            tol (float): Relative tolerance to declare convergence.
            verbose (bool): Verbosity mode.
            random_state (int): Random number generation.
            copy_x (bool): If False, it may overwrite the input data.
            metric (str): The distance metric to use ('euclidean' or 'cosine').
        """
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            copy_x=copy_x,
        )
        self.metric = metric

    def _e_step(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Perform the expectation step of the KMeans algorithm.

        Args:
            X (np.ndarray): The input data to cluster.

        Returns:
            Tuple[np.ndarray, float]: A tuple containing:
                - labels (np.ndarray): The labels of each point.
                - inertia (float): Sum of squared distances to nearest cluster center.
        """
        if self.metric == "cosine":
            distances = 1 - cosine_similarity(X, self.cluster_centers_)
        else:
            distances = euclidean_distances(X, self.cluster_centers_)
        labels = np.argmin(distances, axis=1)
        inertia = np.sum(np.min(distances, axis=1) ** 2)
        return labels, inertia

    def fit(self, X: np.ndarray, y=None, sample_weight=None) -> "CustomKMeans":
        """
        Compute k-means clustering.

        Args:
            X (np.ndarray): The input data to cluster.
            y (ignored): Not used, present here for API consistency by convention.
            sample_weight (optional): Weight for each sample.

        Returns:
            CustomKMeans: The instance of the CustomKMeans object fitted to the data.
        """
        super().fit(X, y, sample_weight)
        self.labels_, self.inertia_ = self._e_step(X)
        return self
