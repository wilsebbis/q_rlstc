"""Classical k-means clustering.

Pure classical implementation for trajectory segment clustering.
Used for episode-end evaluation, NOT per-step in RL loop.
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class KMeansResult:
    """Result from k-means clustering.
    
    Attributes:
        centroids: Final cluster centroids (k x d).
        labels: Cluster assignment for each point.
        objective: Final overall distance.
        n_iterations: Number of iterations used.
        converged: Whether algorithm converged.
    """
    centroids: np.ndarray
    labels: np.ndarray
    objective: float
    n_iterations: int
    converged: bool


class ClassicalKMeans:
    """Classical k-means clustering using Euclidean distance.
    
    Standard Lloyd's algorithm for clustering trajectory segments.
    """
    
    def __init__(
        self,
        n_clusters: int = 10,
        max_iter: int = 50,
        convergence_threshold: float = 0.01,
        seed: int = 42,
    ):
        """Initialize classical k-means.
        
        Args:
            n_clusters: Number of clusters (k).
            max_iter: Maximum iterations.
            convergence_threshold: Stop when centroid shift < this.
            seed: Random seed for initialization.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.convergence_threshold = convergence_threshold
        self.rng = np.random.default_rng(seed)
        
        self.centroids_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
    
    def _distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Euclidean distance between two vectors."""
        return float(np.linalg.norm(x - y))
    
    def _distance_matrix(
        self, 
        data: np.ndarray, 
        centroids: np.ndarray
    ) -> np.ndarray:
        """Compute distances from all points to all centroids.
        
        Args:
            data: Data points (n x d).
            centroids: Centroids (k x d).
        
        Returns:
            Distance matrix (n x k).
        """
        # Vectorized for efficiency
        n_samples = len(data)
        n_centroids = len(centroids)
        
        # (n, 1, d) - (1, k, d) -> (n, k, d) -> (n, k)
        diff = data[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        
        return distances
    
    def _assign_clusters(
        self, 
        data: np.ndarray, 
        centroids: np.ndarray
    ) -> np.ndarray:
        """Assign each point to nearest centroid.
        
        Args:
            data: Data points (n x d).
            centroids: Centroids (k x d).
        
        Returns:
            Cluster assignments (n,).
        """
        distances = self._distance_matrix(data, centroids)
        return np.argmin(distances, axis=1)
    
    def _update_centroids(
        self, 
        data: np.ndarray, 
        labels: np.ndarray
    ) -> np.ndarray:
        """Update centroids as mean of assigned points.
        
        Args:
            data: Data points (n x d).
            labels: Cluster assignments.
        
        Returns:
            New centroids (k x d).
        """
        new_centroids = np.zeros((self.n_clusters, data.shape[1]))
        
        for k in range(self.n_clusters):
            mask = labels == k
            if np.any(mask):
                new_centroids[k] = data[mask].mean(axis=0)
            else:
                # Empty cluster: reinitialize randomly
                new_centroids[k] = data[self.rng.integers(len(data))]
        
        return new_centroids
    
    def _compute_objective(
        self, 
        data: np.ndarray, 
        centroids: np.ndarray, 
        labels: np.ndarray
    ) -> float:
        """Compute overall distance (sum of distances to centroids).
        
        Args:
            data: Data points.
            centroids: Cluster centroids.
            labels: Cluster assignments.
        
        Returns:
            Total distance.
        """
        total_sq = 0.0
        for i, (point, label) in enumerate(zip(data, labels)):
            total_sq += np.linalg.norm(point - centroids[label]) ** 2
        return float(np.sqrt(total_sq / len(data)))
    
    def _initialize_centroids(self, data: np.ndarray) -> np.ndarray:
        """Initialize centroids using k-means++ style.
        
        Args:
            data: Data points.
        
        Returns:
            Initial centroids.
        """
        n_samples = len(data)
        centroids = [data[self.rng.integers(n_samples)]]
        
        for _ in range(1, self.n_clusters):
            # Compute distance to nearest centroid for each point
            distances = np.array([
                min(self._distance(point, c) for c in centroids)
                for point in data
            ])
            
            # Sample with probability proportional to distance^2
            probs = distances ** 2
            probs_sum = probs.sum()
            if probs_sum > 0:
                probs = probs / probs_sum
            else:
                probs = np.ones(n_samples) / n_samples
            
            new_idx = self.rng.choice(n_samples, p=probs)
            centroids.append(data[new_idx])
        
        return np.array(centroids)
    
    def fit(self, data: np.ndarray) -> KMeansResult:
        """Fit k-means to data.
        
        Args:
            data: Data points (n x d).
        
        Returns:
            KMeansResult with final clustering.
        """
        data = np.asarray(data)
        if len(data) < self.n_clusters:
            raise ValueError(f"Need at least {self.n_clusters} samples")
        
        # Initialize
        centroids = self._initialize_centroids(data)
        labels = self._assign_clusters(data, centroids)
        
        converged = False
        iteration = 0
        for iteration in range(self.max_iter):
            # Update centroids
            new_centroids = self._update_centroids(data, labels)
            
            # Check convergence
            centroid_shift = np.max([
                np.linalg.norm(new_centroids[k] - centroids[k])
                for k in range(self.n_clusters)
            ])
            
            centroids = new_centroids
            
            if centroid_shift < self.convergence_threshold:
                converged = True
                break
            
            # Reassign
            labels = self._assign_clusters(data, centroids)
        
        # Compute final objective
        objective = self._compute_objective(data, centroids, labels)
        
        self.centroids_ = centroids
        self.labels_ = labels
        
        return KMeansResult(
            centroids=centroids,
            labels=labels,
            objective=objective,
            n_iterations=iteration + 1,
            converged=converged,
        )
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for new data.
        
        Args:
            data: Data points (n x d).
        
        Returns:
            Cluster assignments.
        """
        if self.centroids_ is None:
            raise ValueError("Must call fit() first")
        
        return self._assign_clusters(data, self.centroids_)


def kmeans_fit(
    data: np.ndarray,
    k: int,
    max_iter: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Convenience function for k-means clustering.
    
    Args:
        data: Data points (n x d).
        k: Number of clusters.
        max_iter: Maximum iterations.
        seed: Random seed.
    
    Returns:
        Tuple of (centroids, labels, objective).
    """
    kmeans = ClassicalKMeans(
        n_clusters=k,
        max_iter=max_iter,
        seed=seed,
    )
    result = kmeans.fit(data)
    return result.centroids, result.labels, result.objective
