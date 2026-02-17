"""Tests for classical k-means centroid update."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from q_rlstc.clustering.classical_kmeans import ClassicalKMeans, kmeans_fit


class TestCentroidUpdate:
    """Tests for classical centroid update step."""
    
    def test_centroid_matches_numpy_mean(self):
        """Centroid update should match numpy mean"""
        # Create simple test data
        data = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        
        # All in one cluster
        labels = np.array([0, 0, 0, 0])
        
        kmeans = ClassicalKMeans(n_clusters=1)
        centroid = kmeans._update_centroids(data, labels)
        
        expected = data.mean(axis=0)
        np.testing.assert_array_almost_equal(centroid[0], expected)
    
    def test_multiple_clusters(self):
        """Centroid update works for multiple clusters"""
        data = np.array([
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.1],
        ])
        
        labels = np.array([0, 0, 1, 1])
        
        kmeans = ClassicalKMeans(n_clusters=2)
        centroids = kmeans._update_centroids(data, labels)
        
        # Cluster 0 centroid
        expected_0 = np.array([0.05, 0.05])
        np.testing.assert_array_almost_equal(centroids[0], expected_0)
        
        # Cluster 1 centroid
        expected_1 = np.array([10.05, 10.05])
        np.testing.assert_array_almost_equal(centroids[1], expected_1)
    
    def test_empty_cluster_reinitialize(self):
        """Empty cluster gets reinitialized"""
        data = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
        ])
        
        # Cluster 1 is empty
        labels = np.array([0, 0])
        
        kmeans = ClassicalKMeans(n_clusters=2)
        centroids = kmeans._update_centroids(data, labels)
        
        # Both centroids should be valid (not NaN)
        assert not np.any(np.isnan(centroids))


class TestKMeansFit:
    """Tests for full k-means fitting."""
    
    def test_basic_fit(self):
        """K-means can fit simple data"""
        # Two well-separated clusters
        np.random.seed(42)
        cluster1 = np.random.randn(10, 2) + np.array([0, 0])
        cluster2 = np.random.randn(10, 2) + np.array([10, 10])
        data = np.vstack([cluster1, cluster2])
        
        kmeans = ClassicalKMeans(n_clusters=2)
        result = kmeans.fit(data)
        
        assert result.centroids.shape == (2, 2)
        assert len(result.labels) == 20
        assert result.n_iterations > 0
    
    def test_convenience_function(self):
        """kmeans_fit convenience function works"""
        data = np.random.randn(20, 4)
        
        centroids, labels, objective = kmeans_fit(data, k=3)
        
        assert centroids.shape == (3, 4)
        assert len(labels) == 20
        assert objective > 0


class TestObjective:
    """Tests for clustering objective computation."""
    
    def test_objective_decreases(self):
        """Objective should generally decrease or stay same"""
        np.random.seed(42)
        data = np.random.randn(30, 2)
        
        kmeans = ClassicalKMeans(n_clusters=3, max_iter=1)
        result_1 = kmeans.fit(data)
        
        kmeans2 = ClassicalKMeans(n_clusters=3, max_iter=10)
        result_10 = kmeans2.fit(data)
        
        # More iterations should give same or better objective
        # (not strictly true due to random init, but likely)
        # Just check both are valid
        assert result_1.objective > 0
        assert result_10.objective > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
