"""Tests for clustering metrics.

Verifies OD computation, silhouette score range, segmentation F1,
incremental OD update, and edge cases.
"""

import numpy as np
import pytest

from q_rlstc.clustering.metrics import (
    overall_distance,
    silhouette_score,
    segmentation_f1,
    incremental_od_update,
    od_improvement_reward,
)


class TestOverallDistance:
    """Tests for overall_distance metric."""

    def test_perfect_clustering(self):
        """OD should be 0 when all points are at their centroids."""
        data = np.array([[1.0, 0.0], [0.0, 1.0]])
        centroids = np.array([[1.0, 0.0], [0.0, 1.0]])
        labels = np.array([0, 1])
        od = overall_distance(data, centroids, labels)
        assert od == pytest.approx(0.0, abs=1e-10)

    def test_known_distance(self):
        """OD should match manual calculation."""
        data = np.array([[0.0], [2.0]])
        centroids = np.array([[1.0]])  # one cluster centroid at 1.0
        labels = np.array([0, 0])
        # dist(0, 1) = 1, dist(2, 1) = 1
        # OD = sqrt(mean(1^2 + 1^2)) = sqrt(1) = 1.0
        od = overall_distance(data, centroids, labels)
        assert od == pytest.approx(1.0, abs=1e-10)

    def test_positive(self):
        """OD should always be non-negative."""
        np.random.seed(42)
        data = np.random.randn(50, 3)
        centroids = np.random.randn(3, 3)
        labels = np.random.randint(0, 3, size=50)
        od = overall_distance(data, centroids, labels)
        assert od >= 0


class TestSilhouetteScore:
    """Tests for silhouette_score metric."""

    def test_range(self):
        """Silhouette score should be in [-1, 1]."""
        np.random.seed(42)
        data = np.random.randn(30, 2)
        labels = np.random.randint(0, 3, size=30)
        score = silhouette_score(data, labels)
        assert -1.0 <= score <= 1.0

    def test_well_separated_clusters(self):
        """Well-separated clusters should have positive silhouette."""
        cluster_a = np.random.randn(20, 2) + np.array([10, 10])
        cluster_b = np.random.randn(20, 2) + np.array([-10, -10])
        data = np.vstack([cluster_a, cluster_b])
        labels = np.array([0] * 20 + [1] * 20)
        score = silhouette_score(data, labels)
        assert score > 0

    def test_single_cluster_returns_zero(self):
        """Single cluster should return 0."""
        data = np.random.randn(10, 2)
        labels = np.zeros(10, dtype=int)
        score = silhouette_score(data, labels)
        assert score == 0.0


class TestSegmentationF1:
    """Tests for segmentation_f1 metric."""

    def test_perfect_match(self):
        """Perfect prediction should give F1 = 1.0."""
        pred = [5, 10, 15]
        true = [5, 10, 15]
        p, r, f1 = segmentation_f1(pred, true, tolerance=0)
        assert f1 == pytest.approx(1.0)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)

    def test_tolerance_match(self):
        """Predictions within tolerance should still match."""
        pred = [6, 11, 14]
        true = [5, 10, 15]
        p, r, f1 = segmentation_f1(pred, true, tolerance=1)
        assert f1 == pytest.approx(1.0)

    def test_no_predictions(self):
        """No predictions should give F1 = 0."""
        _, _, f1 = segmentation_f1([], [5, 10], tolerance=1)
        assert f1 == 0.0

    def test_no_ground_truth(self):
        """No ground truth should give F1 = 0."""
        _, _, f1 = segmentation_f1([5, 10], [], tolerance=1)
        assert f1 == 0.0

    def test_both_empty(self):
        """Both empty should give F1 = 1.0 (vacuous truth)."""
        p, r, f1 = segmentation_f1([], [], tolerance=1)
        assert f1 == 1.0


class TestIncrementalOD:
    """Tests for incremental_od_update."""

    def test_first_segment(self):
        """First segment should just return its cost."""
        result = incremental_od_update(0.0, 0, 2.5)
        assert result == pytest.approx(2.5)

    def test_running_average(self):
        """Should compute running average correctly."""
        # Start with OD=3.0 over 2 segments, add segment with cost 0.0
        result = incremental_od_update(3.0, 2, 0.0)
        # (3.0 * 2 + 0.0) / 3 = 2.0
        assert result == pytest.approx(2.0)


class TestODImprovementReward:
    """Tests for od_improvement_reward."""

    def test_positive_improvement(self):
        """OD decrease should give positive reward."""
        reward = od_improvement_reward(5.0, 3.0)
        assert reward > 0

    def test_negative_improvement(self):
        """OD increase should give negative reward."""
        reward = od_improvement_reward(3.0, 5.0)
        assert reward < 0

    def test_no_change(self):
        """No OD change should give zero reward."""
        reward = od_improvement_reward(3.0, 3.0)
        assert reward == pytest.approx(0.0)

    def test_scaling(self):
        """Scale factor should multiply reward."""
        r1 = od_improvement_reward(5.0, 3.0, scale=1.0)
        r2 = od_improvement_reward(5.0, 3.0, scale=2.0)
        assert r2 == pytest.approx(2 * r1)
