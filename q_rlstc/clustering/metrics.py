"""Clustering quality metrics for evaluation.

Provides metrics for:
- Overall distance (OD) - main reward signal
- Silhouette score - cluster separation quality
- Segmentation F1 - accuracy vs ground truth boundaries
"""

import numpy as np
from typing import List, Set, Tuple, Optional


def overall_distance(
    data: np.ndarray,
    centroids: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute overall distance (OD) metric.
    
    OD = sqrt(mean(||x_i - c_{y_i}||^2))
    
    Lower is better.
    
    Args:
        data: Data points (n x d).
        centroids: Cluster centroids (k x d).
        labels: Cluster assignments (n,).
    
    Returns:
        Overall distance.
    """
    data = np.asarray(data)
    centroids = np.asarray(centroids)
    labels = np.asarray(labels)
    
    total_sq_dist = 0.0
    for i, (point, label) in enumerate(zip(data, labels)):
        dist = np.linalg.norm(point - centroids[label])
        total_sq_dist += dist ** 2
    
    return np.sqrt(total_sq_dist / len(data))


def silhouette_score(
    data: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute silhouette score for clustering quality.
    
    Measures how similar points are to their own cluster vs other clusters.
    Range: [-1, 1], higher is better.
    
    Args:
        data: Data points (n x d).
        labels: Cluster assignments.
    
    Returns:
        Mean silhouette coefficient.
    """
    data = np.asarray(data)
    labels = np.asarray(labels)
    n_samples = len(data)
    n_clusters = len(np.unique(labels))
    
    if n_clusters <= 1 or n_clusters >= n_samples:
        return 0.0
    
    silhouette_values = np.zeros(n_samples)
    
    for i in range(n_samples):
        cluster_i = labels[i]
        same_cluster = data[labels == cluster_i]
        
        # a(i) = mean distance to same cluster
        if len(same_cluster) > 1:
            a_i = np.mean([np.linalg.norm(data[i] - x) for x in same_cluster if not np.array_equal(x, data[i])])
        else:
            a_i = 0.0
        
        # b(i) = min mean distance to other clusters
        b_i = np.inf
        for cluster_j in np.unique(labels):
            if cluster_j != cluster_i:
                other_cluster = data[labels == cluster_j]
                if len(other_cluster) > 0:
                    mean_dist = np.mean([np.linalg.norm(data[i] - x) for x in other_cluster])
                    b_i = min(b_i, mean_dist)
        
        if b_i == np.inf:
            b_i = 0.0
        
        # Silhouette coefficient
        if max(a_i, b_i) > 0:
            silhouette_values[i] = (b_i - a_i) / max(a_i, b_i)
        else:
            silhouette_values[i] = 0.0
    
    return np.mean(silhouette_values)


def _boundary_set(boundaries: List[int], tolerance: int = 1) -> Set[int]:
    """Convert boundary list to set with tolerance.
    
    Args:
        boundaries: List of boundary indices.
        tolerance: How many indices off to still count as match.
    
    Returns:
        Set of boundary indices with tolerance extended.
    """
    result: Set[int] = set()
    for b in boundaries:
        for t in range(-tolerance, tolerance + 1):
            result.add(b + t)
    return result


def segmentation_f1(
    predicted_boundaries: List[int],
    true_boundaries: List[int],
    tolerance: int = 1,
) -> Tuple[float, float, float]:
    """Compute F1 score for segmentation boundaries.
    
    Args:
        predicted_boundaries: Predicted split points.
        true_boundaries: Ground truth split points.
        tolerance: Index tolerance for boundary matching.
    
    Returns:
        Tuple of (precision, recall, f1).
    """
    if len(true_boundaries) == 0 and len(predicted_boundaries) == 0:
        return 1.0, 1.0, 1.0
    
    if len(true_boundaries) == 0:
        return 0.0, 1.0, 0.0
    
    if len(predicted_boundaries) == 0:
        return 1.0, 0.0, 0.0
    
    # Count true positives
    true_set = _boundary_set(true_boundaries, tolerance)
    pred_set = set(predicted_boundaries)
    
    tp = len([p for p in pred_set if p in true_set])
    
    precision = tp / len(predicted_boundaries)
    recall = tp / len(true_boundaries)
    
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return precision, recall, f1


def incremental_od_update(
    current_od: float,
    n_segments: int,
    new_segment_cost: float,
) -> float:
    """Incrementally update OD when adding a new segment.
    
    Used for efficient reward computation during RL.
    
    Args:
        current_od: Current overall distance.
        n_segments: Current number of segments.
        new_segment_cost: Cost (distance) of new segment.
    
    Returns:
        Updated overall distance.
    """
    if n_segments == 0:
        return new_segment_cost
    
    # Running average update
    total = current_od * n_segments + new_segment_cost
    return total / (n_segments + 1)


def od_improvement_reward(
    od_before: float,
    od_after: float,
    scale: float = 1.0,
) -> float:
    """Compute reward from OD improvement.
    
    Reward is positive when OD decreases (better clustering).
    
    Args:
        od_before: OD before action.
        od_after: OD after action.
        scale: Scaling factor.
    
    Returns:
        Reward signal.
    """
    improvement = od_before - od_after
    return improvement * scale
