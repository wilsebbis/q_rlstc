"""Clustering quality metrics for evaluation.

Provides metrics for:
- Overall distance (OD) - main reward signal
- Silhouette score - cluster separation quality
- Segmentation F1 - accuracy vs ground truth boundaries
"""

import numpy as np
from typing import List, Set, Tuple, Optional
from fastdtw import fastdtw
import similaritymeasures
from scipy.spatial.distance import euclidean


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


def weighted_valcr(
    per_segment_ods: List[float],
    per_segment_lengths: List[int],
    basesim: float,
    epsilon: float = 1e-8,
) -> float:
    """Length-weighted Validation Competitive Ratio (wValCR).

    Standard ValCR averages per-segment CRs equally, which creates a
    fragmentation attractor: many short segments each contribute low
    distance but equal weight. wValCR reweights by segment length so
    long, semantically meaningful segments dominate the score.

        wValCR = Σ (len_i / Σ len_j) · (OD_i / basesim)

    Lower is better.  When all segments have equal length this reduces
    to standard ValCR.

    Args:
        per_segment_ods: OD (distance to nearest centroid) per segment.
        per_segment_lengths: Number of GPS points in each segment.
        basesim: Fold-specific baseline OD (always-extend OD).
        epsilon: Floor for basesim to prevent divide-by-zero.

    Returns:
        Length-weighted competitive ratio.
    """
    if not per_segment_ods or not per_segment_lengths:
        return float("inf")

    ods = np.asarray(per_segment_ods, dtype=float)
    lengths = np.asarray(per_segment_lengths, dtype=float)

    if len(ods) != len(lengths):
        raise ValueError(
            f"Length mismatch: {len(ods)} ODs vs {len(lengths)} segment lengths"
        )

    total_length = lengths.sum()
    if total_length == 0:
        return float("inf")

    weights = lengths / total_length  # w_i = len_i / Σ len_j
    safe_basesim = max(float(basesim), epsilon)
    per_seg_cr = ods / safe_basesim  # CR_i = OD_i / basesim

    return float(np.dot(weights, per_seg_cr))


def random_policy_advantage(
    agent_valcr: float,
    random_valcr: float,
) -> float:
    """Random-policy advantage at matched CUT budget.

        Δ_rand = random_ValCR - agent_ValCR

    Positive values indicate the agent outperforms random at the same
    CUT budget — evidence that the agent learned meaningful segment
    placement rather than exploiting the length-sensitivity degeneracy.

    Convention: lower ValCR = better, so positive Δ_rand = agent wins.

    Args:
        agent_valcr: Agent's ValCR at the target CUT budget.
        random_valcr: Random policy's ValCR at the same CUT budget.

    Returns:
        Advantage delta (positive = agent outperforms random).
    """
    return random_valcr - agent_valcr

def compute_dtw_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """Compute Dynamic Time Warping distance between two trajectories."""
    if len(traj1) == 0 or len(traj2) == 0:
        return 0.0
    distance, _ = fastdtw(traj1, traj2, dist=euclidean)
    return float(distance)

def compute_frechet_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """Compute Discrete Fréchet distance between two trajectories."""
    if len(traj1) == 0 or len(traj2) == 0:
        return 0.0
    return float(similaritymeasures.frechet_dist(traj1, traj2))

def evaluate_standard_metrics(clusters_e: dict) -> Tuple[float, float]:
    """Compute standard trajectory metrics (Avg DTW, Avg Fréchet).
    
    Args:
        clusters_e: Dict mapping centroid index to list:
                    [0]: list of distances
                    [1]: list of sub-trajectories (Traj objects)
                    [2]: center trajectory (List[Point])
                    [3]: time dict
                    [4]: segment lengths
    
    Returns:
        (mean_dtw, mean_frechet)
    """
    total_dtw = 0.0
    total_frechet = 0.0
    count = 0
    
    for c_idx, cluster_data in clusters_e.items():
        sub_trajs = cluster_data[1]  # List[Traj]
        center = cluster_data[2]     # List[Point]
        
        if not sub_trajs or not center:
            continue
            
        center_pts = np.array([[p.x, p.y] for p in center])
        if len(center_pts) == 0:
            continue
            
        for traj in sub_trajs:
            traj_pts = np.array([[p.x, p.y] for p in traj.points])
            if len(traj_pts) == 0:
                continue
                
            total_dtw += compute_dtw_distance(traj_pts, center_pts)
            total_frechet += compute_frechet_distance(traj_pts, center_pts)
            count += 1
            
    if count == 0:
        return 0.0, 0.0
        
    return total_dtw / count, total_frechet / count

def compute_mdl_cost(clusters_e: dict) -> float:
    """Compute Simplified Minimum Description Length (MDL) for a trajectory clustering.
    
    Acts as a defensive standard against ValCR fragmentation degeneracy by intrinsically 
    penalizing excessive representative points.
    
    L(H) = Sum of path lengths of all centroids.
    L(D|H) = Sum of distances from each point in sub-trajectories to the centroid.
    MDL = L(H) + L(D|H)
    """
    l_h = 0.0
    l_dh = 0.0
    
    for cid, cluster_data in clusters_e.items():
        sub_trajs = cluster_data[1]  # List[Traj]
        center = cluster_data[2]     # List[Point]
        
        if not sub_trajs or not center:
            continue
            
        # calculate length of cluster center trajectory L(H)
        if len(center) >= 2:
            pts = np.array([[p.x, p.y] for p in center])
            l_h += np.sum(np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1)))
            
        # L(D|H) is captured by the total Over-Distance (Euclidean errors)
        dists = cluster_data[0]
        l_dh += sum(dists)
        
    return l_h + l_dh

def compute_mhd_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """Compute Modified Hausdorff Distance between two trajectories.
    Robust to local noise and inherently worst-case deviation bounding.
    """
    if len(traj1) == 0 or len(traj2) == 0:
        return 0.0
        
    # Directed Hausdorff distance h(traj1, traj2) and h(traj2, traj1)
    from scipy.spatial.distance import directed_hausdorff
    forward_hd = directed_hausdorff(traj1, traj2)[0]
    backward_hd = directed_hausdorff(traj2, traj1)[0]
    
    return max(forward_hd, backward_hd)

def evaluate_advanced_metrics(clusters_e: dict) -> Tuple[float, float]:
    """Compute advanced robust trajectory metrics (MDL, Mean MHD).
    
    Args:
        clusters_e: Dict mapping centroid index to cluster data structure.
        
    Returns:
        (total_mdl, mean_mhd)
    """
    total_mdl = compute_mdl_cost(clusters_e)
    
    total_mhd = 0.0
    count = 0
    
    for c_idx, cluster_data in clusters_e.items():
        sub_trajs = cluster_data[1]  # List[Traj]
        center = cluster_data[2]     # List[Point]
        
        if not sub_trajs or not center:
            continue
            
        center_pts = np.array([[p.x, p.y] for p in center])
        if len(center_pts) == 0:
            continue
            
        for traj in sub_trajs:
            traj_pts = np.array([[p.x, p.y] for p in traj.points])
            if len(traj_pts) == 0:
                continue
                
            total_mhd += compute_mhd_distance(traj_pts, center_pts)
            count += 1
            
    mean_mhd = (total_mhd / count) if count > 0 else 0.0
    
    return total_mdl, mean_mhd

