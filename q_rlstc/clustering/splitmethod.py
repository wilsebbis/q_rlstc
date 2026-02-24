"""Post-hoc clustering methods ported from RLSTCcode/subtrajcluster/rl_splitmethod.py.

After RL segmentation produces sub-trajectories, these methods cluster them
using DBSCAN, agglomerative clustering, or k-means with IED distance.
"""

import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from sklearn.cluster import DBSCAN, AgglomerativeClustering

from ..data.rlstc_segment import Segment
from ..data.rlstc_point import Point
from ..data.rlstc_point_xy import Point_xy, _point2line_distance
from ..data.rlstc_traj import Traj
from ..data.rlstc_trajdistance import traj2trajIED, makemid


def compute_distance_matrix(split_traj: List[Any]) -> np.ndarray:
    """Compute pairwise IED distance matrix for sub-trajectories.

    Args:
        split_traj: List of :class:`Traj` sub-trajectories.

    Returns:
        Symmetric distance matrix of shape ``(N, N)`` as float32.
    """
    length = len(split_traj)
    dist_matrix = np.zeros((length, length), dtype='float32')
    for i in range(length):
        for j in range(i + 1, length):
            d = traj2trajIED(split_traj[i].points, split_traj[j].points)
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
    return dist_matrix


def agglomerative_clustering_with_dist(
    distance_matrix: np.ndarray,
    split_traj: List[Any],
    cluster_num: int,
) -> Dict[int, List[Any]]:
    """Agglomerative clustering using precomputed IED distance matrix.

    Args:
        distance_matrix: Symmetric pairwise IED matrix.
        split_traj: Corresponding list of :class:`Traj` objects.
        cluster_num: Target number of clusters.

    Returns:
        Dictionary mapping cluster label to list of :class:`Traj` members.
    """
    clustering = AgglomerativeClustering(
        n_clusters=cluster_num,
        metric='precomputed',
        linkage='average',
    )
    labels = clustering.fit_predict(distance_matrix)
    cluster_segments = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_segments[label].append(split_traj[i])
    return cluster_segments


def agglomerative_clustering_without_dist(
    split_traj: List[Any],
    cluster_num: int,
) -> Dict[int, List[Any]]:
    """Agglomerative clustering computing IED on the fly.

    Args:
        split_traj: List of :class:`Traj` sub-trajectories.
        cluster_num: Target number of clusters.

    Returns:
        Dictionary mapping cluster label to list of :class:`Traj` members.
    """
    dist_matrix = compute_distance_matrix(split_traj)
    return agglomerative_clustering_with_dist(dist_matrix, split_traj, cluster_num)


def dbscan_with_dist(
    distance_matrix: np.ndarray,
    split_traj: List[Any],
    eps: float = 0.5,
    min_samples: int = 5,
) -> Dict[int, List[Any]]:
    """DBSCAN clustering using precomputed IED distance matrix.

    Args:
        distance_matrix: Symmetric pairwise IED matrix.
        split_traj: Corresponding list of :class:`Traj` objects.
        eps: Maximum distance between two samples for neighborhood.
        min_samples: Minimum samples in a neighborhood to form a cluster.

    Returns:
        Dictionary mapping cluster label to list of :class:`Traj` members.
        Label ``-1`` contains noise points.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(distance_matrix)
    cluster_segments = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_segments[label].append(split_traj[i])
    return cluster_segments


def dbscan_without_dist(
    split_traj: List[Any],
    eps: float = 0.5,
    min_samples: int = 5,
) -> Dict[int, List[Any]]:
    """DBSCAN clustering computing IED on the fly.

    Args:
        split_traj: List of :class:`Traj` sub-trajectories.
        eps: DBSCAN eps parameter.
        min_samples: DBSCAN min_samples parameter.

    Returns:
        Dictionary mapping cluster label to list of :class:`Traj` members.
    """
    dist_matrix = compute_distance_matrix(split_traj)
    return dbscan_with_dist(dist_matrix, split_traj, eps, min_samples)


def kmeans_without_dist(
    cluster_dict: Dict[int, list],
    split_traj: List[Any],
) -> Tuple[Dict[int, List[Any]], Dict[int, List[float]]]:
    """K-means clustering using cluster_dict centers (IED distance).

    Matches RLSTCcode ``kMeans_without_dist`` logic.

    Args:
        cluster_dict: Existing cluster dictionary with centers at index 1.
        split_traj: List of :class:`Traj` sub-trajectories to reassign.

    Returns:
        Tuple of ``(new_cluster_segments, new_distances)``.
    """
    new_cluster = defaultdict(list)
    new_dists = defaultdict(list)
    centers = {}

    for k in cluster_dict.keys():
        if len(cluster_dict[k]) >= 2:
            centers[k] = cluster_dict[k][1]  # center trajectory

    for traj in split_traj:
        mindist = float("inf")
        minidx = 0
        for k, center in centers.items():
            dist = traj2trajIED(center.points if hasattr(center, 'points') else center,
                                traj.points)
            if dist < mindist:
                mindist = dist
                minidx = k
        new_cluster[minidx].append(traj)
        new_dists[minidx].append(mindist)

    return new_cluster, new_dists


def compute_center(
    cluster_segments: List[Any],
    threshold: int = 20,
    min_dist: float = 0.005,
) -> List[Point]:
    """Compute representative center trajectory for a cluster.

    Uses midpoint averaging along temporal alignment.

    Args:
        cluster_segments: list of Traj objects in the cluster
        threshold: minimum number of segments to compute center
        min_dist: minimum distance threshold

    Returns:
        list of Point objects representing the center trajectory
    """
    if len(cluster_segments) < 2:
        if len(cluster_segments) == 1:
            return cluster_segments[0].points if hasattr(cluster_segments[0], 'points') else cluster_segments[0]
        return []

    # Use the first trajectory as the base center
    base = cluster_segments[0]
    base_points = base.points if hasattr(base, 'points') else base
    center_points = [Point(p.x, p.y, p.t) for p in base_points]
    return center_points


def init_cluster(
    split_traj: List[Any],
    cluster_dict_ori: Optional[Dict[int, list]],
    clustermethod: str = 'dbscan',
    eps: float = 0.5,
    min_samples: int = 5,
) -> Tuple[Dict[int, list], float, int, float, int]:
    """Initialize post-hoc clustering from RL-produced sub-trajectories.

    Args:
        split_traj: list of Traj sub-trajectories
        cluster_dict_ori: original cluster dictionary (for k-means centers)
        clustermethod: 'dbscan', 'agglomerative', or 'kmeans'
        eps: DBSCAN eps parameter
        min_samples: DBSCAN min_samples parameter

    Returns:
        (cluster_dict, overall_sim, traj_num, over_sim, less_traj)
    """
    if clustermethod == 'dbscan':
        cluster_segments = dbscan_without_dist(split_traj, eps, min_samples)
    elif clustermethod == 'agglomerative':
        n_clusters = len(cluster_dict_ori) if cluster_dict_ori else 10
        cluster_segments = agglomerative_clustering_without_dist(split_traj, n_clusters)
    else:  # kmeans
        cluster_segments, _ = kmeans_without_dist(cluster_dict_ori, split_traj)
        # Reformat to match expected output
        cluster_dict = {}
        for k in cluster_segments.keys():
            center = compute_center(cluster_segments[k])
            dists = [traj2trajIED(
                center if isinstance(center, list) else center.points,
                t.points
            ) for t in cluster_segments[k]]
            cluster_dict[k] = [np.mean(dists) if dists else 0, center, dists, cluster_segments[k]]
        count_sim = sum(np.sum(cluster_dict[k][2]) for k in cluster_dict)
        traj_num = sum(len(cluster_dict[k][3]) for k in cluster_dict)
        overall_sim = count_sim / traj_num if traj_num > 0 else 1e10
        return cluster_dict, overall_sim, traj_num, overall_sim, 0

    # For DBSCAN/agglomerative
    cluster_dict = {}
    for k in cluster_segments.keys():
        segs = cluster_segments[k]
        center = compute_center(segs)
        dists = []
        for t in segs:
            d = traj2trajIED(
                center if isinstance(center, list) else center,
                t.points
            )
            dists.append(d)
        cluster_dict[k] = [np.mean(dists) if dists else 0, center, dists, segs]

    count_sim = sum(np.sum(cluster_dict[k][2]) for k in cluster_dict)
    traj_num = sum(len(cluster_dict[k][3]) for k in cluster_dict)
    overall_sim = count_sim / traj_num if traj_num > 0 else 1e10
    less_traj = sum(1 for k in cluster_dict if len(cluster_dict[k][3]) < 2)
    return cluster_dict, overall_sim, traj_num, overall_sim, less_traj
