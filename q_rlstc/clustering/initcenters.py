"""Cluster center initialization ported from RLSTCcode/subtrajcluster/initcenters.py.

K-means++ style initialization using IED distance for trajectory clustering.
"""

import argparse
import pickle
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from ..data.rlstc_trajdistance import traj2trajIED


def initialize_centers(data: List[Any], K: int) -> List[Any]:
    """K-means++ initialization using traj2trajIED.

    Selects K initial centers from data such that subsequent centers
    are maximally distant from existing centers.

    Args:
        data: list of Traj objects
        K: number of centers

    Returns:
        list of K Traj objects as initial centers
    """
    centers = [random.choice(data)]
    while len(centers) < K:
        distances = [
            min(traj2trajIED(center.points, traj.points) for center in centers)
            for traj in data
        ]
        new_center = data[distances.index(max(distances))]
        centers.append(new_center)
    return centers


def getbaseclus(
    trajs: List[Any],
    k: int,
    subtrajs: List[Any],
) -> Dict[int, list]:
    """Build initial cluster dictionary from sub-trajectories.

    Assigns each sub-trajectory to its nearest center, then builds
    the cluster_dict structure expected by the MDP environment.

    Args:
        trajs: list of Traj objects (used for center initialization)
        k: number of clusters
        subtrajs: list of Traj objects (sub-trajectories to assign)

    Returns:
        cluster_dict: {cluster_id: [aver_dist, center, dists, segments]}
    """
    centers = initialize_centers(trajs, k)
    cluster_segments = defaultdict(list)
    dists_dict = defaultdict(list)

    for i in range(len(subtrajs)):
        mindist = float("inf")
        minidx = 0
        for j in range(k):
            dist = traj2trajIED(centers[j].points, subtrajs[i].points)
            if dist == 1e10:
                continue
            if dist < mindist:
                mindist = dist
                minidx = j
        if mindist == float("inf"):
            continue
        cluster_segments[minidx].append(subtrajs[i])
        dists_dict[minidx].append(mindist)

    # Ensure all clusters have at least one member
    for i in range(k):
        if len(cluster_segments[i]) == 0:
            cluster_segments[i].append(centers[i])
            dists_dict[i].append(0)

    cluster_dict = {}
    for i in cluster_segments.keys():
        center = centers[i]
        temp_dist = dists_dict[i]
        aver_dist = np.mean(temp_dist)
        cluster_dict[i] = [aver_dist, center, temp_dist, cluster_segments[i]]

    return cluster_dict


def saveclus(
    k: int,
    subtrajs: List[Any],
    trajs: List[Any],
    amount: int,
) -> List[Tuple[float, float, Dict[int, list]]]:
    """Build and save cluster centers with overall similarity.

    Matches RLSTCcode/subtrajcluster/initcenters.py saveclus().

    Returns:
        list of [(overall_sim, overall_sim, cluster_dict)]
    """
    trajs = trajs[:amount]
    cluster_dict = getbaseclus(trajs, k, subtrajs)
    count_sim, traj_num = 0, 0
    for i in cluster_dict.keys():
        count_sim += np.sum(cluster_dict[i][2])
        traj_num += len(cluster_dict[i][3])
    overall_sim = count_sim / traj_num if traj_num > 0 else 1e10
    return [(overall_sim, overall_sim, cluster_dict)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize cluster centers")
    parser.add_argument("-subtrajsfile", default='../data/traclus_subtrajs')
    parser.add_argument("-trajsfile", default='../data/Tdrive_norm_traj')
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("-amount", type=int, default=1000)
    parser.add_argument("-centerfile", default='../data/tdrive_clustercenter')
    args = parser.parse_args()

    subtrajs = pickle.load(open(args.subtrajsfile, 'rb'))
    trajs = pickle.load(open(args.trajsfile, 'rb'))
    res = saveclus(args.k, subtrajs, trajs, args.amount)
    pickle.dump(res, open(args.centerfile, 'wb'), protocol=2)
    print(f"Saved {args.k} cluster centers → {args.centerfile}")
