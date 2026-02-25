"""Incremental IED clustering and cluster-center maintenance.

This module implements the core clustering logic for the RLSTCcode MDP
environment.  As the RL agent scans through a trajectory and decides
where to cut, these functions incrementally update the distance from
the growing sub-trajectory to each cluster center, assign segments to
the nearest cluster, and recompute cluster centers.

**Key concepts:**

- **incremental IED:** Instead of recomputing the full ``traj2trajIED``
  from scratch each time the agent extends a sub-trajectory by one
  point, the ``incremental_sp`` / ``incremental_nsp`` functions update
  a cached partial-distance dictionary (``k_dict``).

- **cluster_dict structure:** A dictionary keyed by cluster ID, where
  each value is a list:
  ``[distances, sub_trajectories, center_points, time_point_dict, segment_lengths]``

See Also:
    :mod:`data.rlstc_trajdistance` — distance functions used here.
    :class:`data.rlstc_mdp.TrajRLclus` — MDP that calls these functions.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .rlstc_point import Point
from .rlstc_segment import Segment
from .rlstc_traj import Traj
from .rlstc_trajdistance import (
    getstaticIED,
    line2lineIDE,
    makemid,
    traj2trajIED,
)

# Type alias for the per-cluster state dict used during incremental IED.
# Keys: 'mid_dist', 'real_dist', 'lastp', 'j'
IncrementalState = Dict[int, Dict[str, Any]]


# ─── Incremental IED: first point (start-point) ───────────────────────


def incremental_sp(
    traj_points_a: List[Point],
    center_points: List[Point],
    k_dict: IncrementalState,
    cluster_id: int,
) -> IncrementalState:
    """Compute IED from a sub-trajectory to a cluster center from scratch.

    Called when the current sub-trajectory has just one segment (i.e. the
    agent just started a new segment).  Sets up the ``k_dict[cluster_id]``
    cache with:

    - ``mid_dist``: IED over the overlapping portion
    - ``real_dist``: IED including non-overlapping tail
    - ``lastp``: the boundary point where overlap ends
    - ``j``: index into ``center_points`` at that boundary

    Args:
        traj_points_a: Points of the current sub-trajectory.
        center_points: Points of the cluster center.
        k_dict: Mutable dictionary of per-cluster incremental state.
        cluster_id: Which cluster's state to update.

    Returns:
        The updated ``k_dict``.
    """
    time_a_start = traj_points_a[0].t
    time_a_end = traj_points_a[-1].t
    time_c_start = center_points[0].t
    time_c_end = center_points[-1].t

    # No temporal overlap
    if time_a_start >= time_c_end or time_a_end <= time_c_start:
        k_dict[cluster_id]['mid_dist'] = 1e10
        k_dict[cluster_id]['real_dist'] = 1e10
        k_dict[cluster_id]['lastp'] = center_points[0]
        k_dict[cluster_id]['j'] = 0
        return k_dict

    last_center_idx = len(center_points) - 1

    if time_a_end >= time_c_end:
        # Sub-trajectory covers the entire center temporally
        full_dist = traj2trajIED(traj_points_a, center_points)
        k_dict[cluster_id]['mid_dist'] = full_dist
        k_dict[cluster_id]['real_dist'] = full_dist
        k_dict[cluster_id]['lastp'] = Point(
            center_points[-1].x, center_points[-1].y, time_a_end,
        )
        k_dict[cluster_id]['j'] = len(center_points) - 1
        return k_dict

    # Center extends beyond sub-trajectory — split at time_a_end
    while center_points[last_center_idx].t > time_a_end:
        last_center_idx -= 1

    if center_points[last_center_idx].t == time_a_end:
        # Exact match — no interpolation needed
        boundary_point = center_points[last_center_idx]
        front_center = center_points[:last_center_idx + 1]
        overlap_dist = traj2trajIED(traj_points_a, front_center)

        back_center = center_points[last_center_idx:]
        tail_dist = getstaticIED(
            back_center,
            traj_points_a[-1].x, traj_points_a[-1].y,
            time_a_end, time_c_end,
        )

        k_dict[cluster_id]['mid_dist'] = overlap_dist
        k_dict[cluster_id]['real_dist'] = overlap_dist + tail_dist
        k_dict[cluster_id]['lastp'] = boundary_point
        k_dict[cluster_id]['j'] = last_center_idx

    if center_points[last_center_idx].t < time_a_end:
        # Need to interpolate a boundary point on the center
        front_center = center_points[:last_center_idx + 1]
        interp_x = makemid(
            center_points[last_center_idx].x, center_points[last_center_idx].t,
            center_points[last_center_idx + 1].x, center_points[last_center_idx + 1].t,
            time_a_end,
        )
        interp_y = makemid(
            center_points[last_center_idx].y, center_points[last_center_idx].t,
            center_points[last_center_idx + 1].y, center_points[last_center_idx + 1].t,
            time_a_end,
        )
        boundary_point = Point(interp_x, interp_y, time_a_end)
        front_center.append(boundary_point)

        back_center = center_points[last_center_idx + 1:]
        back_center.insert(0, boundary_point)

        overlap_dist = traj2trajIED(traj_points_a, front_center)
        tail_dist = getstaticIED(
            back_center,
            traj_points_a[-1].x, traj_points_a[-1].y,
            time_a_end, time_c_end,
        )

        k_dict[cluster_id]['mid_dist'] = overlap_dist
        k_dict[cluster_id]['real_dist'] = overlap_dist + tail_dist
        k_dict[cluster_id]['lastp'] = boundary_point
        k_dict[cluster_id]['j'] = last_center_idx

    return k_dict


# ─── Incremental IED: subsequent points (non-start-point) ─────────────


def incremental_nsp(
    traj_points_a: List[Point],
    center_points: List[Point],
    k_dict: IncrementalState,
    cluster_id: int,
    point_index: int,
) -> IncrementalState:
    """Incrementally update IED after the sub-trajectory gains one more point.

    Instead of recomputing the full IED, this function only computes the
    distance contribution of the newly added segment
    ``[traj_points_a[i-1], traj_points_a[i]]`` and adds it to the cached
    partial distance in ``k_dict``.

    Args:
        traj_points_a: Points of the current sub-trajectory.
        center_points: Points of the cluster center.
        k_dict: Mutable per-cluster incremental state.
        cluster_id: Which cluster's state to update.
        point_index: Index of the newly added point in ``traj_points_a``.

    Returns:
        The updated ``k_dict``.
    """
    time_a_start = traj_points_a[0].t
    time_a_end = traj_points_a[-1].t
    time_c_start = center_points[0].t
    time_c_end = center_points[-1].t

    # No temporal overlap
    if time_a_start >= time_c_end or time_a_end <= time_c_start:
        k_dict[cluster_id]['mid_dist'] = 1e10
        k_dict[cluster_id]['real_dist'] = 1e10
        k_dict[cluster_id]['lastp'] = center_points[0]
        k_dict[cluster_id]['j'] = 0
        return k_dict

    # If previous distance was infinite, compute from scratch
    if k_dict[cluster_id]['mid_dist'] == 1e10:
        return incremental_sp(traj_points_a, center_points, k_dict, cluster_id)

    # Build the new single-segment sub-trajectory
    new_segment = [traj_points_a[point_index - 1], traj_points_a[point_index]]

    if time_c_end == time_a_end:
        # Center ends exactly where sub-trajectory ends
        remaining_center = center_points[k_dict[cluster_id]['j']:]
        cached_boundary = k_dict[cluster_id]['lastp']
        if center_points[k_dict[cluster_id]['j']].t <= cached_boundary.t:
            remaining_center[0] = cached_boundary
        else:
            remaining_center.insert(0, cached_boundary)

        incremental_dist = traj2trajIED(new_segment, remaining_center)
        k_dict[cluster_id]['mid_dist'] += incremental_dist
        k_dict[cluster_id]['real_dist'] = k_dict[cluster_id]['mid_dist']
        k_dict[cluster_id]['lastp'] = center_points[-1]
        k_dict[cluster_id]['j'] = len(center_points) - 1
        return k_dict

    if time_c_end < time_a_end and time_c_end > traj_points_a[point_index - 1].t:
        # Center ends within the new segment
        remaining_center = center_points[k_dict[cluster_id]['j']:]
        cached_boundary = k_dict[cluster_id]['lastp']
        if center_points[k_dict[cluster_id]['j']].t <= cached_boundary.t:
            remaining_center[0] = cached_boundary
        else:
            remaining_center.insert(0, cached_boundary)

        incremental_dist = traj2trajIED(new_segment, remaining_center)
        static_boundary = Point(center_points[-1].x, center_points[-1].y, time_a_end)
        k_dict[cluster_id]['mid_dist'] += incremental_dist
        k_dict[cluster_id]['real_dist'] = k_dict[cluster_id]['mid_dist']
        k_dict[cluster_id]['lastp'] = static_boundary
        k_dict[cluster_id]['j'] = len(center_points) - 1
        return k_dict

    if time_c_end < time_a_end and time_c_end <= traj_points_a[point_index - 1].t:
        # Center already ended before the new segment
        static_boundary = Point(center_points[-1].x, center_points[-1].y, time_a_end)
        segment_dist = line2lineIDE(
            traj_points_a[point_index - 1], traj_points_a[point_index],
            k_dict[cluster_id]['lastp'], static_boundary,
        )
        k_dict[cluster_id]['mid_dist'] += segment_dist
        k_dict[cluster_id]['real_dist'] = k_dict[cluster_id]['mid_dist']
        k_dict[cluster_id]['lastp'] = static_boundary
        k_dict[cluster_id]['j'] = len(center_points) - 1
        return k_dict

    if time_a_end < time_c_end:
        # Sub-trajectory ends before center — split center at time_a_end
        end_idx = len(center_points) - 1
        while center_points[end_idx].t > time_a_end:
            end_idx -= 1

        # Build the center slice from cached boundary to end_idx
        front_center = center_points[k_dict[cluster_id]['j']:end_idx + 1]
        cached_boundary = k_dict[cluster_id]['lastp']
        if cached_boundary.t >= front_center[0].t:
            front_center[0] = cached_boundary
        else:
            front_center.insert(0, cached_boundary)

        if center_points[end_idx].t == time_a_end:
            boundary_point = center_points[end_idx]
            overlap_dist = traj2trajIED(new_segment, front_center)

            back_center = center_points[end_idx:]
            tail_dist = getstaticIED(
                back_center,
                traj_points_a[-1].x, traj_points_a[-1].y,
                time_a_end, time_c_end,
            )

            k_dict[cluster_id]['mid_dist'] += overlap_dist
            k_dict[cluster_id]['real_dist'] = k_dict[cluster_id]['mid_dist'] + tail_dist
            k_dict[cluster_id]['lastp'] = boundary_point
            k_dict[cluster_id]['j'] = end_idx

        if center_points[end_idx].t < time_a_end:
            # Interpolate boundary on center
            interp_x = makemid(
                center_points[end_idx].x, center_points[end_idx].t,
                center_points[end_idx + 1].x, center_points[end_idx + 1].t,
                time_a_end,
            )
            interp_y = makemid(
                center_points[end_idx].y, center_points[end_idx].t,
                center_points[end_idx + 1].y, center_points[end_idx + 1].t,
                time_a_end,
            )
            boundary_point = Point(interp_x, interp_y, time_a_end)
            front_center.append(boundary_point)

            back_center = center_points[end_idx + 1:]
            back_center.insert(0, boundary_point)

            overlap_dist = traj2trajIED(new_segment, front_center)
            tail_dist = getstaticIED(
                back_center,
                traj_points_a[-1].x, traj_points_a[-1].y,
                time_a_end, time_c_end,
            )

            k_dict[cluster_id]['mid_dist'] += overlap_dist
            k_dict[cluster_id]['real_dist'] = k_dict[cluster_id]['mid_dist'] + tail_dist
            k_dict[cluster_id]['lastp'] = boundary_point
            k_dict[cluster_id]['j'] = end_idx

        return k_dict

    return k_dict


# ─── IED dispatch: start-point vs. non-start-point ────────────────────


def incremental_IED(
    traj_points_a: List[Point],
    center_points: List[Point],
    k_dict: IncrementalState,
    cluster_id: int,
    point_index: int,
    start_point_index: int,
) -> IncrementalState:
    """Dispatch to start-point or non-start-point incremental IED.

    On the first extension (``point_index == start_point_index + 1``),
    calls :func:`incremental_sp`.  Otherwise calls :func:`incremental_nsp`.

    Args:
        traj_points_a: Points of the current sub-trajectory.
        center_points: Points of the cluster center.
        k_dict: Per-cluster incremental state.
        cluster_id: Which cluster to update.
        point_index: Current point index in the sub-trajectory.
        start_point_index: Index where the current segment started.

    Returns:
        Updated ``k_dict``.
    """
    if point_index == start_point_index + 1:
        return incremental_sp(traj_points_a, center_points, k_dict, cluster_id)
    return incremental_nsp(traj_points_a, center_points, k_dict, cluster_id, point_index)


# ─── Find nearest cluster ─────────────────────────────────────────────


def incremental_mindist(
    trajectory: Traj,
    start_index: int,
    current_index: int,
    k_dict: IncrementalState,
    cluster_dict: Dict[int, list],
    episode: int,
) -> Tuple[float, int]:
    """Find the cluster whose center is closest to the current sub-trajectory.

    Iterates over all clusters, incrementally updating the IED for each,
    and returns the minimum distance and corresponding cluster ID.

    Args:
        trajectory: The full trajectory being segmented.
        start_index: Point index where the current segment starts.
        current_index: Point index of the segment's current endpoint.
        k_dict: Per-cluster incremental state dictionary.
        cluster_dict: Full cluster dictionary (see module docstring).
        episode: Episode index (unused, kept for API compatibility).

    Returns:
        Tuple of ``(min_distance, best_cluster_id)``.
    """
    min_distance = 1e10
    best_cluster_id = -1

    sub_traj_points = trajectory.points[start_index:current_index + 1]
    relative_current = current_index - start_index

    for count, cluster_id in enumerate(cluster_dict.keys()):
        center_points = cluster_dict[cluster_id][2]
        if len(center_points) == 0:
            continue

        k_dict = incremental_IED(
            sub_traj_points, center_points, k_dict,
            cluster_id, relative_current, 0,
        )

        candidate_dist = k_dict[cluster_id]['real_dist']
        if count == 0 or candidate_dist < min_distance:
            min_distance = candidate_dist
            best_cluster_id = cluster_id

    return min_distance, best_cluster_id


# ─── Cluster dictionary maintenance ───────────────────────────────────


def add2clusdict(
    points: List[Point],
    cluster_dict: Dict[int, list],
    cluster_id: int,
) -> None:
    """Add a sub-trajectory's points to the cluster's time-point dictionary.

    Updates ``cluster_dict[cluster_id][3]``, a dictionary mapping
    timestamps to ``[point_list, overlap_count, sum_x, sum_y]``.
    This is used later by :func:`computecenter` to produce a new
    representative center trajectory.

    Args:
        points: Points of the sub-trajectory to add.
        cluster_dict: Full cluster dictionary (mutated in place).
        cluster_id: Target cluster to add points to.
    """
    time_dict = cluster_dict[cluster_id][3]

    # Increment overlap count for existing timestamps within range
    for timestamp in time_dict.keys():
        if timestamp >= points[0].t and timestamp <= points[-1].t:
            time_dict[timestamp][1] += 1

    # Add each point
    for point in points:
        if point.t not in time_dict:
            # New timestamp: [point_list, overlap_count, sum_x, sum_y]
            time_dict[point.t] = [[point], 1, point.x, point.y]

            # Count how many existing sub-trajectories overlap this timestamp
            for existing_traj in cluster_dict[cluster_id][1][:-1]:
                if point.t >= existing_traj.ts and point.t <= existing_traj.te:
                    time_dict[point.t][1] += 1
        else:
            time_dict[point.t][0].append(point)
            time_dict[point.t][2] += point.x
            time_dict[point.t][3] += point.y


def computecenter(
    cluster_dict: Dict[int, list],
    cluster_id: int,
    threshold_count: int,
    threshold_time: float,
) -> List[Point]:
    """Recompute the representative center trajectory for a cluster.

    Groups timestamps within ``threshold_time`` of each other, averages
    the coordinates of points falling into each group, and produces a
    new center trajectory from those averaged points.

    Only timestamps whose overlap count meets ``threshold_count`` are
    included (or the mean count, if no timestamp meets the threshold).

    Args:
        cluster_dict: Full cluster dictionary.
        cluster_id: Which cluster to recompute.
        threshold_count: Minimum number of overlapping sub-trajectories
            for a timestamp to be included in center computation.
        threshold_time: Maximum temporal gap to group timestamps together.

    Returns:
        New center trajectory as a list of :class:`Point` objects.
    """
    time_dict = cluster_dict[cluster_id][3]
    sorted_timestamps = sorted(time_dict.keys())

    # Filter timestamps by overlap count
    overlap_counts = [time_dict[ts][1] for ts in sorted_timestamps]
    qualifying_timestamps = [
        ts for ts in sorted_timestamps
        if time_dict[ts][1] >= threshold_count
    ]

    # Fallback: use timestamps above mean overlap count
    if len(qualifying_timestamps) == 0:
        mean_count = np.mean(overlap_counts)
        qualifying_timestamps = [
            ts for ts in sorted_timestamps
            if time_dict[ts][1] >= mean_count
        ]

    # Group qualifying timestamps and average coordinates
    center_points: List[Point] = []
    group_start = 0
    group_idx = group_start + 1
    group_point_count = len(time_dict[qualifying_timestamps[group_start]][0])
    sum_x = time_dict[qualifying_timestamps[group_start]][2]
    sum_y = time_dict[qualifying_timestamps[group_start]][3]
    sum_t = qualifying_timestamps[group_start]

    while group_idx < len(qualifying_timestamps):
        time_gap = qualifying_timestamps[group_idx] - qualifying_timestamps[group_start]

        if time_gap <= threshold_time:
            # Still within the same temporal group
            group_point_count += len(time_dict[qualifying_timestamps[group_idx]][0])
            sum_x += time_dict[qualifying_timestamps[group_idx]][2]
            sum_y += time_dict[qualifying_timestamps[group_idx]][3]
            sum_t += qualifying_timestamps[group_idx]

            if group_idx == len(qualifying_timestamps) - 1:
                # Last timestamp — finalize this group
                num_timestamps = group_idx - group_start + 1
                avg_x = sum_x / group_point_count
                avg_y = sum_y / group_point_count
                avg_t = sum_t / num_timestamps
                center_points.append(Point(avg_x, avg_y, avg_t))
            group_idx += 1
        else:
            # Finalize current group and start a new one
            num_timestamps = group_idx - group_start
            avg_x = sum_x / group_point_count
            avg_y = sum_y / group_point_count
            avg_t = sum_t / num_timestamps
            center_points.append(Point(avg_x, avg_y, avg_t))

            group_start = group_idx
            group_idx = group_start + 1
            group_point_count = len(time_dict[qualifying_timestamps[group_start]][0])
            sum_x = time_dict[qualifying_timestamps[group_start]][2]
            sum_y = time_dict[qualifying_timestamps[group_start]][3]
            sum_t = qualifying_timestamps[group_start]

    return center_points


# ─── Overall distance (ValCR) metrics ──────────────────────────────────


def compute_overdist(cluster_dict: Dict[int, list]) -> float:
    """Compute Overall Distance (raw ValCR) across all clusters.

    Averages distance-to-center over all assigned sub-trajectories.

    Args:
        cluster_dict: Full cluster dictionary.

    Returns:
        Mean IED across all cluster members.
    """
    total_count = 0
    total_distance = 0.0
    for cluster_id in cluster_dict.keys():
        distances = cluster_dict[cluster_id][0]
        if len(distances) != 0:
            total_count += len(distances)
            total_distance += sum(distances)
    return total_distance / total_count if total_count > 0 else 0.0


def compute_overdist_per_point(cluster_dict: Dict[int, list]) -> float:
    """Per-point normalized ValCR: mean of (IED / segment_length).

    Removes length-dependence so short segments don't trivially lower
    the metric.  Requires ``cluster_dict[i][4]`` to contain segment
    lengths (parallel to ``[0]`` distances).

    Args:
        cluster_dict: Full cluster dictionary with segment lengths at index 4.

    Returns:
        Length-normalized average distance.
    """
    count = 0
    total_normalized = 0.0
    for cluster_id in cluster_dict.keys():
        distances = cluster_dict[cluster_id][0]
        segment_lengths = cluster_dict[cluster_id][4] if len(cluster_dict[cluster_id]) > 4 else []
        for dist, seg_len in zip(distances, segment_lengths):
            if seg_len > 0:
                total_normalized += dist / seg_len
                count += 1
    return total_normalized / count if count > 0 else 0.0


def compute_overdist_length_weighted(cluster_dict: Dict[int, list]) -> float:
    """Length-weighted ValCR: total_IED / total_points.

    Equivalent to per-point average distance to nearest center.
    Robust to segment count inflation.

    Args:
        cluster_dict: Full cluster dictionary with segment lengths at index 4.

    Returns:
        Total IED divided by total point count.
    """
    total_distance = 0.0
    total_points = 0
    for cluster_id in cluster_dict.keys():
        distances = cluster_dict[cluster_id][0]
        segment_lengths = cluster_dict[cluster_id][4] if len(cluster_dict[cluster_id]) > 4 else []
        total_distance += sum(distances) if distances else 0.0
        total_points += sum(segment_lengths) if segment_lengths else 0
    return total_distance / total_points if total_points > 0 else 0.0


def compute_sse(cluster_dict: Dict[int, list]) -> float:
    """Compute Sum of Squared Errors (SSE) across all clusters.

    SSE = sum_i sum_{s in C_i} IED(s, center_i)^2

    Captures within-cluster compactness as a squared penalty,
    complementing the linear-average OD/ValCR metric.  SSE penalises
    outlier segments more heavily and is analogous to the k-means
    objective function adapted to the IED distance.

    Args:
        cluster_dict: Full cluster dictionary.

    Returns:
        Total SSE across all clusters.
    """
    sse = 0.0
    for cluster_id in cluster_dict:
        distances = cluster_dict[cluster_id][0]
        if distances:
            sse += sum(d * d for d in distances)
    return sse


def compute_cluster_summary(
    cluster_dict: Dict[int, list],
    basesim: float,
) -> Dict[str, Any]:
    """Compute a full suite of clustering quality metrics.

    Returns a dictionary with:
        - od: raw Overall Distance
        - val_cr: Competitive Ratio (od / basesim)
        - sse: Sum of Squared Errors
        - n_segments: total segments assigned
        - n_active_clusters: clusters with at least one segment
        - mean_cluster_size: average segments per active cluster

    Args:
        cluster_dict: Full cluster dictionary.
        basesim: Baseline distance for CR computation.

    Returns:
        Dictionary of metrics.
    """
    od = compute_overdist(cluster_dict)
    sse = compute_sse(cluster_dict)

    n_segments = 0
    n_active = 0
    for cluster_id in cluster_dict:
        n = len(cluster_dict[cluster_id][0])
        if n > 0:
            n_segments += n
            n_active += 1

    return {
        "od": od,
        "val_cr": od / basesim if basesim > 0 else float("inf"),
        "sse": sse,
        "n_segments": n_segments,
        "n_active_clusters": n_active,
        "mean_cluster_size": n_segments / n_active if n_active > 0 else 0.0,
    }


# ─── Cluster center re-estimation ─────────────────────────────────────


def update_centers(
    cluster_dict: Dict[int, list],
    threshold_count: int,
    threshold_time: float,
) -> Tuple[float, Dict[int, list]]:
    """Recompute all cluster centers and return updated overall distance.

    For each cluster with assigned sub-trajectories, recomputes the
    representative center trajectory via :func:`computecenter`.

    Args:
        cluster_dict: Full cluster dictionary (mutated in place).
        threshold_count: Minimum overlap count for center computation.
        threshold_time: Maximum time gap for grouping timestamps.

    Returns:
        Tuple of ``(overall_distance, cluster_dict)``.
    """
    for cluster_id in cluster_dict.keys():
        if len(cluster_dict[cluster_id][0]) != 0:
            new_center = computecenter(cluster_dict, cluster_id, threshold_count, threshold_time)
            if len(new_center) != 0:
                cluster_dict[cluster_id][2] = new_center

    overall_distance = compute_overdist(cluster_dict)
    return overall_distance, cluster_dict
