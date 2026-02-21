"""TRACLUS-style MDL preprocessing for trajectory simplification.

Port of RLSTCcode's ``preprocessing.py`` into Q-RLSTC.  Provides the full
pipeline: coordinate filtering → length filtering → normalization →
MDL simplification.

The MDL simplification is the key TRACLUS contribution: it greedily
compresses trajectories by replacing segments with straight lines whenever
the MDL cost of the simplified representation is lower than the original.
"""

import math
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .synthetic import Point, Trajectory
from ..clustering.trajdistance import traj_mdl_comp


# ---------------------------------------------------------------------------
# Coordinate filtering (Beijing bounding box for T-Drive)
# ---------------------------------------------------------------------------

def filter_by_coordinates(
    trajectories: List[List[List[float]]],
    lon_range: Tuple[float, float] = (115.4, 117.5),
    lat_range: Tuple[float, float] = (39.4, 41.6),
) -> List[List[List[float]]]:
    """Filter trajectory points to a geographic bounding box.

    Default bounding box is Beijing (for T-Drive dataset).

    Args:
        trajectories: List of trajectories, each a list of [lon, lat, time].
        lon_range: (min_lon, max_lon).
        lat_range: (min_lat, max_lat).

    Returns:
        Filtered trajectories (only points within the bounding box).
    """
    result = []
    for traj in trajectories:
        filtered = [
            p for p in traj
            if lat_range[0] <= p[1] <= lat_range[1]
            and lon_range[0] <= p[0] <= lon_range[1]
        ]
        if len(filtered) > 0:
            result.append(filtered)
    return result


# ---------------------------------------------------------------------------
# Length filtering / splitting
# ---------------------------------------------------------------------------

def filter_by_length(
    trajectories: List[List],
    max_length: int = 500,
    min_length: int = 10,
) -> List[List]:
    """Filter trajectories by length, subsampling if too long.

    Args:
        trajectories: Raw trajectory data.
        max_length: Maximum number of points.
        min_length: Minimum number of points.

    Returns:
        Trajectories within the length bounds.
    """
    import random as _rng

    result = []
    for traj in trajectories:
        length = len(traj)
        if length > max_length:
            # Subsample while maintaining order
            indices = sorted(_rng.sample(range(length), max_length))
            result.append([traj[i] for i in indices])
        elif length >= min_length:
            result.append(traj)
    return result


def split_trajectory(
    traj: list,
    max_length: int = 500,
    min_length: int = 10,
) -> List[list]:
    """Split a long trajectory into sub-trajectories.

    Args:
        traj: Trajectory point list.
        max_length: Maximum segment length.
        min_length: Minimum segment length.

    Returns:
        List of sub-trajectory point lists.
    """
    sub_trajs = []
    start = 0
    while start < len(traj):
        end = min(start + max_length, len(traj))
        if end - start >= min_length:
            sub_trajs.append(traj[start:end])
        start = end
    return sub_trajs


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_locations(trajectories: List[List]) -> List[List]:
    """Z-score normalize spatial coordinates across all trajectories.

    Args:
        trajectories: List of trajectories, each point is [lon, lat, time].

    Returns:
        Normalized trajectories (same structure).
    """
    lons, lats = [], []
    for traj in trajectories:
        for p in traj:
            lons.append(p[0])
            lats.append(p[1])

    mean_lon, std_lon = np.mean(lons), np.std(lons)
    mean_lat, std_lat = np.mean(lats), np.std(lats)

    # Prevent division by zero
    if std_lon < 1e-10:
        std_lon = 1.0
    if std_lat < 1e-10:
        std_lat = 1.0

    result = []
    for traj in trajectories:
        norm_traj = [
            [(p[0] - mean_lon) / std_lon, (p[1] - mean_lat) / std_lat, p[2]]
            for p in traj
        ]
        result.append(norm_traj)
    return result


def normalize_time(trajectories: List[List]) -> List[List]:
    """Z-score normalize timestamps across all trajectories.

    Args:
        trajectories: List of trajectories, each point is [lon, lat, time].

    Returns:
        Normalized trajectories (same structure).
    """
    all_t = []
    for traj in trajectories:
        for p in traj:
            all_t.append(p[2])

    mean_t, std_t = np.mean(all_t), np.std(all_t)
    if std_t < 1e-10:
        std_t = 1.0

    result = []
    for traj in trajectories:
        norm_traj = [
            [p[0], p[1], (p[2] - mean_t) / std_t]
            for p in traj
        ]
        result.append(norm_traj)
    return result


# ---------------------------------------------------------------------------
# Convert raw arrays to Trajectory objects
# ---------------------------------------------------------------------------

def arrays_to_trajectories(traj_data: List[List]) -> List[Trajectory]:
    """Convert list of [lon, lat, time] arrays → Trajectory objects.

    Args:
        traj_data: List of trajectories, each a list of [lon, lat, time].

    Returns:
        List of Trajectory objects.
    """
    trajectories = []
    for i, traj in enumerate(traj_data):
        points = [Point(x=float(p[0]), y=float(p[1]), t=float(p[2])) for p in traj]
        trajectories.append(Trajectory(points=points, traj_id=i))
    return trajectories


# ---------------------------------------------------------------------------
# MDL Simplification (TRACLUS core algorithm)
# ---------------------------------------------------------------------------

def simplify_trajectory(trajectory: Trajectory) -> Trajectory:
    """Simplify a trajectory using the MDL principle.

    Greedily extends segments as long as the MDL cost of the simplified
    (straight-line) representation is lower than storing the original
    points.  When the simplified cost exceeds the original cost, a split
    point is created.

    This is the core TRACLUS contribution from Lee et al. 2007.

    Args:
        trajectory: Input trajectory.

    Returns:
        Simplified trajectory (fewer points, same start/end).
    """
    points = trajectory.points
    if len(points) < 3:
        return trajectory

    simp_points = [points[0]]
    start_index = 0
    length = 1

    while start_index + length < len(points):
        curr_index = start_index + length
        cost_simp = traj_mdl_comp(points, start_index, curr_index, "simp")
        cost_orig = traj_mdl_comp(points, start_index, curr_index, "orign")

        if cost_simp > cost_orig:
            simp_points.append(points[curr_index])
            start_index = curr_index
            length = 1
        else:
            length += 1

    # Ensure we include the last point
    if simp_points[-1] != points[-1]:
        last = points[-1]
        if not (abs(simp_points[-1].x - last.x) < 1e-15 and
                abs(simp_points[-1].y - last.y) < 1e-15 and
                abs(simp_points[-1].t - last.t) < 1e-15):
            simp_points.append(last)

    return Trajectory(
        points=simp_points,
        traj_id=trajectory.traj_id,
    )


def simplify_all(trajectories: List[Trajectory]) -> List[Trajectory]:
    """Apply MDL simplification to all trajectories.

    Args:
        trajectories: List of input trajectories.

    Returns:
        List of simplified trajectories.
    """
    return [simplify_trajectory(t) for t in trajectories]


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_tdrive(
    raw_trajs: List[List],
    max_length: int = 500,
    min_length: int = 10,
    apply_simplification: bool = True,
) -> List[Trajectory]:
    """Full preprocessing pipeline for T-Drive data.

    Pipeline:
        1. Filter by Beijing bounding box
        2. Filter by length
        3. Normalize locations (z-score)
        4. Normalize time (z-score)
        5. Convert to Trajectory objects
        6. MDL simplification (optional)

    Args:
        raw_trajs: Raw trajectory data (list of [lon, lat, time] lists).
        max_length: Maximum trajectory length.
        min_length: Minimum trajectory length.
        apply_simplification: Whether to apply MDL simplification.

    Returns:
        List of preprocessed Trajectory objects.
    """
    # Step 1: Geographic filter
    filtered = filter_by_coordinates(raw_trajs)

    # Step 2: Length filter
    filtered = filter_by_length(filtered, max_length, min_length)

    # Step 3-4: Normalize
    normalized = normalize_locations(filtered)
    normalized = normalize_time(normalized)

    # Step 5: Convert to objects
    trajectories = arrays_to_trajectories(normalized)

    # Step 6: MDL simplification
    if apply_simplification:
        trajectories = simplify_all(trajectories)

    return trajectories
