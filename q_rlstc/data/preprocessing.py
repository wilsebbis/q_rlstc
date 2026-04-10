"""Data preprocessing utilities ported from RLSTCcode/subtrajcluster/preprocessing.py.

Functions for normalizing GPS trajectories, enforcing length constraints,
converting raw data to Point/Traj objects, and MDL-based simplification.

Pipeline order (matches RLSTCcode)::

    raw data → filter by bounding box → enforce length → z-score normalize
    → convert to Point/Traj → MDL simplify → pickle output

All pure NumPy — no TensorFlow dependency.
"""

import argparse
import math
import pickle
import random
from typing import Any, List, Optional, Tuple

import numpy as np

from .rlstc_point import Point
from .rlstc_traj import Traj
from .rlstc_trajdistance import traj_mdl_comp


def processtrajs(
    trajs: List[List[List[float]]],
    lon_range: Tuple[float, float] = (115.4, 117.5),
    lat_range: Tuple[float, float] = (39.4, 41.6),
) -> List[List[List[float]]]:
    """Filter trajectory points to within a geographic bounding box.

    Removes any point whose longitude or latitude falls outside the
    specified ranges.  Trajectories with no remaining points are dropped.

    Args:
        trajs: List of trajectories, each a list of ``[lon, lat, time]``.
        lon_range: ``(min_longitude, max_longitude)``.
            Defaults to Beijing (T-Drive dataset).
        lat_range: ``(min_latitude, max_latitude)``.
            Defaults to Beijing (T-Drive dataset).

    Returns:
        Filtered list of trajectories.
    """
    trajslist = []
    for i in range(len(trajs)):
        temptraj = []
        for j in range(len(trajs[i])):
            lon, lat = trajs[i][j][0], trajs[i][j][1]
            if lat_range[0] <= lat <= lat_range[1] and lon_range[0] <= lon <= lon_range[1]:
                temptraj.append(trajs[i][j])
        if len(temptraj) != 0:
            trajslist.append(temptraj)
    return trajslist


def processlength(
    trajs: List[List[Any]],
    max_length: int = 500,
    min_length: int = 10,
) -> List[List[Any]]:
    """Enforce min/max trajectory length via random subsampling.

    Trajectories longer than ``max_length`` are randomly subsampled
    (preserving temporal order).  Trajectories shorter than
    ``min_length`` are dropped.

    Args:
        trajs: Raw trajectory lists.
        max_length: Maximum number of points per trajectory.
        min_length: Minimum number of points per trajectory.

    Returns:
        Filtered and subsampled trajectory lists.
    """
    trajdata = []
    for i in range(len(trajs)):
        length = len(trajs[i])
        if length > max_length:
            length_list = list(range(length))
            random_sample = random.sample(length_list, max_length)
            sorted_sample = sorted(random_sample)
            temp_traj = [trajs[i][idx] for idx in sorted_sample]
            trajdata.append(temp_traj)
        elif min_length <= length <= max_length:
            trajdata.append(trajs[i])
    return trajdata


def split_traj(
    traj: List[Any],
    max_length: int,
    min_length: int,
) -> List[List[Any]]:
    """Split a trajectory into fixed-length sub-trajectories.

    Args:
        traj: A single trajectory (list of points).
        max_length: Maximum points per sub-trajectory.
        min_length: Minimum points for a sub-trajectory to be kept.

    Returns:
        List of sub-trajectory slices.
    """
    sub_trajs = []
    start = 0
    while start < len(traj):
        end = start + max_length
        if end > len(traj):
            end = len(traj)
        if end - start + 1 >= min_length:
            sub_trajs.append(traj[start:end])
        start = end
    return sub_trajs


def split_by_time_gap(trajs: List[List[Any]], max_gap_seconds: float = 1800.0) -> List[List[Any]]:
    """Split trajectories when consecutive points differ in time by more than max_gap_seconds.

    This addresses 'straight line' artifacts over missing data periods, creating
    separate trajectories when there is a significant tracking gap.

    Args:
        trajs: List of trajectories, each a list of ``[lon, lat, time]``.
        max_gap_seconds: Maximum allowed time gap between consecutive points.

    Returns:
        List of split trajectory lists.
    """
    split_trajs = []
    for traj in trajs:
        if not traj:
            continue
        current_sub = [traj[0]]
        for i in range(1, len(traj)):
            # traj[i] component 2 is the timestamp
            time_diff = traj[i][2] - traj[i-1][2]
            if time_diff > max_gap_seconds:
                split_trajs.append(current_sub)
                current_sub = [traj[i]]
            else:
                current_sub.append(traj[i])
        if current_sub:
            split_trajs.append(current_sub)
    return split_trajs


def normloctrajs(trajs: List[List[Any]]) -> List[np.ndarray]:
    """Z-score normalize longitude/latitude across all trajectories.

    Computes the global mean and standard deviation for longitude and
    latitude across all points, then normalizes each coordinate.
    Timestamps are left unchanged.

    Args:
        trajs: List of trajectories, each a list of ``[lon, lat, time]``.

    Returns:
        List of NumPy arrays of shape ``(n_points, 3)`` with normalized coords.
    """
    lons, lats = [], []
    for traj in trajs:
        for pt in traj:
            lons.append(pt[0])
            lats.append(pt[1])
    mean_lon, mean_lat = np.mean(lons), np.mean(lats)
    std_lon, std_lat = np.std(lons), np.std(lats)

    norm_trajs = []
    for traj in trajs:
        tmp_traj = []
        for pt in traj:
            norm_lon = (pt[0] - mean_lon) / std_lon
            norm_lat = (pt[1] - mean_lat) / std_lat
            tmp_traj.append([norm_lon, norm_lat, pt[2]])
        norm_trajs.append(np.array(tmp_traj))
    return norm_trajs


def normtimetrajs(trajs: List[List[Any]]) -> List[np.ndarray]:
    """Z-score normalize timestamps across all trajectories.

    Computes the global mean and standard deviation of all timestamps,
    then normalizes.  Spatial coordinates are left unchanged.

    Args:
        trajs: List of trajectories, each a list of ``[lon, lat, time]``.

    Returns:
        List of NumPy arrays of shape ``(n_points, 3)`` with normalized timestamps.
    """
    ts = []
    for traj in trajs:
        for pt in traj:
            ts.append(pt[2])
    mean_t, std_t = np.mean(ts), np.std(ts)

    norm_trajs = []
    for traj in trajs:
        tmp_traj = []
        for pt in traj:
            norm_t = (pt[2] - mean_t) / std_t
            tmp_traj.append([pt[0], pt[1], norm_t])
        norm_trajs.append(np.array(tmp_traj))
    return norm_trajs


def convert2traj(trajdata: List[Any]) -> List[Traj]:
    """Convert raw coordinate arrays to Point/Traj objects.

    Args:
        trajdata: List of trajectories, each a list/array of ``[x, y, t]``.

    Returns:
        List of :class:`Traj` objects.
    """
    trajlists = []
    for i in range(len(trajdata)):
        traj_points = []
        for j in range(len(trajdata[i])):
            p = Point(trajdata[i][j][0], trajdata[i][j][1], trajdata[i][j][2])
            traj_points.append(p)
        ts, te = traj_points[0].t, traj_points[-1].t
        size = len(traj_points)
        traj = Traj(traj_points, size, ts, te, i)
        trajlists.append(traj)
    return trajlists


def simplify(points: List[Point], traj_id: Optional[int]) -> Traj:
    """MDL-based trajectory simplification (Minimum Description Length).

    Greedily merges consecutive points as long as the simplified
    representation is cheaper (in MDL cost) than the original.  When
    the simplified cost exceeds the original, a breakpoint is inserted.

    Args:
        points: Ordered list of trajectory points.
        traj_id: Trajectory identifier for the output Traj.

    Returns:
        Simplified :class:`Traj` with fewer points.
    """
    simp_points = [points[0]]
    start_index = 0
    length = 1
    while start_index + length < len(points):
        curr_index = start_index + length
        cost_simp = traj_mdl_comp(points, start_index, curr_index, 'simp')
        cost_origin = traj_mdl_comp(points, start_index, curr_index, 'orign')
        if cost_simp > cost_origin:
            simp_points.append(points[curr_index])
            start_index = curr_index
            length = 1
        else:
            length += 1
    if not simp_points[-1].equal(points[-1]):
        simp_points.append(points[-1])
    ts = simp_points[0].t
    te = simp_points[-1].t
    size = len(simp_points)
    return Traj(simp_points, size, ts, te, traj_id)


def getsimptrajs(trajs: List[Traj]) -> List[Traj]:
    """Apply MDL simplification to all trajectories.

    Args:
        trajs: List of :class:`Traj` objects to simplify.

    Returns:
        List of simplified :class:`Traj` objects.
    """
    return [simplify(trajs[i].points, trajs[i].traj_id) for i in range(len(trajs))]


def preprocess_pipeline(
    traj_path: str,
    output_path: str,
    max_length: int = 500,
    min_length: int = 10,
) -> List[Traj]:
    """Full preprocessing pipeline: filter → length → normalize → simplify.

    Matches RLSTCcode/subtrajcluster/preprocessing.py ``__main__`` block.

    Args:
        traj_path: Path to raw trajectory pickle file.
        output_path: Path to write the processed trajectories.
        max_length: Maximum points per trajectory.
        min_length: Minimum points per trajectory.

    Returns:
        List of simplified :class:`Traj` objects.
    """
    trajs = pickle.load(open(traj_path, 'rb'))
    trajslist = processtrajs(trajs)
    trajslist = split_by_time_gap(trajslist)
    trajs = processlength(trajslist, max_length, min_length)
    norm_trajs = normtimetrajs(trajs)
    trajlists = convert2traj(norm_trajs)
    simpletrajs = getsimptrajs(trajlists)
    pickle.dump(simpletrajs, open(output_path, 'wb'), protocol=2)
    print(f"Preprocessed {len(simpletrajs)} trajectories → {output_path}")
    return simpletrajs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess trajectory data")
    parser.add_argument("-trajfile", default='../data/Tdrive', help="Raw input")
    parser.add_argument("-maxlen", type=int, default=500)
    parser.add_argument("-minlen", type=int, default=10)
    parser.add_argument("-output", default='../data/Tdrive_norm_traj')
    args = parser.parse_args()
    preprocess_pipeline(args.trajfile, args.output, args.maxlen, args.minlen)
