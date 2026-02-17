"""Loaders for real-world trajectory datasets.

Supports T-Drive, GeoLife, and Porto. Each loader:
  1. Parses raw files into Trajectory/Point objects
  2. Computes heuristic segmentation boundaries (speed + direction change)
  3. Samples to a target count
  4. Returns a SyntheticDataset (compatible with all training code)

Usage:
    from q_rlstc.data.real_datasets import load_real_dataset
    dataset = load_real_dataset("tdrive", n_trajectories=50)
"""

import csv
import json
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from .synthetic import SyntheticDataset, Trajectory, Point


# ── Paths ─────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent

DATASET_PATHS = {
    "tdrive": _DATA_DIR / "TDRIVE" / "taxi_log_2008_by_id",
    "geolife": _DATA_DIR / "GEOLIFE" / "Data",
    "porto": _DATA_DIR / "PORTO.csv",
}


# ── Heuristic boundary detection ─────────────────────────────────

def _compute_heuristic_boundaries(
    points: List[Point],
    speed_threshold: float = 0.3,
    angle_threshold: float = 60.0,
    min_segment_len: int = 5,
) -> List[int]:
    """Detect segmentation boundaries using speed drops + direction changes.
    
    A boundary is placed where EITHER:
      - Speed drops below threshold (stop detection)
      - Direction changes by more than angle_threshold degrees
    
    Subject to minimum segment length to avoid micro-segments.
    
    Args:
        points: Trajectory points (must have t field as seconds or sequential).
        speed_threshold: Speed below which a stop is detected (units/sec).
        angle_threshold: Angle change in degrees that triggers a boundary.
        min_segment_len: Minimum points between boundaries.
    
    Returns:
        Sorted list of boundary indices.
    """
    n = len(points)
    if n < 2 * min_segment_len:
        return []
    
    boundaries = []
    last_boundary = 0
    
    for i in range(1, n - 1):
        if i - last_boundary < min_segment_len:
            continue
        
        # Speed check
        dt = abs(points[i].t - points[i - 1].t)
        if dt > 0:
            dist = points[i].distance(points[i - 1])
            speed = dist / dt
            if speed < speed_threshold:
                boundaries.append(i)
                last_boundary = i
                continue
        
        # Direction change check
        if i >= 2:
            dx1 = points[i].x - points[i - 1].x
            dy1 = points[i].y - points[i - 1].y
            dx2 = points[i + 1].x - points[i].x
            dy2 = points[i + 1].y - points[i].y
            
            len1 = np.sqrt(dx1**2 + dy1**2)
            len2 = np.sqrt(dx2**2 + dy2**2)
            
            if len1 > 1e-10 and len2 > 1e-10:
                cos_angle = (dx1 * dx2 + dy1 * dy2) / (len1 * len2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle_deg = np.degrees(np.arccos(cos_angle))
                
                if angle_deg > angle_threshold:
                    boundaries.append(i)
                    last_boundary = i
    
    return boundaries


# ── T-Drive loader ────────────────────────────────────────────────

def _parse_tdrive_file(filepath: Path, taxi_id: int) -> Optional[Trajectory]:
    """Parse a single T-Drive taxi log file.
    
    Format: taxi_id,datetime,longitude,latitude
    """
    points = []
    try:
        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 4:
                    continue
                try:
                    dt = datetime.strptime(parts[1].strip(), "%Y-%m-%d %H:%M:%S")
                    lng = float(parts[2])
                    lat = float(parts[3])
                    t = dt.timestamp()
                    points.append(Point(x=lng, y=lat, t=t))
                except (ValueError, IndexError):
                    continue
    except Exception:
        return None
    
    if len(points) < 10:
        return None
    
    # Sort by time
    points.sort(key=lambda p: p.t)
    
    # Normalize time to start from 0
    t0 = points[0].t
    for p in points:
        p.t -= t0
    
    boundaries = _compute_heuristic_boundaries(points)
    return Trajectory(points=points, traj_id=taxi_id, boundaries=boundaries)


def load_tdrive(
    n_trajectories: int = 50,
    min_points: int = 20,
    max_points: int = 500,
    seed: int = 42,
) -> SyntheticDataset:
    """Load T-Drive taxi trajectories from Beijing.
    
    Args:
        n_trajectories: Number of trajectories to sample.
        min_points: Minimum points per trajectory.
        max_points: Maximum points per trajectory (truncate longer ones).
        seed: Random seed for sampling.
    
    Returns:
        SyntheticDataset with heuristic boundaries.
    """
    data_dir = DATASET_PATHS["tdrive"]
    if not data_dir.exists():
        raise FileNotFoundError(f"T-Drive data not found at {data_dir}")
    
    # Collect all taxi files
    taxi_files = sorted(data_dir.glob("*.txt"))
    
    rng = np.random.default_rng(seed)
    rng.shuffle(taxi_files)
    
    trajectories = []
    for filepath in taxi_files:
        if len(trajectories) >= n_trajectories:
            break
        
        taxi_id = int(filepath.stem)
        traj = _parse_tdrive_file(filepath, taxi_id)
        
        if traj is None or len(traj.points) < min_points:
            continue
        
        # Truncate if too long
        if len(traj.points) > max_points:
            traj.points = traj.points[:max_points]
            # Recompute boundaries within truncated range
            traj.boundaries = [b for b in traj.boundaries if b < max_points]
        
        trajectories.append(traj)
    
    if len(trajectories) < n_trajectories:
        print(f"  Warning: Only found {len(trajectories)}/{n_trajectories} valid T-Drive trajectories")
    
    return SyntheticDataset(trajectories=trajectories)


# ── GeoLife loader ────────────────────────────────────────────────

def _parse_geolife_plt(filepath: Path) -> Optional[List[Point]]:
    """Parse a single GeoLife PLT file.
    
    Format (after 6-line header):
        latitude,longitude,0,altitude,days_since_18991230,date,time
    """
    points = []
    try:
        with open(filepath, "r") as f:
            # Skip 6-line header
            for _ in range(6):
                next(f)
            
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue
                try:
                    lat = float(parts[0])
                    lng = float(parts[1])
                    date_str = parts[5].strip()
                    time_str = parts[6].strip()
                    dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
                    t = dt.timestamp()
                    points.append(Point(x=lng, y=lat, t=t))
                except (ValueError, IndexError):
                    continue
    except Exception:
        return None
    
    return points if len(points) >= 10 else None


def _load_geolife_labels(user_dir: Path) -> Optional[List[Tuple[datetime, datetime, str]]]:
    """Load transportation mode labels if available."""
    labels_file = user_dir / "labels.txt"
    if not labels_file.exists():
        return None
    
    labels = []
    try:
        with open(labels_file, "r") as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                start = datetime.strptime(parts[0].strip(), "%Y/%m/%d %H:%M:%S")
                end = datetime.strptime(parts[1].strip(), "%Y/%m/%d %H:%M:%S")
                mode = parts[2].strip()
                labels.append((start, end, mode))
    except Exception:
        return None
    
    return labels


def _compute_label_boundaries(
    points: List[Point],
    labels: List[Tuple[datetime, datetime, str]],
) -> List[int]:
    """Compute boundaries from transportation mode transitions."""
    if not labels:
        return []
    
    boundaries = []
    current_mode = None
    
    for i, p in enumerate(points):
        pt_time = datetime.fromtimestamp(p.t)
        
        # Find which mode this point belongs to
        mode = None
        for start, end, m in labels:
            if start <= pt_time <= end:
                mode = m
                break
        
        if mode is not None and mode != current_mode and current_mode is not None:
            boundaries.append(i)
        
        if mode is not None:
            current_mode = mode
    
    return boundaries


def load_geolife(
    n_trajectories: int = 50,
    min_points: int = 20,
    max_points: int = 500,
    seed: int = 42,
    prefer_labeled: bool = True,
) -> SyntheticDataset:
    """Load GeoLife GPS trajectories.
    
    Args:
        n_trajectories: Number of trajectories to sample.
        min_points: Minimum points per trajectory.
        max_points: Maximum points (truncate longer ones).
        seed: Random seed.
        prefer_labeled: If True, prefer users with transportation mode labels.
    
    Returns:
        SyntheticDataset with labeled or heuristic boundaries.
    """
    data_dir = DATASET_PATHS["geolife"]
    if not data_dir.exists():
        raise FileNotFoundError(f"GeoLife data not found at {data_dir}")
    
    user_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    rng = np.random.default_rng(seed)
    
    # Sort labeled users first if preferred
    if prefer_labeled:
        labeled = [d for d in user_dirs if (d / "labels.txt").exists()]
        unlabeled = [d for d in user_dirs if not (d / "labels.txt").exists()]
        rng.shuffle(labeled)
        rng.shuffle(unlabeled)
        user_dirs = list(labeled) + list(unlabeled)
    else:
        rng.shuffle(user_dirs)
    
    trajectories = []
    traj_id = 0
    
    for user_dir in user_dirs:
        if len(trajectories) >= n_trajectories:
            break
        
        traj_dir = user_dir / "Trajectory"
        if not traj_dir.exists():
            continue
        
        labels = _load_geolife_labels(user_dir)
        plt_files = sorted(traj_dir.glob("*.plt"))
        rng.shuffle(plt_files)
        
        for plt_file in plt_files:
            if len(trajectories) >= n_trajectories:
                break
            
            points = _parse_geolife_plt(plt_file)
            if points is None or len(points) < min_points:
                continue
            
            # Sort by time
            points.sort(key=lambda p: p.t)
            
            # Truncate if too long
            if len(points) > max_points:
                points = points[:max_points]
            
            # Normalize time
            t0 = points[0].t
            for p in points:
                p.t -= t0
            
            # Get boundaries
            if labels:
                boundaries = _compute_label_boundaries(points, labels)
                if not boundaries:
                    boundaries = _compute_heuristic_boundaries(points)
            else:
                boundaries = _compute_heuristic_boundaries(points)
            
            traj = Trajectory(
                points=points, traj_id=traj_id, boundaries=boundaries
            )
            trajectories.append(traj)
            traj_id += 1
    
    if len(trajectories) < n_trajectories:
        print(f"  Warning: Only found {len(trajectories)}/{n_trajectories} valid GeoLife trajectories")
    
    return SyntheticDataset(trajectories=trajectories)


# ── Porto loader ──────────────────────────────────────────────────

def load_porto(
    n_trajectories: int = 50,
    min_points: int = 10,
    max_points: int = 500,
    seed: int = 42,
) -> SyntheticDataset:
    """Load Porto taxi trajectories.
    
    Porto CSV format:
        TRIP_ID, CALL_TYPE, ..., POLYLINE (JSON of [[lng,lat], ...])
    
    Each GPS point is recorded at 15-second intervals.
    
    Args:
        n_trajectories: Number of trajectories to sample.
        min_points: Minimum points per trajectory.
        max_points: Maximum points (truncate).
        seed: Random seed.
    
    Returns:
        SyntheticDataset with heuristic boundaries.
    """
    csv_path = DATASET_PATHS["porto"]
    if not csv_path.exists():
        raise FileNotFoundError(f"Porto data not found at {csv_path}")
    
    rng = np.random.default_rng(seed)
    
    # Porto has 1.7M rows — we use reservoir sampling to avoid loading all
    trajectories = []
    reservoir_size = n_trajectories * 10  # Sample 10x, then pick best
    reservoir = []
    
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Find POLYLINE column
        polyline_idx = header.index("POLYLINE")
        
        for row_idx, row in enumerate(reader):
            if len(row) <= polyline_idx:
                continue
            
            polyline_str = row[polyline_idx].strip()
            if not polyline_str or polyline_str == "[]":
                continue
            
            try:
                coords = json.loads(polyline_str)
            except json.JSONDecodeError:
                continue
            
            if len(coords) < min_points:
                continue
            
            # Reservoir sampling
            if len(reservoir) < reservoir_size:
                reservoir.append((row_idx, coords))
            else:
                j = rng.integers(0, row_idx + 1)
                if j < reservoir_size:
                    reservoir[j] = (row_idx, coords)
            
            # Early exit if we've scanned enough
            if row_idx > 500_000:
                break
    
    # Pick n_trajectories from reservoir
    rng.shuffle(reservoir)
    reservoir = reservoir[:n_trajectories]
    
    for traj_id, (row_idx, coords) in enumerate(reservoir):
        if len(coords) > max_points:
            coords = coords[:max_points]
        
        # Porto records every 15 seconds
        points = [
            Point(x=c[0], y=c[1], t=i * 15.0)
            for i, c in enumerate(coords)
            if len(c) >= 2
        ]
        
        if len(points) < min_points:
            continue
        
        boundaries = _compute_heuristic_boundaries(points)
        trajectories.append(Trajectory(
            points=points, traj_id=traj_id, boundaries=boundaries
        ))
    
    if len(trajectories) < n_trajectories:
        print(f"  Warning: Only found {len(trajectories)}/{n_trajectories} valid Porto trajectories")
    
    return SyntheticDataset(trajectories=trajectories)


# ── Unified interface ─────────────────────────────────────────────

LOADERS = {
    "tdrive": load_tdrive,
    "geolife": load_geolife,
    "porto": load_porto,
}


def load_real_dataset(
    name: str,
    n_trajectories: int = 50,
    seed: int = 42,
    **kwargs,
) -> SyntheticDataset:
    """Load a real-world dataset by name.
    
    Args:
        name: Dataset name ("tdrive", "geolife", "porto").
        n_trajectories: Number of trajectories to sample.
        seed: Random seed.
        **kwargs: Additional args passed to specific loader.
    
    Returns:
        SyntheticDataset compatible with all training code.
    """
    name = name.lower().replace("-", "").replace("_", "")
    
    if name not in LOADERS:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from: {list(LOADERS.keys())}"
        )
    
    return LOADERS[name](n_trajectories=n_trajectories, seed=seed, **kwargs)
