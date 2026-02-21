#!/usr/bin/env python3
"""Data bridge: convert RLSTCcode pickle data → Q-RLSTC Trajectory objects.

RLSTCcode stores data as pickled lists of ``Traj`` objects (with
``point.Point(x, y, t)`` members).  This module loads those pickles and
converts them into ``q_rlstc.data.synthetic.Trajectory`` objects so that
both systems operate on identical data.

Usage::

    from experiments.data_bridge import load_rlstc_pickle, load_cluster_centers

    trajectories = load_rlstc_pickle("path/to/Tdrive_norm_traj")
    centers, baseline_od = load_cluster_centers("path/to/tdrive_clustercenter")
"""

import pickle
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Add RLSTCcode to path so we can unpickle its objects
# ---------------------------------------------------------------------------
_RLSTC_ROOT = Path(__file__).resolve().parent.parent.parent / "RLSTCcode" / "subtrajcluster"
if _RLSTC_ROOT.exists():
    sys.path.insert(0, str(_RLSTC_ROOT))


def _ensure_qrlstc_types():
    """Import Q-RLSTC data types (lazy to avoid circular imports)."""
    from q_rlstc.data.synthetic import Point as QPoint, Trajectory
    return QPoint, Trajectory


# ---------------------------------------------------------------------------
# Pickle loaders
# ---------------------------------------------------------------------------

def load_rlstc_pickle(path: str) -> list:
    """Load a raw RLSTCcode pickle (list of Traj objects).

    The pickle was written with ``protocol=2`` by ``preprocessing.py``.
    We open with ``encoding='bytes'`` for Python-3 compatibility with
    Python-2 pickles, then ``encoding='latin1'`` as fallback.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"RLSTCcode data file not found: {p}")

    for enc in ("latin1", "bytes"):
        try:
            with open(p, "rb") as f:
                data = pickle.load(f, encoding=enc)
            return data
        except Exception:
            continue
    raise RuntimeError(f"Could not unpickle {p}")


def convert_trajectories(rlstc_trajs: list) -> list:
    """Convert RLSTCcode Traj objects → Q-RLSTC Trajectory objects.

    Both systems use ``Point(x, y, t)``; we just need to instantiate
    Q-RLSTC's dataclass versions.

    Args:
        rlstc_trajs: List of RLSTCcode ``Traj`` objects.

    Returns:
        List of ``q_rlstc.data.synthetic.Trajectory`` objects.
    """
    QPoint, Trajectory = _ensure_qrlstc_types()

    trajectories = []
    for i, traj in enumerate(rlstc_trajs):
        points = [
            QPoint(x=float(p.x), y=float(p.y), t=float(p.t))
            for p in traj.points
        ]
        trajectories.append(
            Trajectory(points=points, traj_id=getattr(traj, "traj_id", i))
        )
    return trajectories


def load_cluster_centers(path: str) -> Tuple[list, float]:
    """Load initial cluster centers from RLSTCcode pickle.

    The pickle contains ``[(_, baseline_od, centers_data)]`` where
    ``centers_data`` is a list of ``(_, center_traj, ...)`` tuples.

    Returns:
        (centers, baseline_od) where centers is a list of lists-of-QPoints.
    """
    QPoint, _ = _ensure_qrlstc_types()

    raw = load_rlstc_pickle(path)
    baseline_od = raw[0][1]
    centers_data = raw[0][2]

    centers = []
    for item in centers_data:
        center_traj = item[1]  # Traj object
        center_points = [
            QPoint(x=float(p.x), y=float(p.y), t=float(p.t))
            for p in center_traj.points
        ]
        centers.append(center_points)

    return centers, float(baseline_od)


# ---------------------------------------------------------------------------
# Convenience: load everything for an experiment
# ---------------------------------------------------------------------------

def load_tdrive_dataset(
    traj_path: str,
    centers_path: str,
    amount: int = 500,
) -> dict:
    """Load T-Drive dataset for cross-comparison experiments.

    Args:
        traj_path: Path to ``Tdrive_norm_traj`` pickle.
        centers_path: Path to ``tdrive_clustercenter`` pickle.
        amount: Number of trajectories to use.

    Returns:
        Dict with keys: trajectories, centers, baseline_od, train_idx, val_idx.
    """
    raw_trajs = load_rlstc_pickle(traj_path)
    trajectories = convert_trajectories(raw_trajs[:amount])
    centers, baseline_od = load_cluster_centers(centers_path)

    # RLSTCcode uses 10% validation split
    val_pct = 0.1
    n_train = int(amount * (1 - val_pct))
    train_idx = list(range(n_train))
    val_idx = list(range(n_train, amount))

    return {
        "trajectories": trajectories,
        "raw_trajs": raw_trajs[:amount],  # Keep originals for RLSTCcode
        "centers": centers,
        "baseline_od": baseline_od,
        "train_idx": train_idx,
        "val_idx": val_idx,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test data bridge")
    parser.add_argument("--traj", default="../RLSTCcode/data/Tdrive_norm_traj")
    parser.add_argument("--centers", default="../RLSTCcode/data/tdrive_clustercenter")
    parser.add_argument("--amount", type=int, default=100)
    args = parser.parse_args()

    data = load_tdrive_dataset(args.traj, args.centers, args.amount)
    print(f"Loaded {len(data['trajectories'])} trajectories")
    print(f"  First trajectory: {data['trajectories'][0].size} points")
    print(f"  Cluster centers: {len(data['centers'])} clusters")
    print(f"  Baseline OD: {data['baseline_od']:.4f}")
    print(f"  Train/Val split: {len(data['train_idx'])}/{len(data['val_idx'])}")
