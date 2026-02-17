"""Load prepared datasets from .npz files.

Provides a single entry point for all experiment scripts to load
pre-generated, normalized datasets.
"""

import numpy as np
from pathlib import Path
from typing import Optional

from .synthetic import SyntheticDataset, Trajectory, Point


# Default dataset directory (relative to project root)
_DEFAULT_DATASET_DIR = Path(__file__).parent.parent.parent / "experiments" / "datasets"


def load_prepared_dataset(
    size: str = "small",
    dataset_dir: Optional[Path] = None,
) -> SyntheticDataset:
    """Load a pre-generated dataset from disk.
    
    Args:
        size: Dataset size name ('small', 'medium', 'large', 'xlarge').
        dataset_dir: Override directory for dataset files.
    
    Returns:
        SyntheticDataset with normalized trajectories and ground truth.
    
    Raises:
        FileNotFoundError: If dataset file doesn't exist. Run
            experiments/generate_datasets.py first.
    """
    dataset_dir = dataset_dir or _DEFAULT_DATASET_DIR
    path = dataset_dir / f"{size}.npz"
    
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset '{size}' not found at {path}. "
            f"Run 'python experiments/generate_datasets.py' first."
        )
    
    data = np.load(path, allow_pickle=False)
    
    # Reconstruct trajectories from flat arrays
    all_points = data["all_points"]          # (total_points, 3)
    point_counts = data["point_counts"]      # (n_trajs,)
    all_boundaries = data["all_boundaries"]  # (total_boundaries,)
    boundary_counts = data["boundary_counts"]  # (n_trajs,)
    traj_ids = data["traj_ids"]              # (n_trajs,)
    
    trajectories = []
    point_offset = 0
    boundary_offset = 0
    
    for i in range(len(point_counts)):
        n_points = point_counts[i]
        n_boundaries = boundary_counts[i]
        
        # Extract this trajectory's points
        traj_arr = all_points[point_offset:point_offset + n_points]
        points = [Point(x=row[0], y=row[1], t=row[2]) for row in traj_arr]
        
        # Extract this trajectory's boundaries
        if n_boundaries > 0:
            boundaries = all_boundaries[boundary_offset:boundary_offset + n_boundaries].tolist()
        else:
            boundaries = []
        
        traj = Trajectory(
            points=points,
            traj_id=int(traj_ids[i]) if traj_ids[i] >= 0 else None,
            boundaries=boundaries,
        )
        trajectories.append(traj)
        
        point_offset += n_points
        boundary_offset += n_boundaries
    
    return SyntheticDataset(trajectories=trajectories)


def get_normalization_params(
    size: str = "small",
    dataset_dir: Optional[Path] = None,
) -> dict:
    """Get the normalization parameters used for a dataset.
    
    Args:
        size: Dataset size name.
        dataset_dir: Override directory.
    
    Returns:
        Dict with 'scale_min', 'scale_max', 'seed', 'n_trajectories'.
    """
    dataset_dir = dataset_dir or _DEFAULT_DATASET_DIR
    path = dataset_dir / f"{size}.npz"
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset '{size}' not found at {path}.")
    
    data = np.load(path, allow_pickle=False)
    
    return {
        "scale_min": data["scale_min"],
        "scale_max": data["scale_max"],
        "seed": int(data["seed"][0]),
        "n_trajectories": int(data["n_trajectories"][0]),
    }
