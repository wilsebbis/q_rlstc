#!/usr/bin/env python3
"""Generate and persist datasets for Q-RLSTC experiments.

Creates 4 dataset sizes with consistent min-max normalization.
Run once; all experiment scripts load from the saved .npz files.

Usage:
    python experiments/generate_datasets.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import json
from datetime import datetime

from q_rlstc.data.synthetic import TrajectoryGenerator, SyntheticDataset, Trajectory, Point


# Dataset size configurations
DATASET_CONFIGS = {
    "small": {"n_trajectories": 20, "n_segments_range": (2, 4)},
    "medium": {"n_trajectories": 50, "n_segments_range": (2, 4)},
    "large": {"n_trajectories": 200, "n_segments_range": (2, 6)},
    "xlarge": {"n_trajectories": 500, "n_segments_range": (2, 8)},
}

SEED = 42
OUTPUT_DIR = Path(__file__).parent / "datasets"


def dataset_to_arrays(dataset: SyntheticDataset):
    """Convert SyntheticDataset to serializable arrays.
    
    Returns:
        traj_points: list of np.ndarray, each (n_points, 3)
        traj_boundaries: list of np.ndarray of boundary indices
        traj_ids: np.ndarray of trajectory IDs
    """
    traj_points = []
    traj_boundaries = []
    traj_ids = []
    
    for traj in dataset.trajectories:
        arr = traj.to_array()  # (n_points, 3) — [x, y, t]
        traj_points.append(arr)
        traj_boundaries.append(np.array(traj.boundaries, dtype=np.int32))
        traj_ids.append(traj.traj_id if traj.traj_id is not None else -1)
    
    return traj_points, traj_boundaries, np.array(traj_ids, dtype=np.int32)


def compute_normalization(traj_points):
    """Compute min-max normalization parameters across all trajectories.
    
    Only normalizes x, y (columns 0, 1). Time (column 2) is left as-is
    since it's already sequential.
    
    Returns:
        scale_min: (2,) array of [x_min, y_min]
        scale_max: (2,) array of [x_max, y_max]
    """
    all_xy = np.vstack([arr[:, :2] for arr in traj_points])
    scale_min = all_xy.min(axis=0)
    scale_max = all_xy.max(axis=0)
    
    # Prevent division by zero
    span = scale_max - scale_min
    span[span < 1e-10] = 1.0
    scale_max = scale_min + span
    
    return scale_min, scale_max


def normalize_trajectories(traj_points, scale_min, scale_max):
    """Apply min-max normalization to [0, 1] for x, y coordinates."""
    span = scale_max - scale_min
    normalized = []
    for arr in traj_points:
        normed = arr.copy()
        normed[:, 0] = (arr[:, 0] - scale_min[0]) / span[0]
        normed[:, 1] = (arr[:, 1] - scale_min[1]) / span[1]
        normalized.append(normed)
    return normalized


def save_dataset(
    name: str,
    traj_points,
    traj_boundaries,
    traj_ids,
    scale_min,
    scale_max,
    config: dict,
):
    """Save dataset to .npz file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.npz"
    
    # Pack variable-length arrays using object arrays
    n_trajs = len(traj_points)
    
    # Store lengths so we can reconstruct
    point_counts = np.array([arr.shape[0] for arr in traj_points], dtype=np.int32)
    boundary_counts = np.array([arr.shape[0] for arr in traj_boundaries], dtype=np.int32)
    
    # Concatenate all points and boundaries into flat arrays
    all_points = np.vstack(traj_points)  # (total_points, 3)
    all_boundaries = np.concatenate(traj_boundaries) if any(len(b) > 0 for b in traj_boundaries) else np.array([], dtype=np.int32)
    
    np.savez_compressed(
        path,
        all_points=all_points,
        point_counts=point_counts,
        all_boundaries=all_boundaries,
        boundary_counts=boundary_counts,
        traj_ids=traj_ids,
        scale_min=scale_min,
        scale_max=scale_max,
        seed=np.array([SEED]),
        n_trajectories=np.array([n_trajs]),
        n_segments_min=np.array([config["n_segments_range"][0]]),
        n_segments_max=np.array([config["n_segments_range"][1]]),
    )
    
    print(f"  Saved: {path} ({path.stat().st_size / 1024:.1f} KB)")
    print(f"    {n_trajs} trajectories, {all_points.shape[0]} total points")
    return path


def main():
    print("=" * 60)
    print("Q-RLSTC Dataset Generator")
    print("=" * 60)
    print(f"  Seed: {SEED}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Sizes: {list(DATASET_CONFIGS.keys())}")
    print()
    
    # We generate ALL sizes from the same seed. The larger datasets
    # are NOT supersets of the smaller ones — they are independent
    # datasets of different sizes, all with seed=42.
    
    all_datasets = {}
    
    for name, config in DATASET_CONFIGS.items():
        print(f"Generating '{name}' dataset...")
        generator = TrajectoryGenerator(seed=SEED)
        dataset = generator.generate_dataset(
            n_trajectories=config["n_trajectories"],
            n_segments_range=config["n_segments_range"],
        )
        all_datasets[name] = (dataset, config)
        print(f"  {dataset.n_trajectories} trajectories generated")
    
    # Compute normalization from the LARGEST dataset so all datasets
    # share the same scale
    print("\nComputing normalization from 'xlarge' dataset...")
    largest_points, _, _ = dataset_to_arrays(all_datasets["xlarge"][0])
    scale_min, scale_max = compute_normalization(largest_points)
    print(f"  x range: [{scale_min[0]:.2f}, {scale_max[0]:.2f}]")
    print(f"  y range: [{scale_min[1]:.2f}, {scale_max[1]:.2f}]")
    
    # Normalize and save each dataset
    print("\nNormalizing and saving datasets...")
    for name, (dataset, config) in all_datasets.items():
        traj_points, traj_boundaries, traj_ids = dataset_to_arrays(dataset)
        traj_points_norm = normalize_trajectories(traj_points, scale_min, scale_max)
        save_dataset(name, traj_points_norm, traj_boundaries, traj_ids, 
                     scale_min, scale_max, config)
    
    # Save metadata
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "seed": SEED,
        "scale_min": scale_min.tolist(),
        "scale_max": scale_max.tolist(),
        "datasets": {
            name: {
                "n_trajectories": config["n_trajectories"],
                "n_segments_range": list(config["n_segments_range"]),
            }
            for name, config in DATASET_CONFIGS.items()
        },
    }
    
    meta_path = OUTPUT_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved: {meta_path}")
    
    print("\n" + "=" * 60)
    print("Done! All datasets generated and normalized.")
    print("=" * 60)


if __name__ == "__main__":
    main()
