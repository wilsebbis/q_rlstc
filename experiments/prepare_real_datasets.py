#!/usr/bin/env python3
"""Prepare real-world datasets for Q-RLSTC experiments.

Loads T-Drive, GeoLife, and Porto trajectories, applies consistent
normalization, and saves to .npz files matching the synthetic pipeline.

Usage:
    python experiments/prepare_real_datasets.py
    python experiments/prepare_real_datasets.py --dataset tdrive --sizes 20 50
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import numpy as np
from datetime import datetime

from q_rlstc.data.real_datasets import load_real_dataset, LOADERS
from q_rlstc.data.synthetic import SyntheticDataset, Trajectory

# Map sizes to trajectory counts
SIZE_MAP = {
    "small": 20,
    "medium": 50,
    "large": 200,
    "xlarge": 500,
}

SEED = 42
OUTPUT_DIR = Path(__file__).parent / "datasets"


def dataset_to_arrays(dataset: SyntheticDataset):
    """Convert dataset to serializable arrays (same as generate_datasets.py)."""
    traj_points = []
    traj_boundaries = []
    traj_ids = []
    
    for traj in dataset.trajectories:
        arr = traj.to_array()
        traj_points.append(arr)
        traj_boundaries.append(np.array(traj.boundaries, dtype=np.int32))
        traj_ids.append(traj.traj_id if traj.traj_id is not None else -1)
    
    return traj_points, traj_boundaries, np.array(traj_ids, dtype=np.int32)


def compute_normalization(traj_points):
    """Compute min-max normalization parameters."""
    all_xy = np.vstack([arr[:, :2] for arr in traj_points])
    scale_min = all_xy.min(axis=0)
    scale_max = all_xy.max(axis=0)
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


def save_dataset(name, traj_points, traj_boundaries, traj_ids, scale_min, scale_max):
    """Save dataset to .npz file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.npz"
    
    point_counts = np.array([arr.shape[0] for arr in traj_points], dtype=np.int32)
    boundary_counts = np.array([arr.shape[0] for arr in traj_boundaries], dtype=np.int32)
    all_points = np.vstack(traj_points)
    all_boundaries = (
        np.concatenate(traj_boundaries) 
        if any(len(b) > 0 for b in traj_boundaries)
        else np.array([], dtype=np.int32)
    )
    
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
        n_trajectories=np.array([len(traj_points)]),
    )
    
    print(f"  Saved: {path} ({path.stat().st_size / 1024:.1f} KB)")
    print(f"    {len(traj_points)} trajectories, {all_points.shape[0]} total points")
    return path


def prepare_dataset(dataset_name: str, sizes: list):
    """Prepare one real-world dataset at multiple sizes."""
    print(f"\n{'=' * 60}")
    print(f"Preparing: {dataset_name.upper()}")
    print(f"{'=' * 60}")
    
    # Load the largest size first to compute normalization
    max_n = max(SIZE_MAP[s] for s in sizes)
    
    print(f"\nLoading {max_n} trajectories for normalization...")
    largest = load_real_dataset(dataset_name, n_trajectories=max_n, seed=SEED)
    print(f"  Got {largest.n_trajectories} trajectories")
    
    largest_points, _, _ = dataset_to_arrays(largest)
    scale_min, scale_max = compute_normalization(largest_points)
    print(f"  x range: [{scale_min[0]:.4f}, {scale_max[0]:.4f}]")
    print(f"  y range: [{scale_min[1]:.4f}, {scale_max[1]:.4f}]")
    
    # Generate each size
    for size_name in sizes:
        n = SIZE_MAP[size_name]
        print(f"\n  Preparing '{dataset_name}_{size_name}' ({n} trajectories)...")
        
        dataset = load_real_dataset(dataset_name, n_trajectories=n, seed=SEED)
        traj_points, traj_boundaries, traj_ids = dataset_to_arrays(dataset)
        traj_points_norm = normalize_trajectories(traj_points, scale_min, scale_max)
        
        save_name = f"{dataset_name}_{size_name}"
        save_dataset(save_name, traj_points_norm, traj_boundaries, traj_ids,
                     scale_min, scale_max)
    
    return scale_min, scale_max


def main():
    parser = argparse.ArgumentParser(description="Prepare real-world datasets")
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=["tdrive", "geolife", "porto"],
        help="Prepare a specific dataset (default: all)"
    )
    parser.add_argument(
        "--sizes", type=str, nargs="+", default=["small", "medium"],
        choices=["small", "medium", "large", "xlarge"],
        help="Sizes to prepare"
    )
    
    args = parser.parse_args()
    
    datasets = [args.dataset] if args.dataset else ["tdrive", "geolife", "porto"]
    
    print("=" * 60)
    print("Q-RLSTC Real-World Dataset Preparation")
    print("=" * 60)
    print(f"  Datasets: {datasets}")
    print(f"  Sizes: {args.sizes}")
    print(f"  Seed: {SEED}")
    
    all_meta = {}
    for ds_name in datasets:
        try:
            scale_min, scale_max = prepare_dataset(ds_name, args.sizes)
            all_meta[ds_name] = {
                "scale_min": scale_min.tolist(),
                "scale_max": scale_max.tolist(),
                "sizes_prepared": args.sizes,
            }
        except FileNotFoundError as e:
            print(f"\n  ⚠ Skipping {ds_name}: {e}")
    
    # Save metadata
    meta_path = OUTPUT_DIR / "real_datasets_metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "seed": SEED,
            "datasets": all_meta,
        }, f, indent=2)
    
    print(f"\nMetadata saved: {meta_path}")
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
