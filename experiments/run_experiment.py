#!/usr/bin/env python3
"""Unified experiment runner for Q-RLSTC dual-version experiments.

Supports:
  - Version A (direct comparison, 5 qubits) and B (quantum-optimized, 8 qubits)
  - Synthetic and real-world datasets (T-Drive, GeoLife, Porto)
  - Multiple dataset sizes (small/medium/large/xlarge)
  - Online training and offline evaluation phases
  - Configurable noise models

Usage:
    # Synthetic data
    python experiments/run_experiment.py --version A --size small --phase online
    python experiments/run_experiment.py --version B --size small --phase online

    # Real-world data
    python experiments/run_experiment.py --version A --dataset tdrive --size small
    python experiments/run_experiment.py --version B --dataset geolife --size medium
    python experiments/run_experiment.py --version A --dataset porto --size small

    # Compare
    python experiments/compare_versions.py --size small
    python experiments/compare_versions.py --size small --dataset tdrive
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import numpy as np
import time
from datetime import datetime

from q_rlstc.config import QRLSTCConfig
from q_rlstc.data.load_dataset import load_prepared_dataset
from q_rlstc.rl.train import Trainer
from q_rlstc.clustering.metrics import segmentation_f1


def create_config(version: str, noise_model: str = "ideal") -> QRLSTCConfig:
    """Create a QRLSTCConfig for the given version.
    
    NISQ-optimized hyperparameters are applied to both versions
    for consistency.
    """
    config = QRLSTCConfig(version=version)
    
    # NISQ-optimized hyperparameters (same for both versions)
    config.rl.gamma = 0.90
    config.spsa.c = 0.10
    config.vqdqn.shots_train = 512
    config.vqdqn.shots_eval = 4096
    
    # Noise model
    config.noise.use_noise = (noise_model != "ideal")
    config.noise.noise_model = noise_model if noise_model != "ideal" else "ideal"
    
    return config


def _resolve_dataset_key(dataset: str, size: str) -> str:
    """Build the .npz filename key: 'small' for synthetic, 'tdrive_small' for real."""
    if dataset == "synthetic":
        return size
    return f"{dataset}_{size}"


def run_online_training(
    version: str,
    size: str,
    dataset: str = "synthetic",
    n_epochs: int = 2,
    noise_model: str = "ideal",
    seed: int = 42,
    output_dir: Path = None,
) -> dict:
    """Run online training experiment.
    
    Returns:
        Dict with all metrics and paths.
    """
    ds_label = f"{dataset}/{size}" if dataset != "synthetic" else size
    print(f"\n{'=' * 60}")
    print(f"Q-RLSTC Experiment: Version {version.upper()}, Dataset={ds_label}, Phase=online")
    print(f"{'=' * 60}")
    
    # Load prepared dataset
    ds_key = _resolve_dataset_key(dataset, size)
    print(f"\nLoading dataset '{ds_key}'...")
    dataset_obj = load_prepared_dataset(size=ds_key)
    print(f"  {dataset_obj.n_trajectories} trajectories loaded")
    
    # Create config
    config = create_config(version, noise_model)
    config.training.n_epochs = n_epochs
    
    # Create trainer
    trainer = Trainer(dataset_obj, config)
    
    # Print circuit info
    info = trainer.agent.get_circuit_info()
    print(f"  Version: {version.upper()}")
    print(f"  Circuit: {info.n_qubits} qubits, {info.n_params} params, depth={info.depth}")
    print(f"  Readout: {trainer.agent.readout_mode}")
    print(f"  Features: {'8D' if version.upper() == 'B' else '5D'}")
    
    # Train
    start_time = time.time()
    result = trainer.train(n_epochs=n_epochs, verbose=True)
    elapsed = time.time() - start_time
    
    # Compute key metrics
    episode_rewards = result.episode_rewards
    avg_final_reward = (
        np.mean(episode_rewards[-10:]) 
        if len(episode_rewards) >= 10 
        else np.mean(episode_rewards) if episode_rewards else 0.0
    )
    
    # Convergence: first episode reaching 90% of max reward
    max_reward = max(episode_rewards) if episode_rewards else 0
    convergence_ep = len(episode_rewards)
    if max_reward > 0:
        target = 0.9 * max_reward
        for i, r in enumerate(episode_rewards):
            if r >= target:
                convergence_ep = i
                break
    
    # Parameter efficiency
    param_eff = avg_final_reward / info.n_params if avg_final_reward > 0 else 0
    
    # Prepare output directory
    if output_dir is None:
        ds_dir = dataset if dataset != "synthetic" else ""
        dir_name = f"{ds_key}_online"
        output_dir = Path("outputs") / f"version_{version.lower()}" / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint
    checkpoint_path = output_dir / "checkpoint.npz"
    trainer.agent.save_checkpoint(str(checkpoint_path))
    
    # Collect metrics
    metrics = {
        "version": version.upper(),
        "dataset": dataset,
        "size": size,
        "phase": "online",
        "noise_model": noise_model,
        "seed": seed,
        "n_epochs": n_epochs,
        "n_episodes": result.n_episodes,
        "n_qubits": info.n_qubits,
        "n_params": info.n_params,
        "circuit_depth": info.depth,
        "readout_mode": trainer.agent.readout_mode,
        "feature_dim": 8 if version.upper() == "B" else 5,
        # --- Key metrics ---
        "delta_od": result.od_history[0] - result.final_od if result.od_history else 0,
        "final_od": result.final_od,
        "final_f1": result.final_f1,
        "avg_final_reward": float(avg_final_reward),
        "convergence_episode": convergence_ep,
        "parameter_efficiency": float(param_eff),
        # --- Time ---
        "runtime_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save full history
    np.savez(
        output_dir / "history.npz",
        episode_rewards=episode_rewards,
        od_history=result.od_history,
    )
    
    # Save config snapshot
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Print summary
    print(f"\n--- Results Summary ---")
    print(f"  ΔOD:                 {metrics['delta_od']:.4f}")
    print(f"  Final OD:            {metrics['final_od']:.4f}")
    print(f"  Final F1:            {metrics['final_f1']:.4f}")
    print(f"  Avg Final Reward:    {metrics['avg_final_reward']:.4f}")
    print(f"  Convergence Episode: {metrics['convergence_episode']}")
    print(f"  Param Efficiency:    {metrics['parameter_efficiency']:.6f}")
    print(f"  Runtime:             {elapsed:.1f}s")
    print(f"\n  Saved to: {output_dir}")
    
    return metrics


def run_offline_eval(
    version: str,
    size: str,
    checkpoint_path: str,
    dataset: str = "synthetic",
    noise_model: str = "ideal",
    output_dir: Path = None,
) -> dict:
    """Run offline evaluation with pre-trained checkpoint.
    
    Loads a pre-trained agent and evaluates on the dataset without
    further training. Used to check for overfitting.
    """
    ds_key = _resolve_dataset_key(dataset, size)
    print(f"\n{'=' * 60}")
    print(f"Q-RLSTC Offline Eval: Version {version.upper()}, Dataset={ds_key}")
    print(f"{'=' * 60}")
    
    # Load dataset
    dataset_obj = load_prepared_dataset(size=ds_key)
    print(f"  {dataset_obj.n_trajectories} trajectories loaded")
    
    # Create config (no training)
    config = create_config(version, noise_model)
    
    # Create trainer and load checkpoint
    trainer = Trainer(dataset_obj, config)
    trainer.agent.load_checkpoint(checkpoint_path)
    trainer.agent.epsilon = 0.0  # Fully greedy
    
    info = trainer.agent.get_circuit_info()
    print(f"  Loaded checkpoint: {checkpoint_path}")
    print(f"  Circuit: {info.n_qubits} qubits, {info.n_params} params")
    
    # Evaluate on all trajectories
    from q_rlstc.rl.train import MDPEnvironment
    from q_rlstc.clustering.metrics import overall_distance, segmentation_f1
    
    total_reward = 0.0
    all_ods = []
    all_f1s = []
    
    for traj in dataset_obj.trajectories:
        env = MDPEnvironment(traj, trainer.feature_extractor, config.clustering.n_clusters)
        state = env.reset()
        
        while not env.done:
            action = trainer.agent.act(state, greedy=True)
            state, reward, done = env.step(action)
            total_reward += reward
        
        all_ods.append(env.current_od)
        
        if traj.boundaries:
            f1 = segmentation_f1(env.boundaries, traj.boundaries, tolerance=1)
            all_f1s.append(f1)
    
    avg_od = np.mean(all_ods)
    avg_f1 = np.mean(all_f1s) if all_f1s else 0.0
    avg_reward = total_reward / dataset_obj.n_trajectories
    
    # Prepare output
    if output_dir is None:
        dir_name = f"{ds_key}_offline"
        output_dir = Path("outputs") / f"version_{version.lower()}" / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "version": version.upper(),
        "dataset": dataset,
        "size": size,
        "phase": "offline",
        "noise_model": noise_model,
        "checkpoint": checkpoint_path,
        "n_trajectories_eval": dataset_obj.n_trajectories,
        "avg_od": float(avg_od),
        "avg_f1": float(avg_f1),
        "avg_reward": float(avg_reward),
        "timestamp": datetime.now().isoformat(),
    }
    
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n--- Offline Eval Results ---")
    print(f"  Avg OD:     {avg_od:.4f}")
    print(f"  Avg F1:     {avg_f1:.4f}")
    print(f"  Avg Reward: {avg_reward:.4f}")
    print(f"\n  Saved to: {output_dir}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Q-RLSTC Experiment Runner")
    parser.add_argument(
        "--version", type=str, required=True, choices=["A", "B", "a", "b"],
        help="Experiment version: A (direct comparison) or B (quantum-optimized)"
    )
    parser.add_argument(
        "--dataset", type=str, default="synthetic",
        choices=["synthetic", "tdrive", "geolife", "porto"],
        help="Dataset source (default: synthetic)"
    )
    parser.add_argument(
        "--size", type=str, default="small",
        choices=["small", "medium", "large", "xlarge"],
        help="Dataset size"
    )
    parser.add_argument(
        "--phase", type=str, default="online",
        choices=["online", "offline"],
        help="Training phase"
    )
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs")
    parser.add_argument("--noise", type=str, default="ideal",
                        choices=["ideal", "eagle", "heron"],
                        help="Noise model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path (required for offline phase)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    if args.phase == "online":
        run_online_training(
            version=args.version.upper(),
            size=args.size,
            dataset=args.dataset,
            n_epochs=args.epochs,
            noise_model=args.noise,
            seed=args.seed,
            output_dir=output_dir,
        )
    elif args.phase == "offline":
        if not args.checkpoint:
            parser.error("--checkpoint is required for offline phase")
        run_offline_eval(
            version=args.version.upper(),
            size=args.size,
            checkpoint_path=args.checkpoint,
            dataset=args.dataset,
            noise_model=args.noise,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
