#!/usr/bin/env python3
"""NISQ comparison experiment for Q-RLSTC.

Compares training performance across:
1. Ideal (noiseless) simulation
2. IBM Eagle noise model
3. IBM Heron noise model

Generates metrics for the research narrative:
- Convergence rate (episodes to 90% max reward)
- Noise resilience ratio (R_real / R_ideal)
- Parameter efficiency
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import time
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any

from q_rlstc.config import QRLSTCConfig, NoiseConfig
from q_rlstc.data.synthetic import generate_synthetic_trajectories
from q_rlstc.rl.train import Trainer, TrainingResult


@dataclass
class ExperimentResult:
    """Result from one experimental condition."""
    noise_model: str
    training_result: TrainingResult
    convergence_episode: int  # First episode to reach 90% of max reward
    avg_final_reward: float
    parameter_efficiency: float  # final_reward / n_params


def find_convergence_episode(rewards: List[float], threshold: float = 0.9) -> int:
    """Find first episode reaching threshold of max reward."""
    if not rewards:
        return -1
    
    max_reward = max(rewards)
    if max_reward <= 0:
        return len(rewards)  # Never converged positively
    
    target = threshold * max_reward
    for i, r in enumerate(rewards):
        if r >= target:
            return i
    return len(rewards)


def run_experiment(
    noise_model: str,
    dataset,
    n_epochs: int = 3,
    seed: int = 42,
) -> ExperimentResult:
    """Run training experiment with specified noise model."""
    print(f"\n{'=' * 50}")
    print(f"Running: {noise_model.upper()}")
    print('=' * 50)
    
    config = QRLSTCConfig()
    config.training.n_epochs = n_epochs
    config.noise.use_noise = (noise_model != "ideal")
    config.noise.noise_model = noise_model if noise_model != "ideal" else "ideal"
    
    # NISQ-optimized hyperparameters
    config.rl.gamma = 0.90
    config.spsa.c = 0.10
    config.vqdqn.shots_train = 512
    config.vqdqn.shots_eval = 4096
    
    trainer = Trainer(dataset, config)
    
    # Print circuit info
    info = trainer.agent.get_circuit_info()
    print(f"Circuit: {info.n_qubits} qubits, {info.n_params} params, depth={info.depth}")
    
    result = trainer.train(n_epochs=n_epochs, verbose=True)
    
    # Compute metrics
    conv_episode = find_convergence_episode(result.episode_rewards)
    avg_reward = np.mean(result.episode_rewards[-10:]) if len(result.episode_rewards) >= 10 else np.mean(result.episode_rewards)
    param_eff = avg_reward / info.n_params if avg_reward > 0 else 0
    
    return ExperimentResult(
        noise_model=noise_model,
        training_result=result,
        convergence_episode=conv_episode,
        avg_final_reward=avg_reward,
        parameter_efficiency=param_eff,
    )


def main():
    """Run NISQ comparison experiment."""
    print("=" * 60)
    print("Q-RLSTC NISQ Comparison Experiment")
    print("=" * 60)
    
    # Configuration
    n_trajectories = 10
    n_epochs = 2
    seed = 42
    
    print(f"\nExperiment Configuration:")
    print(f"  Trajectories: {n_trajectories}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Seed: {seed}")
    
    # Generate shared dataset
    print("\nGenerating synthetic trajectories...")
    dataset = generate_synthetic_trajectories(
        n_trajectories=n_trajectories,
        n_segments_range=(2, 4),
        seed=seed,
    )
    print(f"  Generated {dataset.n_trajectories} trajectories")
    
    # Run experiments
    noise_models = ["ideal", "eagle", "heron"]
    results: Dict[str, ExperimentResult] = {}
    
    for noise_model in noise_models:
        results[noise_model] = run_experiment(
            noise_model=noise_model,
            dataset=dataset,
            n_epochs=n_epochs,
            seed=seed,
        )
    
    # Analysis
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print("\n### Convergence Rate (episodes to 90% max reward)")
    for name, result in results.items():
        print(f"  {name:8}: {result.convergence_episode:4} episodes")
    
    print("\n### Average Final Reward (last 10 episodes)")
    for name, result in results.items():
        print(f"  {name:8}: {result.avg_final_reward:.4f}")
    
    print("\n### Noise Resilience Ratio (R_noisy / R_ideal)")
    ideal_reward = results["ideal"].avg_final_reward
    for name, result in results.items():
        if name != "ideal" and ideal_reward > 0:
            ratio = result.avg_final_reward / ideal_reward
            status = "✓ Success" if ratio > 0.8 else "⚠ Degraded"
            print(f"  {name:8}: {ratio:.3f} {status}")
    
    print("\n### Parameter Efficiency (reward / 30 params)")
    for name, result in results.items():
        print(f"  {name:8}: {result.parameter_efficiency:.4f}")
    
    print("\n### Training Time")
    for name, result in results.items():
        print(f"  {name:8}: {result.training_result.runtime_seconds:.2f}s")
    
    # Save results
    output_dir = Path("./outputs")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"nisq_comparison_{timestamp}.npz"
    
    np.savez(
        output_path,
        noise_models=noise_models,
        ideal_rewards=results["ideal"].training_result.episode_rewards,
        eagle_rewards=results["eagle"].training_result.episode_rewards,
        heron_rewards=results["heron"].training_result.episode_rewards,
        ideal_od=results["ideal"].training_result.od_history,
        eagle_od=results["eagle"].training_result.od_history,
        heron_od=results["heron"].training_result.od_history,
    )
    print(f"\nResults saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
