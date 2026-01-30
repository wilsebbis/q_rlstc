#!/usr/bin/env python3
"""End-to-end demo for Q-RLSTC.

Generates synthetic trajectories, trains VQ-DQN agent for segmentation,
and reports clustering quality metrics.

Usage:
    python experiments/run_synth_demo.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import time
from datetime import datetime

from q_rlstc.config import QRLSTCConfig
from q_rlstc.data.synthetic import generate_synthetic_trajectories
from q_rlstc.rl.train import Trainer


def main():
    """Run synthetic demo."""
    print("=" * 60)
    print("Q-RLSTC: Quantum-Enhanced RL for Sub-Trajectory Clustering")
    print("=" * 60)
    print()
    
    # Configuration
    config = QRLSTCConfig()
    config.training.n_epochs = 2
    config.training.n_trajectories = 20
    config.vqdqn.shots_train = 256  # Lower for faster demo
    config.noise.use_noise = False  # Ideal simulation for demo
    
    print("Configuration:")
    print(f"  Trajectories: {config.training.n_trajectories}")
    print(f"  Epochs: {config.training.n_epochs}")
    print(f"  VQ-DQN: {config.vqdqn.n_qubits} qubits, {config.vqdqn.n_layers} layers")
    print(f"  Shots: {config.vqdqn.shots_train}")
    print(f"  Noise: {'Enabled' if config.noise.use_noise else 'Disabled'}")
    print()
    
    # Generate data
    print("Generating synthetic trajectories...")
    start = time.time()
    dataset = generate_synthetic_trajectories(
        n_trajectories=config.training.n_trajectories,
        n_segments_range=(2, 4),
        seed=42,
    )
    print(f"  Generated {dataset.n_trajectories} trajectories in {time.time() - start:.2f}s")
    
    # Show sample trajectory info
    sample = dataset.trajectories[0]
    print(f"  Sample trajectory: {sample.size} points, {len(sample.boundaries)} boundaries")
    print()
    
    # Create trainer
    print("Initializing VQ-DQN agent...")
    trainer = Trainer(dataset, config)
    
    # Print circuit info
    circuit_info = trainer.agent.get_circuit_info()
    print(f"VQ-DQN Circuit:")
    print(f"  Qubits: {circuit_info.n_qubits}")
    print(f"  Layers: {circuit_info.n_layers}")
    print(f"  Parameters: {circuit_info.n_params}")
    print(f"  Depth: {circuit_info.depth}")
    print(f"  Gates: {circuit_info.gate_counts}")
    print()
    
    # Train
    print("Training...")
    print("-" * 40)
    result = trainer.train(n_epochs=config.training.n_epochs, verbose=True)
    print("-" * 40)
    print()
    
    # Report results
    print("Results:")
    print(f"  Total episodes: {result.n_episodes}")
    print(f"  Final OD: {result.final_od:.4f}")
    print(f"  Final F1: {result.final_f1:.4f}")
    print(f"  Runtime: {result.runtime_seconds:.2f}s")
    print()
    
    # Save outputs
    output_dir = Path("./outputs")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"demo_results_{timestamp}.npz"
    
    np.savez(
        output_path,
        episode_rewards=result.episode_rewards,
        od_history=result.od_history,
        final_od=result.final_od,
        final_f1=result.final_f1,
        runtime=result.runtime_seconds,
    )
    print(f"Results saved to: {output_path}")
    
    # Summary
    print()
    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    main()
