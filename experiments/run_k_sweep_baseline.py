#!/usr/bin/env python3
"""Run K-Sweep Baseline Comparison between Classical RLSTC and Q-RLSTC."""

import os
import sys
import time
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.run_thesis_experiments import (
    ModelSpec, build_agent, train_and_evaluate, TrainingMode
)

def run_k_sweep_baseline():
    os.makedirs("results", exist_ok=True)
    # Target values of K to sweep over
    # We use currently available center files matching these K values
    k_values = [10]  # Baseline requested by advisor
    
    classic_ods = []
    classic_times = []
    
    quantum_ods = []
    quantum_times = []
    
    traj_path = "q_rlstc/data/geolife_norm_traj"
    centers_path = "q_rlstc/data/geolife_clustercenter"
    
    amount = 50  # Trajectory sample size for test speed
    epochs = 2
    seed = 42
    
    classical_spec = ModelSpec(
        name="Classical RLSTC (Original)",
        kind="original",  
        training_mode=TrainingMode.RLSTC_PARITY,
    )
    
    quantum_spec = ModelSpec(
        name="Q-RLSTC (Baseline)",
        kind="quantum",
        n_qubits=5,
        n_layers=3,
        shots=0,
        training_mode=TrainingMode.RLSTC_PARITY,
    )
    
    print(f"Beginning Baseline Comparison over K={k_values} in PARITY mode...")
    
    for k in k_values:
        print(f"\n{'='*40}\n   Running K={k}\n{'='*40}")
        
        # 1. Run Classical
        print("\n--- Running Classical Model ---")
        agent_c = build_agent(classical_spec, seed)
        t0 = time.time()
        res_c = train_and_evaluate(agent_c, classical_spec, traj_path, centers_path, amount, epochs, seed)
        ct = time.time() - t0
        classic_ods.append(res_c["val_ods"][-1]) 
        classic_times.append(ct)
        
        # 2. Run Quantum
        print("\n--- Running Quantum Model ---")
        agent_q = build_agent(quantum_spec, seed)
        t0 = time.time()
        res_q = train_and_evaluate(agent_q, quantum_spec, traj_path, centers_path, amount, epochs, seed)
        qt = time.time() - t0
        quantum_ods.append(res_q["val_ods"][-1])
        quantum_times.append(qt)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(k_values, classic_ods, marker='o', label="Classical RLSTC")
    ax1.plot(k_values, quantum_ods, marker='x', label="Q-RLSTC")
    ax1.set_xlabel("K (Clusters)")
    ax1.set_ylabel("Overall Distance")
    ax1.set_title("Clustering Quality vs K")
    ax1.set_xticks(k_values)
    ax1.legend()
    
    ax2.plot(k_values, classic_times, marker='o', label="Classical RLSTC")
    ax2.plot(k_values, quantum_times, marker='x', label="Q-RLSTC")
    ax2.set_xlabel("K (Clusters)")
    ax2.set_ylabel("Execution Time (s)")
    ax2.set_title("Execution Time vs K")
    ax2.set_xticks(k_values)
    ax2.legend()
    
    out_file = "results/k_sweep_baseline.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    print(f"\nData Collection Complete. Saved plots to {out_file}")

if __name__ == '__main__':
    run_k_sweep_baseline()
