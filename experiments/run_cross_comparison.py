#!/usr/bin/env python3
"""Cross-system comparison: RLSTCcode (classical MLP) vs Q-RLSTC (VQC).

Runs both systems on the same T-Drive data with matched hyperparameters.
The ONLY intentional difference is the function approximator:
  - Classical: 5→64→2 MLP, SGD, ~450 params
  - Quantum D: 5q × 3-layer VQC, SPSA, 30 params

Usage::

    python experiments/run_cross_comparison.py \\
        --traj-path  ../RLSTCcode/data/Tdrive_norm_traj \\
        --centers-path ../RLSTCcode/data/tdrive_clustercenter \\
        --amount 500 \\
        --output-dir results/cross_comparison
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project roots are importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RLSTC_ROOT = _PROJECT_ROOT.parent / "RLSTCcode" / "subtrajcluster"
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_RLSTC_ROOT))


# ---------------------------------------------------------------------------
# Classical RLSTCcode runner
# ---------------------------------------------------------------------------

def run_classical_experiment(
    traj_path: str,
    centers_path: str,
    amount: int,
    output_dir: Path,
    seed: int = 1,
) -> dict:
    """Run the classical RLSTCcode training pipeline.

    Imports RLSTCcode modules directly and runs training with matched
    hyperparameters.

    Returns:
        Dict with training_cr, validation_cr, elapsed_time, param_count.
    """
    print("\n" + "=" * 60)
    print("  CLASSICAL EXPERIMENT (RLSTCcode — MLP 5→64→2)")
    print("=" * 60)

    # Import RLSTCcode modules
    from MDP import TrajRLclus
    from rl_nn import DeepQNetwork
    from cluster import compute_overdist

    import random
    np.random.seed(seed)
    random.seed(seed)

    # Match RLSTCcode defaults
    validation_percent = 0.1
    sidx = int(amount * (1 - validation_percent))
    eidx = amount

    env = TrajRLclus(traj_path, centers_path, centers_path)
    RL = DeepQNetwork(env.n_features, env.n_actions)

    # Count parameters
    param_count = sum(
        int(np.prod(w.shape)) for w in RL.model.get_weights()
    )
    print(f"  Parameters: {param_count}")

    # Training loop — matches rl_train.py exactly
    batch_size = 32
    n_rounds = 2
    results = {
        "system": "classical_rlstc",
        "param_count": param_count,
        "training_cr": [],
        "validation_cr": [],
    }

    idxlist = list(range(amount))
    start_time = time.time()

    for round_num in range(n_rounds):
        random.shuffle(idxlist)
        total_reward = 0.0

        for episode in idxlist:
            observation, steps = env.reset(episode, "T")

            for index in range(1, steps):
                done = (index == steps - 1)
                action = RL.act(observation)
                observation_, reward = env.step(episode, action, index, "T")
                if reward != 0:
                    total_reward += reward
                RL.remember(observation, action, reward, observation_, done)
                if done:
                    break
                if len(RL.memory) > batch_size:
                    RL.replay(episode, batch_size)
                    RL.soft_update(0.05)
                observation = observation_

            # Periodic evaluation
            if episode % 500 == 0 and episode != 0:
                # Validation CR
                env.allsubtraj_E = []
                for e in range(sidx, eidx):
                    obs, s = env.reset(e, "E")
                    for idx in range(1, s):
                        act = RL.online_act(obs)
                        obs, _ = env.step(e, act, idx, "E")

                val_od = compute_overdist(env.clusters_E)
                val_cr = float(val_od / env.basesim_E)

                train_od = compute_overdist(env.clusters_T)
                train_cr = float(train_od / env.basesim_T)

                results["training_cr"].append(train_cr)
                results["validation_cr"].append(val_cr)

                print(f"  Round {round_num+1}, ep {episode}: "
                      f"Train CR={train_cr:.4f}, Val CR={val_cr:.4f}")

                # Reset eval clusters
                for i in env.clusters_E.keys():
                    from collections import defaultdict
                    env.clusters_E[i][0] = []
                    env.clusters_E[i][1] = []
                    env.clusters_E[i][3] = defaultdict(list)

        env.update_cluster("T")

    elapsed = time.time() - start_time
    results["elapsed_time"] = elapsed
    results["final_training_cr"] = results["training_cr"][-1] if results["training_cr"] else None
    results["final_validation_cr"] = results["validation_cr"][-1] if results["validation_cr"] else None

    # Save model
    model_path = output_dir / "classical_model.h5"
    RL.save(str(model_path))
    results["model_path"] = str(model_path)

    print(f"\n  Classical done in {elapsed:.1f}s")
    print(f"  Final Train CR: {results['final_training_cr']}")
    print(f"  Final Val CR:   {results['final_validation_cr']}")

    return results


# ---------------------------------------------------------------------------
# Quantum Q-RLSTC runner (Version D, noiseless)
# ---------------------------------------------------------------------------

def run_quantum_experiment(
    traj_path: str,
    centers_path: str,
    amount: int,
    output_dir: Path,
    seed: int = 42,
) -> dict:
    """Run Q-RLSTC Version D (5q, noiseless statevector).

    Uses the existing Q-RLSTC Trainer with matched hyperparameters:
      - replay_buffer_size = 5000 (not 10000)
      - soft_update_tau = 0.05
      - batch_size = 32
      - n_actions = 2
      - statevector simulator (exact, no shot noise)

    Returns:
        Dict with training_cr, validation_cr, elapsed_time, param_count.
    """
    print("\n" + "=" * 60)
    print("  QUANTUM EXPERIMENT (Q-RLSTC Version D — VQC 5q×3L)")
    print("=" * 60)

    from q_rlstc.config import QRLSTCConfig

    # Build config for Version D
    config = QRLSTCConfig(version="D")

    # Match RLSTCcode hyperparameters exactly
    config.rl.replay_buffer_size = 5000
    config.rl.batch_size = 32
    config.rl.gamma = 0.99
    config.rl.epsilon_start = 1.0
    config.rl.epsilon_end = 0.1
    config.rl.epsilon_decay = 0.99
    config.rl.n_actions = 2

    # Noiseless statevector (infinite shots, no noise)
    config.noise.backend_mode = "ideal"
    config.vqdqn.n_qubits = 5
    config.vqdqn.n_layers = 3

    # Count parameters: 2 * n_qubits * n_layers
    param_count = 2 * config.vqdqn.n_qubits * config.vqdqn.n_layers
    print(f"  Parameters: {param_count}")

    # Load data via bridge
    from experiments.data_bridge import load_tdrive_dataset
    data = load_tdrive_dataset(traj_path, centers_path, amount)

    # Initialize trainer
    from q_rlstc.rl.train import Trainer
    trainer = Trainer(config, output_dir=str(output_dir / "quantum"))

    results = {
        "system": "quantum_qrlstc_v_d",
        "param_count": param_count,
        "training_cr": [],
        "validation_cr": [],
    }

    start_time = time.time()

    # Run training using the Trainer's built-in loop
    try:
        metrics = trainer.train(
            trajectories=data["trajectories"],
            n_epochs=2,
            seed=seed,
        )
        results["training_cr"] = metrics.get("training_cr_history", [])
        results["validation_cr"] = metrics.get("validation_cr_history", [])
    except Exception as e:
        print(f"  ⚠ Q-RLSTC training error: {e}")
        results["error"] = str(e)

    elapsed = time.time() - start_time
    results["elapsed_time"] = elapsed
    results["final_training_cr"] = results["training_cr"][-1] if results["training_cr"] else None
    results["final_validation_cr"] = results["validation_cr"][-1] if results["validation_cr"] else None

    print(f"\n  Quantum done in {elapsed:.1f}s")
    print(f"  Final Train CR: {results['final_training_cr']}")
    print(f"  Final Val CR:   {results['final_validation_cr']}")

    return results


# ---------------------------------------------------------------------------
# Comparison and reporting
# ---------------------------------------------------------------------------

def compare_results(classical: dict, quantum: dict, output_dir: Path):
    """Compare and report on both experiments."""
    print("\n" + "=" * 60)
    print("  CROSS-SYSTEM COMPARISON")
    print("=" * 60)

    report = {
        "classical": classical,
        "quantum": quantum,
    }

    # Summary table
    print(f"\n{'Metric':<30} {'Classical':>15} {'Quantum D':>15}")
    print("-" * 60)
    print(f"{'Parameters':<30} {classical['param_count']:>15} {quantum['param_count']:>15}")
    print(f"{'Parameter ratio':<30} {'1.0×':>15} "
          f"{classical['param_count']/max(quantum['param_count'],1):.1f}× fewer")
    print(f"{'Training time (s)':<30} {classical.get('elapsed_time',0):>15.1f} "
          f"{quantum.get('elapsed_time',0):>15.1f}")

    if classical.get("final_validation_cr") and quantum.get("final_validation_cr"):
        c_cr = classical["final_validation_cr"]
        q_cr = quantum["final_validation_cr"]
        diff_pct = (q_cr - c_cr) / c_cr * 100 if c_cr != 0 else float("inf")
        print(f"{'Final Validation CR':<30} {c_cr:>15.4f} {q_cr:>15.4f}")
        print(f"{'CR difference':<30} {'':>15} {diff_pct:>+14.2f}%")
    else:
        print(f"{'Final Validation CR':<30} "
              f"{classical.get('final_validation_cr', 'N/A'):>15} "
              f"{quantum.get('final_validation_cr', 'N/A'):>15}")

    # Save results
    results_path = output_dir / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cross-system comparison: RLSTCcode vs Q-RLSTC Version D"
    )
    parser.add_argument(
        "--traj-path",
        default=str(_RLSTC_ROOT.parent / "data" / "Tdrive_norm_traj"),
        help="Path to RLSTCcode trajectory pickle",
    )
    parser.add_argument(
        "--centers-path",
        default=str(_RLSTC_ROOT.parent / "data" / "tdrive_clustercenter"),
        help="Path to RLSTCcode cluster centers pickle",
    )
    parser.add_argument("--amount", type=int, default=500)
    parser.add_argument("--output-dir", default="results/cross_comparison")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--run",
        choices=["both", "classical", "quantum"],
        default="both",
        help="Which system(s) to run",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classical_results = None
    quantum_results = None

    if args.run in ("both", "classical"):
        classical_results = run_classical_experiment(
            args.traj_path, args.centers_path, args.amount, output_dir, args.seed
        )

    if args.run in ("both", "quantum"):
        quantum_results = run_quantum_experiment(
            args.traj_path, args.centers_path, args.amount, output_dir, args.seed
        )

    if classical_results and quantum_results:
        compare_results(classical_results, quantum_results, output_dir)
    elif classical_results:
        with open(output_dir / "classical_results.json", "w") as f:
            json.dump(classical_results, f, indent=2, default=str)
        print(f"\nClassical results saved to {output_dir / 'classical_results.json'}")
    elif quantum_results:
        with open(output_dir / "quantum_results.json", "w") as f:
            json.dump(quantum_results, f, indent=2, default=str)
        print(f"\nQuantum results saved to {output_dir / 'quantum_results.json'}")


if __name__ == "__main__":
    main()
