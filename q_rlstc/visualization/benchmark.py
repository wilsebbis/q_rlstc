"""
Benchmark runner for Q-RLSTC experiments.

Adapted from TheFinalQRLSTC/visualization/benchmark.py and
New_QRLSTC/unified_benchmark.py.

Provides:
- Tiered configs (small / medium / large)
- Checkpoint / resume capability
- Automatic plot generation
- Formatted summary table
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ── Tier configurations ───────────────────────────────────────────────

TIER_CONFIGS = {
    "small": {
        "n_trajectories": 20,
        "n_epochs": 3,
        "description": "Quick smoke test (~5 min)",
    },
    "medium": {
        "n_trajectories": 50,
        "n_epochs": 8,
        "description": "Standard evaluation (~15 min)",
    },
    "large": {
        "n_trajectories": 100,
        "n_epochs": 15,
        "description": "Full evaluation (~45+ min)",
    },
}


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class BenchmarkResults:
    """Container for a single experiment run's results."""

    version: str = "A"
    version_label: str = "Classical Parity (5q)"
    tier: str = "small"
    noise_model: str = "ideal"
    compute_backend: str = "cpu"
    seed: int = 42

    # Circuit info
    n_qubits: int = 0
    n_layers: int = 0
    n_params: int = 0
    circuit_depth: int = 0
    feature_dim: int = 5
    readout_mode: str = "standard"

    # Key metrics
    delta_od: float = 0.0
    final_od: float = 0.0
    final_f1: float = 0.0
    avg_final_reward: float = 0.0
    convergence_episode: int = 0
    parameter_efficiency: float = 0.0

    # Time
    runtime_seconds: float = 0.0
    timestamp: str = ""

    # Histories (not saved to summary JSON, only to npz)
    episode_rewards: List[float] = field(default_factory=list)
    od_history: List[float] = field(default_factory=list)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return JSON-safe summary dict (excludes large arrays)."""
        d = asdict(self)
        d.pop("episode_rewards", None)
        d.pop("od_history", None)
        return _make_serializable(d)

    def save(self, path: Path):
        """Save summary to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_summary_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "BenchmarkResults":
        """Load from JSON."""
        with open(path) as f:
            return cls(**json.load(f))


def _make_serializable(obj):
    """Recursively convert numpy types for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ── Benchmark Runner ──────────────────────────────────────────────────

class BenchmarkRunner:
    """Run reproducible Q-RLSTC benchmarks with checkpointing.

    Usage:
        runner = BenchmarkRunner(tier="small")
        results = runner.run()
        runner.generate_plots(results)
        runner.print_summary(results)
    """

    def __init__(
        self,
        tier: str = "small",
        seed: int = 42,
        output_dir: Optional[Path] = None,
        include_noise: bool = False,
        versions: Optional[List[str]] = None,
        compute_backend: str = "auto",
    ):
        self.tier = tier
        self.seed = seed
        self.include_noise = include_noise
        self.versions = versions or ["A", "B"]
        self.compute_backend = compute_backend
        self.config = TIER_CONFIGS[tier]

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir or Path("outputs") / f"benchmark_{ts}"
        self.plots_dir = self.output_dir / "plots"
        self.checkpoint_path = self.output_dir / ".checkpoint.json"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Checkpoint ────────────────────────────────────────────────

    def save_checkpoint(self, completed: Dict[str, BenchmarkResults]):
        """Save progress to disk."""
        state = {
            "tier": self.tier,
            "seed": self.seed,
            "completed": {
                k: v.to_summary_dict() for k, v in completed.items()
            },
        }
        with open(self.checkpoint_path, "w") as f:
            json.dump(state, f, indent=2)

    def load_checkpoint(self) -> Dict[str, BenchmarkResults]:
        """Load checkpoint if it exists, else return empty."""
        if not self.checkpoint_path.exists():
            return {}
        with open(self.checkpoint_path) as f:
            state = json.load(f)
        return {
            k: BenchmarkResults(**v)
            for k, v in state.get("completed", {}).items()
        }

    # ── Run ───────────────────────────────────────────────────────

    def run(self, resume: bool = False) -> Dict[str, BenchmarkResults]:
        """Run the full benchmark suite.

        Returns:
            Dict mapping run_key -> BenchmarkResults.
            Keys like "A_ideal", "B_ideal", "A_eagle", etc.
        """
        # Lazy imports to keep module importable without full deps
        from q_rlstc.config import QRLSTCConfig, VERSION_LABELS
        from q_rlstc.data.synthetic import generate_synthetic_trajectories
        from q_rlstc.rl.train import Trainer
        from q_rlstc.accelerator import resolve_backend, get_device_info

        results = self.load_checkpoint() if resume else {}

        # Resolve compute backend
        backend = resolve_backend(self.compute_backend)
        backend_name = backend.value
        dev_info = get_device_info()

        n_traj = self.config["n_trajectories"]
        n_epochs = self.config["n_epochs"]
        version_names = [
            f"  {v}: {VERSION_LABELS.get(v, v)}" for v in self.versions
        ]

        print(f"\n{'=' * 60}")
        print(f"Q-RLSTC Benchmark — Tier: {self.tier}")
        print(f"  Trajectories: {n_traj}, Epochs: {n_epochs}")
        print(f"  Compute: {backend_name.upper()}")
        print(f"  Noise: {'OFF (noiseless)' if not self.include_noise else 'ON (ideal + eagle + heron)'}")
        print(f"  Versions:")
        for vn in version_names:
            print(vn)
        print(f"  Output: {self.output_dir}")
        print(f"{'=' * 60}")

        # Shared dataset
        print("\nGenerating synthetic trajectories...")
        dataset = generate_synthetic_trajectories(
            n_trajectories=n_traj,
            n_segments_range=(2, 4),
            seed=self.seed,
        )
        print(f"  Generated {dataset.n_trajectories} trajectories")

        # Determine runs
        noise_models = ["ideal"]
        if self.include_noise:
            noise_models += ["eagle", "heron"]

        for version in self.versions:
            for noise in noise_models:
                key = f"{version}_{noise}"
                if key in results:
                    print(f"\n⏭ Skipping {key} (already in checkpoint)")
                    continue

                label = VERSION_LABELS.get(version, version)
                noise_str = "noiseless" if noise == "ideal" else noise
                print(f"\n{'─' * 50}")
                print(f"Running: {label} — {noise_str}")
                print(f"{'─' * 50}")

                config = QRLSTCConfig(version=version,
                                     compute_backend=backend_name)
                config.training.n_epochs = n_epochs
                config.rl.gamma = 0.90
                config.spsa.c = 0.10
                config.vqdqn.shots_train = 512
                config.vqdqn.shots_eval = 4096
                config.noise.use_noise = (noise != "ideal")
                config.noise.noise_model = noise if noise != "ideal" else "ideal"

                trainer = Trainer(dataset, config)
                info = trainer.agent.get_circuit_info()

                print(f"  Circuit: {info.n_qubits}q, {info.n_params} params, "
                      f"depth={info.depth}")

                start = time.time()
                result = trainer.train(n_epochs=n_epochs, verbose=True)
                elapsed = time.time() - start

                # Metrics
                ep_rewards = result.episode_rewards
                avg_final = (
                    float(np.mean(ep_rewards[-10:]))
                    if len(ep_rewards) >= 10
                    else float(np.mean(ep_rewards)) if ep_rewards else 0.0
                )
                conv_ep = len(ep_rewards)
                max_r = max(ep_rewards) if ep_rewards else 0
                if max_r > 0:
                    target = 0.9 * max_r
                    for i, r in enumerate(ep_rewards):
                        if r >= target:
                            conv_ep = i
                            break
                param_eff = avg_final / info.n_params if avg_final > 0 else 0

                br = BenchmarkResults(
                    version=version,
                    version_label=VERSION_LABELS.get(version, version),
                    tier=self.tier,
                    noise_model=noise,
                    compute_backend=backend_name,
                    seed=self.seed,
                    n_qubits=info.n_qubits,
                    n_layers=info.n_layers,
                    n_params=info.n_params,
                    circuit_depth=info.depth,
                    feature_dim=8 if version.upper() == "B" else 5,
                    readout_mode=trainer.agent.readout_mode,
                    delta_od=(result.od_history[0] - result.final_od
                              if result.od_history else 0),
                    final_od=result.final_od,
                    final_f1=result.final_f1,
                    avg_final_reward=avg_final,
                    convergence_episode=conv_ep,
                    parameter_efficiency=float(param_eff),
                    runtime_seconds=elapsed,
                    timestamp=datetime.now().isoformat(),
                    episode_rewards=ep_rewards,
                    od_history=result.od_history,
                )
                results[key] = br

                # Save checkpoint after each run
                self.save_checkpoint(results)

                # Save NPZ history
                np.savez(
                    self.output_dir / f"{key}_history.npz",
                    episode_rewards=ep_rewards,
                    od_history=result.od_history,
                )

                print(f"  ΔOD={br.delta_od:.4f}  F1={br.final_f1:.4f}  "
                      f"Runtime={elapsed:.1f}s")

        # Save combined metrics
        summary = {k: v.to_summary_dict() for k, v in results.items()}
        summary_path = self.output_dir / "metrics_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return results

    # ── Plots ─────────────────────────────────────────────────────

    def generate_plots(self, results: Dict[str, BenchmarkResults]):
        """Generate all plots from benchmark results."""
        from .plot_utils import (
            plot_learning_curves,
            plot_od_convergence,
            plot_metric_comparison,
            plot_noise_impact,
            plot_epsilon_schedule,
            plot_circuit_summary,
        )

        print("\nGenerating plots...")
        p = self.plots_dir

        # 1. Learning curves (A vs B, ideal)
        r_a = results.get("A_ideal")
        r_b = results.get("B_ideal")
        if r_a:
            plot_learning_curves(
                rewards_a=r_a.episode_rewards,
                rewards_b=r_b.episode_rewards if r_b else None,
                out_path=p / "learning_curves.png",
            )
            print("  ✓ learning_curves.png")

        # 2. OD convergence
        if r_a and r_a.od_history:
            plot_od_convergence(
                od_a=r_a.od_history,
                od_b=r_b.od_history if r_b else None,
                out_path=p / "od_convergence.png",
            )
            print("  ✓ od_convergence.png")

        # 3. Metric comparison
        if r_a and r_b:
            from q_rlstc.config import VERSION_LABELS
            metrics = {
                VERSION_LABELS.get(r_a.version, "A"): {
                    "F1": r_a.final_f1,
                    "ΔOD": r_a.delta_od,
                    "Param Eff": r_a.parameter_efficiency,
                    "Avg Reward": r_a.avg_final_reward,
                },
                VERSION_LABELS.get(r_b.version, "B"): {
                    "F1": r_b.final_f1,
                    "ΔOD": r_b.delta_od,
                    "Param Eff": r_b.parameter_efficiency,
                    "Avg Reward": r_b.avg_final_reward,
                },
            }
            plot_metric_comparison(metrics, out_path=p / "metric_comparison.png")
            print("  ✓ metric_comparison.png")

        # 4. Noise impact
        noise_rewards = {}
        for key, res in results.items():
            if key.startswith("A_"):
                noise_rewards[key.split("_", 1)[1]] = res.episode_rewards
        if len(noise_rewards) > 1:
            plot_noise_impact(noise_rewards, out_path=p / "noise_impact.png")
            print("  ✓ noise_impact.png")

        # 5. Epsilon schedule
        n_ep = len(r_a.episode_rewards) if r_a else 100
        plot_epsilon_schedule(n_episodes=n_ep, out_path=p / "epsilon_schedule.png")
        print("  ✓ epsilon_schedule.png")

        # 6. Circuit summary
        if r_a:
            info_a = {
                "n_qubits": r_a.n_qubits, "n_layers": r_a.n_layers,
                "n_params": r_a.n_params, "depth": r_a.circuit_depth,
                "feature_dim": r_a.feature_dim,
                "readout_mode": r_a.readout_mode,
            }
            info_b = None
            if r_b:
                info_b = {
                    "n_qubits": r_b.n_qubits, "n_layers": r_b.n_layers,
                    "n_params": r_b.n_params, "depth": r_b.circuit_depth,
                    "feature_dim": r_b.feature_dim,
                    "readout_mode": r_b.readout_mode,
                }
            plot_circuit_summary(info_a, info_b,
                                 out_path=p / "circuit_summary.png")
            print("  ✓ circuit_summary.png")

        print(f"\nAll plots saved to: {p}")

    # ── Summary ───────────────────────────────────────────────────

    def print_summary(self, results: Dict[str, BenchmarkResults]):
        """Print formatted comparison table with rich metrics."""
        from q_rlstc.config import VERSION_LABELS

        # Detect backend from results
        backends_used = set(r.compute_backend for r in results.values())
        backend_str = ", ".join(sorted(backends_used)).upper() or "CPU"

        print(f"\n{'=' * 105}")
        print(f"  BENCHMARK SUMMARY — Tier: {self.tier} — Backend: {backend_str}")
        print(f"{'=' * 105}")

        header = (
            f"  {'Run':<28} {'Qubits':>6} {'Params':>7} "
            f"{'Episodes':>8} {'Init OD':>8} {'Final OD':>8} "
            f"{'ΔOD':>8} {'OD Impr%':>8} {'F1':>7} "
            f"{'AvgRew':>8} {'Conv.Ep':>8} {'ParamEff':>9} {'Time':>8}"
        )
        print(header)
        sep = (
            f"  {'─' * 28} {'─' * 6} {'─' * 7} "
            f"{'─' * 8} {'─' * 8} {'─' * 8} "
            f"{'─' * 8} {'─' * 8} {'─' * 7} "
            f"{'─' * 8} {'─' * 8} {'─' * 9} {'─' * 8}"
        )
        print(sep)

        total_time = 0.0
        for key in sorted(results.keys()):
            r = results[key]
            label = VERSION_LABELS.get(r.version, r.version)
            noise_str = "noiseless" if r.noise_model == "ideal" else r.noise_model
            run_label = f"{label} [{noise_str}]"

            n_ep = len(r.episode_rewards) if r.episode_rewards else 0
            init_od = r.od_history[0] if r.od_history else 0.0
            od_impr = (r.delta_od / init_od * 100) if init_od > 0 else 0.0
            total_time += r.runtime_seconds

            row = (
                f"  {run_label:<28} {r.n_qubits:>6} {r.n_params:>7} "
                f"{n_ep:>8} {init_od:>8.4f} {r.final_od:>8.4f} "
                f"{r.delta_od:>8.4f} {od_impr:>7.1f}% {r.final_f1:>7.4f} "
                f"{r.avg_final_reward:>8.4f} {r.convergence_episode:>8} "
                f"{r.parameter_efficiency:>9.6f} "
                f"{r.runtime_seconds:>7.1f}s"
            )
            print(row)

        print(sep)
        print(f"  Total runtime: {total_time:.1f}s")

        # Noise resilience
        ideal_a = results.get("A_ideal")
        if ideal_a and any(k != "A_ideal" and k.startswith("A_")
                           for k in results):
            print(f"\n  Noise Resilience (vs Classical Parity noiseless):")
            for key, r in sorted(results.items()):
                if key.startswith("A_") and key != "A_ideal" \
                        and ideal_a.avg_final_reward > 0:
                    ratio = r.avg_final_reward / ideal_a.avg_final_reward
                    status = "✓" if ratio > 0.8 else "⚠"
                    print(f"    {key:<14}: {ratio:.3f} {status}")

        print(f"\n  Output: {self.output_dir}")
        print(f"{'=' * 105}\n")

