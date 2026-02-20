# Benchmarking

The `BenchmarkRunner` provides reproducible, tiered experiments with checkpointing, automatic plot generation, and a rich summary table.

## Quick Start

```bash
# Run both versions, noiseless, auto-detect backend
python experiments/run_full_benchmark.py

# Medium tier, Quantum Enhanced only, with noise
python experiments/run_full_benchmark.py --tier medium --version B --noise

# Compare compute backends
python experiments/run_full_benchmark.py --compare-backends
```

## Version Labels

| Flag | Internal | Human Label | Description |
|---|---|---|---|
| `--version A` | Version A | **Classical Parity (5q)** | Same algorithm as classical RLSTC, but with quantum distance computation. Direct 1:1 comparison baseline. |
| `--version B` | Version B | **Quantum Enhanced (8q)** | Additional quantum features: 8 qubits, data reuploading, richer encoding, optimised readout. |
| `--version both` | A + B | Both | Default — runs both for side-by-side comparison. |

## Tiers

| Tier | Trajectories | Epochs | Approx. Time |
|---|---|---|---|
| `small` | 20 | 3 | ~5 min |
| `medium` | 50 | 8 | ~15 min |
| `large` | 100 | 15 | ~45+ min |

## CLI Flags

```
--tier small|medium|large     Benchmark tier (default: small)
--version A|B|both            Version selection (default: both)
--noise                       Include Eagle + Heron noise runs
--backend auto|cpu|mlx|cuda   Compute backend (default: auto)
--compare-backends            Run on all available backends
--seed N                      Random seed (default: 42)
--resume                      Resume from checkpoint
--output-dir PATH             Override output directory
```

## Summary Table

The benchmark prints a 13-column summary table:

```
Run                          Qubits  Params Episodes  Init OD  Final OD      ΔOD  OD Impr%      F1   AvgRew  Conv.Ep  ParamEff     Time
Classical Parity (5q) [noiseless]     5      20       60   1.2345   0.8901   0.3444    27.9% 0.7234   0.4521       23  0.022605    12.3s
Quantum Enhanced (8q) [noiseless]     8      56       60   1.2345   0.7890   0.4455    36.1% 0.8012   0.5678       18  0.010139    24.7s
```

**Columns:**

| Column | Description |
|---|---|
| Run | Version label + noise status |
| Qubits | Number of qubits in VQ-DQN circuit |
| Params | Trainable variational parameters |
| Episodes | Total training episodes completed |
| Init OD | Overall Distance at start |
| Final OD | Overall Distance at end |
| ΔOD | Absolute OD improvement |
| OD Impr% | Percentage OD improvement |
| F1 | Segmentation F1 score |
| AvgRew | Average reward over last 10 episodes |
| Conv.Ep | Episode at which 90% of max reward was reached |
| ParamEff | Avg reward / number of parameters |
| Time | Wall-clock runtime |

## Checkpointing

The benchmark saves progress after each run to `.checkpoint.json` in the output directory. Use `--resume` to continue from where you left off if a run is interrupted.

## Generated Plots

The benchmark auto-generates these plots in `outputs/benchmark_*/plots/`:

| File | Content |
|---|---|
| `learning_curves.png` | Reward curves A vs B |
| `od_convergence.png` | OD vs epoch convergence |
| `metric_comparison.png` | Grouped bar chart of all metrics |
| `epsilon_schedule.png` | ε-greedy exploration decay |
| `circuit_summary.png` | VQ-DQN circuit comparison table |
| `noise_impact.png` | Noise resilience (only with `--noise`) |

## Python API

```python
from q_rlstc.visualization import BenchmarkRunner

runner = BenchmarkRunner(
    tier="medium",
    versions=["A", "B"],
    compute_backend="auto",
    include_noise=False,
)

results = runner.run()
runner.generate_plots(results)
runner.print_summary(results)
```

## Output Structure

```
outputs/benchmark_20260220_103000/
├── .checkpoint.json
├── metrics_summary.json
├── A_ideal_history.npz
├── B_ideal_history.npz
└── plots/
    ├── learning_curves.png
    ├── od_convergence.png
    ├── metric_comparison.png
    ├── epsilon_schedule.png
    └── circuit_summary.png
```

---

← [Compute Backends](compute_backends.md) | [Experimental Design](experimental_design.md) →
