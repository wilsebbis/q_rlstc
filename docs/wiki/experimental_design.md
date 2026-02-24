# Experimental Design


---

## Cross-Comparability Requirements

If the quantum and classical implementations differ in anything beyond the function approximator, results are not attributable to the quantum component.

### What Must Be Identical

| Component | Classical Control | Quantum Experiment | Same? |
|---|---|---|---|
| State representation | 5D vector (from `rlstc_mdp.py`) | 5D vector | ✅ |
| Action space | {extend, cut} | {extend, cut} | ✅ |
| Reward function | OD delta + CUT\_PENALTY + EXTEND\_COST | Same | ✅ |
| Replay buffer | Size 5,000, uniform | Size 5,000, uniform | ✅ |
| Exploration | ε-greedy, same schedule | ε-greedy, same schedule | ✅ |
| Target network | Double DQN, same freq | Double DQN, same freq | ✅ |
| **Function approximator** | **MLP (various sizes)** | **VQ-DQN (5q, 3L HEA)** | ❌ Variable |
| Optimizer | SPSA (same hyperparams) | SPSA (same hyperparams) | ✅ |
| Loss | Huber (δ=1.0) | Huber (δ=1.0) | ✅ |
| Dataset | Same seed, same split | Same seed, same split | ✅ |
| Distance metric | IED (rlstc\_trajdistance.py) | IED | ✅ |
| Q-value clamping | ±10.0 | ±10.0 | ✅ |
| TD target clamping | ±10.0 | ±10.0 | ✅ |
| L\_MIN constraint | 3 | 3 | ✅ |

### Common Pitfalls

| Pitfall | Why It Breaks Comparability |
|---|---|
| Different optimizers (Adam vs SPSA) | Optimizer effects dominate approximator effects |
| Different batch sizes | Affects gradient variance independently |
| Different shot counts | Added noise is an uncontrolled variable |
| Different random seeds | Trajectory order and exploration path differ |
| Quantum with noise, classical without | Measuring noise tolerance, not approximation quality |

## Classical Baselines

| Control | Architecture | Params | Purpose |
|---|---|---|---|
| **A: Linear** | 5→2 (no hidden layers) | 12 | Is the problem trivially linear? |
| **B: Medium MLP** | 5→64→2 | 514 | Moderate capacity baseline |
| **C: Deep MLP** | 5→32→32→2 | 1,314 | High capacity — classical ceiling |

**Critical:** All controls use SPSA (not SGD/Adam), identical to the quantum agent. This isolates the function approximator as the only variable.

## Primary Metrics

| Metric | Measures | Report As |
|---|---|---|
| **ValCR** (raw) | Mean segment-to-center IED / base similarity | Table + Pareto |
| **nValCR** (per-point) | Mean of (IED/segment\_length) / base similarity | D1 diagnostic |
| **wValCR** (length-weighted) | Total\_IED / total\_points / base similarity | D1 diagnostic |
| **CUT%** | Fraction of actions that are CUT | Per-epoch |
| **#Segments** | Total segments produced | Per-epoch |
| **Q-margin** | Q(extend) − Q(cut) | D2 diagnostic |
| **Parameter count** | Total trainable parameters | One-time |

### Metric Pathology Awareness

Raw ValCR is **structurally degenerate**: IED grows with segment length, so cutting always lowers the metric. This is diagnosed by D1 and mitigated via budget-constrained reporting (Pareto table).

## Experiment Matrix

### Diagnostic Experiments (D1–D5)

| ID | Name | Variable | Measures |
|---|---|---|---|
| **D1** | ValCR vs CUT% | Random CUT probability (0%–100%) | Metric degeneracy; reports raw, nValCR, wValCR |
| **D2** | Q-margin | Per-epoch Q(ext)−Q(cut) | Policy bias formation |
| **D3** | Training action dist | Per-epoch CUT% in training | Action distribution drift |
| **D4** | Policy basin test | Forced all-cut / all-extend / alternating | Basin structure |
| **D5** | Buffer histogram | Buffer CUT% vs on-policy CUT% | Replay distribution drift |

### Core Benchmarks (E1–E6)

| ID | Name | Variable | Measures |
|---|---|---|---|
| **E1** | Core Quantum Utility | VQ-DQN vs Controls A/B/C | Parameter efficiency |
| **E2** | NISQ Viability | Eagle / Heron noise models | Noise degradation |
| **E3** | Shot Sensitivity | 128 / 512 / 2048 shots | Sampling noise floor |
| **E4** | Drift Resilience | Temporal distribution shift | Robustness |
| **E5** | Low-Data | 10% / 25% / 50% data fractions | Sample efficiency |
| **E6** | Version Progression | Circuit architecture variants | Design ablation |

### Scalability (S1)

| ID | Name | Variable | Measures |
|---|---|---|---|
| **S1** | Inference timing | 250–1000 trajectories | Wall-clock overhead |

### Multi-Seed Protocol

All E-series experiments support `--seeds` for multi-seed runs:

```bash
python experiments/run_thesis_experiments.py --experiments E1 --amount 50 --epochs 3 \
    --seeds 42,123,7,99,2025
```

Reports mean ± std across seeds for ValCR, CUT%, Q-margin, and #segments.

## Analysis Plan

### Pareto Frontier Analysis
- Plot: agents overlaid on D1 random baseline curve (ValCR vs CUT%)
- Table: best ValCR at CUT ≤ {5%, 10%, 20%, 30%, 40%, 50%}
- Purpose: eliminates degenerate solutions from headline results

### Q-Learning Dynamics
- Q-value evolution pre/post clamping
- Q-margin trends across epochs (VQ-DQN vs classical)
- Replay buffer drift quantification

### NISQ Sensitivity
- ValCR vs shot count curves
- Noise model degradation (Eagle, Heron)
- Training stability under finite sampling

---

**Next:** [Debugging Guide →](debugging.md)
