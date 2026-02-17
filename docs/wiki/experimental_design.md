# Experimental Design

[← Back to README](../../README.md) · [Noise & Hardware](noise_and_hardware.md) · **Experimental Design** · [Debugging →](debugging.md)

---

## Cross-Comparability Requirements

If the quantum and classical implementations differ in anything beyond the function approximator, results are not attributable to the quantum component.

### What Must Be Identical

| Component | Classical Control | Quantum Experiment | Same? |
|---|---|---|---|
| Feature extraction | StateFeatureExtractor | StateFeatureExtractor | ✅ |
| State representation | 5D vector | 5D vector | ✅ |
| Action space | {extend, cut} | {extend, cut} | ✅ |
| Reward function | OD + boundary sharpness | OD + boundary sharpness | ✅ |
| Replay buffer | Size 5000, uniform | Size 5000, uniform | ✅ |
| Exploration | ε-greedy, same schedule | ε-greedy, same schedule | ✅ |
| Target network | Double DQN, same freq | Double DQN, same freq | ✅ |
| **Function approximator** | **MLP (5→H→2)** | **VQ-DQN (5q, 2 layers)** | ❌ Variable |
| Optimizer | SPSA (same hyperparams) | SPSA (same hyperparams) | ✅ |
| Loss | Huber (δ=1.0) | Huber (δ=1.0) | ✅ |
| Dataset | Same seed | Same seed | ✅ |

### Common Pitfalls

| Pitfall | Why It Breaks Comparability |
|---|---|
| Adam for classical, SPSA for quantum | Optimizer effects dominate approximator effects |
| Different batch sizes | Affects gradient variance independently |
| Different learning rate schedules | Convergence changes are optimizer artifacts |
| Different shot counts | Added noise is an uncontrolled variable |
| Different random seeds | Trajectory order and exploration path differ |
| Quantum with noise, classical without | Measuring noise tolerance, not approximation quality |

## Classical Baselines

| Control | Architecture | Params | Purpose |
|---|---|---|---|
| **A: Parameter-matched** | 5→4→2 MLP | 30 | Quantum gains from structure or just right param count? |
| **B: Architecture-matched** | 5→64→2 MLP | ~450 | Classical ceiling — how well can unconstrained classical do? |
| **C: Linear** | 5→2 | 12 | Is the problem trivially linear? |

**Critical:** All controls use SPSA (not SGD/Adam with backprop).

## Primary Metrics

| Metric | Measures | Report As |
|---|---|---|
| **Overall Distance (OD)** | Clustering quality at convergence | Mean ± std over 5 seeds |
| **Segmentation F1** | Boundary detection accuracy | Against ground truth |
| **Convergence rate** | Episodes to reach 90% of final OD | Lower = more sample-efficient |
| **Parameter count** | Model complexity | Quantum advantage claim |

## Quantum-Specific Metrics

| Metric | What It Reveals |
|---|---|
| **Noise resilience ratio** | `OD_noisy / OD_ideal` — noise degradation |
| **Parameter efficiency** | `OD_quantum(20p) / OD_classical(20p)` |
| **Circuit fidelity** | Validates depth choice |
| **Shot sensitivity** | OD vs. shots — noise floor |
| **Gradient variance** | SPSA gradient norm variance |

## Experimental Matrix

### Project-Level (Minimum)

| Experiment | Variable | Fixed | Measures |
|---|---|---|---|
| **E1** | VQ-DQN vs. MLP(30) vs. MLP(450) vs. Linear | All else, SPSA | Core quantum utility |
| **E2** | Ideal vs. Eagle vs. Heron | VQ-DQN | NISQ viability |
| **E3** | 128, 256, 512, 1024, 4096 shots | VQ-DQN ideal | Noise floor |
| **E4** | 1, 2, 3 variational layers | Same qubit count | Expressivity vs. noise |
| **E5** | 2, 4, 8, 16 true boundaries | All systems | Dataset scaling |

### Thesis-Level (Additional)

| Experiment | Variable | Measures |
|---|---|---|
| **E6** | Angle vs. ZZFeatureMap vs. amplitude | Best encoding for RL state |
| **E7** | Linear vs. ring vs. full entanglement | Connectivity vs. noise |
| **E8** | SPSA vs. parameter-shift | Gradient estimation quality |
| **E9** | Pre-train ideal, fine-tune noisy | Noise adaptation |

## Analysis Plan

### Boundary Sharpness Analysis

1. **Distribution comparison** — Histogram of sharpness at predicted vs. true boundaries (quantum vs. classical)
2. **Decision boundary** — Plot `Q(cut) − Q(extend)` vs. boundary sharpness
3. **False positive analysis** — Are low-sharpness cuts from noise (quantum) or overfitting (classical)?

### Clustering Quality by Complexity

Segment trajectories by number of true boundaries (2, 4, 8, 16) and measure OD for each. Does quantum degrade faster on complex trajectories?

### Cluster Assignment Stability

Across 5 seeds: how consistent are cluster assignments? Higher variance in quantum → noise dominates.

---

**Next:** [Debugging Guide →](debugging.md)
