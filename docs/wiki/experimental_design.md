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
| Distance metric | IED (trajdistance.py) | IED (trajdistance.py) | ✅ |
| Preprocessing | MDL simplification | MDL simplification | ✅ |

### Common Pitfalls

| Pitfall | Why It Breaks Comparability |
|---|---|
| Adam for classical, SPSA for quantum | Optimizer effects dominate approximator effects |
| Different batch sizes | Affects gradient variance independently |
| Different learning rate schedules | Convergence changes are optimizer artifacts |
| Different shot counts | Added noise is an uncontrolled variable |
| Different random seeds | Trajectory order and exploration path differ |
| Quantum with noise, classical without | Measuring noise tolerance, not approximation quality |
| Different distance metrics | IED vs. OD proxy produces different reward signals |

## Classical Baselines

| Control | Architecture | Params | Purpose |
|---|---|---|---|
| **A: Parameter-matched** | 5→4→2 MLP | 30 | Quantum gains from structure or just right param count? |
| **B: Architecture-matched** | 5→64→2 MLP | ~450 | Classical ceiling — how well can unconstrained classical do? |
| **C: Linear** | 5→2 | 12 | Is the problem trivially linear? |
| **D: RLSTCcode direct** | 5→64→2 MLP (TF/Keras) | ~450 | Original paper implementation for ground truth |

**Critical:** Controls A–C use SPSA (not SGD/Adam with backprop). Control D uses original SGD.

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
| **E6** | Version A vs. B vs. C vs. D | Ideal backend | Version comparison |
| **E7** | Q-RLSTC (D) vs. RLSTCcode classical | Same T-Drive data | Classical parity validation |

### Version-Specific Experiments

| Experiment | Version | Variable | Measures |
|---|---|---|---|
| **V1** | A vs. D | State features (Q-RLSTC proxy vs. VLDB exact) | Feature design impact |
| **V2** | B | 5 vs. 8 qubits, standard vs. multi-observable readout | Hilbert space utilisation |
| **V3** | C | EQC vs. HEA ansatz, SAC vs. ε-greedy | Quantum-native architecture gains |
| **V4** | C | DROP action enabled vs. disabled | Noise filtering benefit |
| **V5** | D | SKIP action enabled vs. disabled | Efficiency of fast-forward |
| **V6** | C | Adaptive (32→512) vs. fixed (512) shots | Shot efficiency |
| **V7** | C | SPSA vs. m-SPSA | Momentum benefit under shot noise |

### Cross-System Comparison (via `run_cross_comparison.py`)

| Experiment | Classical Arm | Quantum Arm | Data | Measures |
|---|---|---|---|---|
| **X1** | RLSTCcode (5→64→2, SGD) | Q-RLSTC Version D (5q HEA, SPSA) | T-Drive 500 | OD parity |
| **X2** | RLSTCcode (same) | Q-RLSTC Version A (5q HEA, SPSA) | T-Drive 500 | Feature design gap |
| **X3** | MLP(30, SPSA) | Q-RLSTC Version A (5q, SPSA) | Synthetic | Pure approximator comparison |

### Thesis-Level (Additional)

| Experiment | Variable | Measures |
|---|---|---|
| **E8** | Angle vs. ZZFeatureMap vs. amplitude | Best encoding for RL state |
| **E9** | Linear vs. ring vs. full entanglement | Connectivity vs. noise |
| **E10** | SPSA vs. parameter-shift | Gradient estimation quality |
| **E11** | Pre-train ideal, fine-tune noisy | Noise adaptation |

## Analysis Plan

### Boundary Sharpness Analysis

1. **Distribution comparison** — Histogram of sharpness at predicted vs. true boundaries (quantum vs. classical)
2. **Decision boundary** — Plot `Q(cut) − Q(extend)` vs. boundary sharpness
3. **False positive analysis** — Are low-sharpness cuts from noise (quantum) or overfitting (classical)?

### Clustering Quality by Complexity

Segment trajectories by number of true boundaries (2, 4, 8, 16) and measure OD for each. Does quantum degrade faster on complex trajectories?

### Cluster Assignment Stability

Across 5 seeds: how consistent are cluster assignments? Higher variance in quantum → noise dominates.

### Version Progression Analysis

Run E6 and plot: OD vs. version (A→B→C→D). Does architectural sophistication improve results, or does simplicity win?

---

**Next:** [Debugging Guide →](debugging.md)
