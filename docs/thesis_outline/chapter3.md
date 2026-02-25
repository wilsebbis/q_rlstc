# Chapter 3: Experimental Setup

## 3.1 Hardware and Software Environment

| Component | Specification |
|---|---|
| Machine | Apple MacBook Pro (M-series) |
| Python | 3.12.x |
| Quantum backend | Qiskit 1.x (Aer statevector simulator) |
| OS | macOS |
| Reproducibility | Git hash, package versions, wall-clock timestamp recorded per run via `_collect_env_metadata()` |

All quantum experiments use simulation. No experiments were executed on physical quantum hardware.

---

## 3.2 Dataset

This study operates in the **small-data regime** defined in Chapter 2: 30 trajectories, 2 epochs, 54 training episodes per seed.

| Property | Value |
|---|---|
| Source | T-Drive taxi dataset (Beijing, 2008–2009) |
| Preprocessing | `Tdrive_norm_traj` (normalised) |
| Cluster centres | `tdrive_clustercenter` (pre-computed) |
| Training / Validation | 27 / 3 trajectories (90/10 deterministic split per seed) |
| State dimensionality | 5 (Version D: $OD_s$, $OD_n$, $OD_b$, $L_b$, $L_f$) |
| Action space | Binary: EXTEND (0) / CUT (1) |

The 30-trajectory budget is intentionally constrained — the research question examines function approximation under data scarcity. Results under larger data regimes with gradient-based optimisation may differ (see §1.6).

---

## 3.3 Evaluation Metrics

### Primary Metric: ValCR (Validation Competitive Ratio)

$$\text{ValCR}(\text{val}) = \frac{OD(\text{policy}, \text{val})}{\max(\text{basesim}(\text{val}), \epsilon)}$$

where $OD(\text{policy}, \text{val})$ is the mean IED between each segment and its nearest cluster centre on the validation fold, and $\text{basesim}(\text{val})$ is the baseline OD computed strictly on that same validation fold under a fixed "always-extend" (no-cut) baseline policy using the same cluster centres. $\epsilon = 10^{-8}$ provides denominator stabilisation. **Lower = better.** ValCR has a structural degeneracy characterised in Chapter 4.

> **Fold-Dependent BaseSim.** By computing basesim dynamically per validation fold rather than loading a static scalar from the global dataset, CR becomes a true fold-relative competitive ratio. This mitigates denominator-driven variance inflation across seeds (a metric pathology detailed in Chapter 4).

> **Median CR.** We additionally report median per-trajectory CR to mitigate sensitivity to outlier denominators. If mean CR and median CR diverge substantially, the mean is unreliable.

### Diagnostic Variants

| Metric | Formula | Purpose |
|---|---|---|
| **nValCR** | Mean of $(\text{IED} / \text{segment\_length}) / \text{basesim}$ | Removes per-segment length coupling |
| **wValCR** | $\text{Total\_IED} / \text{total\_points} / \text{basesim}$ | Length-weighted correction |

### Secondary Metrics

| Metric | Measures |
|---|---|
| SSE | Within-cluster compactness |
| CUT% | Segmentation aggressiveness |
| Q-margin | $Q(\text{extend}) - Q(\text{cut})$ — policy preference direction |
| #Segments | Segmentation volume |

### Reporting Format: Budget-Constrained Pareto

Raw ValCR comparisons across agents with different CUT rates are misleading (Chapter 4). The primary reporting format is:

- **Best ValCR at matched CUT thresholds:** CUT ≤ {5%, 10%, 20%, 30%, 40%}
- **Pareto frontier overlay:** learned agents plotted against the D1 random baseline curve
- **OD and basesim reported separately** to enable denominator audit
- **Median CR alongside mean CR** as a robustness check

This eliminates degenerate "wins" from fragmentation and guards against denominator-driven outliers.

---

## 3.4 Cross-Comparability Requirements

Every pipeline component except the function approximator is identical across all RL models:

| Component | Shared? |
|---|---|
| State representation, action space, reward, replay buffer, ε-schedule, target network, loss, dataset, distance metric, L_MIN, Q/TD clamping | ✅ Identical |
| **Function approximator** | ❌ Experimental variable |
| Optimiser | SPSA for quantum + SPSA cohort; Adam for Adam cohort |

---

## 3.5 Experiment Matrix

### Diagnostic Experiments (D1–D5)

| ID | Name | Variable | Measures |
|---|---|---|---|
| **D1** | ValCR vs CUT% | Random CUT probability (0%–100%) | Metric degeneracy; raw/nValCR/wValCR |
| **D2** | Q-margin | Per-epoch $Q(\text{ext}) - Q(\text{cut})$ | Policy bias formation |
| **D3** | Training action dist | Per-epoch CUT% | Action distribution drift |
| **D4** | Policy basin test | Forced all-cut / all-extend / alternating | Basin structure |
| **D5** | Buffer histogram | Buffer vs on-policy CUT% | Replay drift |

### Core Benchmarks (E1–E6)

| ID | Name | Variable | Measures |
|---|---|---|---|
| **E1** | Core Quantum Utility | VQ-DQN vs all controls | Pareto performance |
| **E2** | NISQ Viability | Eagle / Heron noise profiles | Noise degradation |
| **E3** | Shot Sensitivity | 128 / 512 / 2,048 shots | Sampling noise floor |
| **E4** | Drift Resilience | Temporal distribution shift | Robustness |
| **E5** | Low-Data | 10% / 25% / 50% data fractions | Sample efficiency |
| **E6** | Stabilisation Ablation | Incremental reward components | Load-bearing vs cosmetic |

### Ablation and Scalability

| ID | Name | Variable | Measures |
|---|---|---|---|
| **AB1** | Entanglement | No-CNOT vs linear CNOT | Mechanism evidence |
| **A1** | $OD_b$ Ablation | 5D vs 4D state | Expert baseline contribution |
| **S1** | Inference Timing | 250–2,500 trajectories | Wall-clock overhead |

---

## 3.6 Multi-Seed Protocol

**Seeds:** 42, 123, 7, 99, 2025 (5 seeds per condition).

| Component | Method |
|---|---|
| Aggregation | Mean ± std of per-seed best-epoch ValCR |
| Best-epoch selection | Argmin ValCR across epochs; earliest epoch breaks ties |
| Significance | Mann-Whitney U (nonparametric pairwise) |
| Effect size | Cohen's d with interpretation labels |
| Confidence interval | Bootstrap 95% CI on mean difference (10,000 resamples) |

---

## 3.7 Reproducibility

### Seeded RNGs

Each run seeds: `np.random.seed(seed)`, `random.seed(seed)`, `ReplayBuffer(seed=seed)`, `TrajectoryScheduler(seed=seed)`. Statevector simulation is fully deterministic (no sampling noise).

### Run Identification

Unique tuple: `(seed, dataset_amount, epochs, model_id, optimizer_kind, shots, noise_model, git_hash)`. Identical tuples produce identical results under statevector simulation.

### Reproduction Commands

```bash
# Full multi-seed E1 (~90 min)
python experiments/run_thesis_experiments.py \
    --experiments E1 --amount 30 --epochs 2 \
    --seeds 42,123,7,99,2025 --output-dir results/thesis_multiseed

# Significance tests
python experiments/run_significance_test.py \
    results/thesis_multiseed/thesis_results_*.json

# Robustness sweeps
python experiments/run_thesis_experiments.py \
    --experiments E2,E3 --amount 30 --epochs 2 \
    --seeds 42,123,7,99,2025 --output-dir results/thesis_robustness

# Entanglement ablation
python experiments/run_thesis_experiments.py \
    --experiments AB1 --amount 30 --epochs 2 \
    --seeds 42,123,7,99,2025 --output-dir results/thesis_ablation
```