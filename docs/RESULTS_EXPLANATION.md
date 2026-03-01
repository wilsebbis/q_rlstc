# Q-RLSTC Results Explanation

**Prepared for advisor meeting — February 2026**

---

## 1. Executive Summary

Q-RLSTC replaces the classical DQN policy network in the RLSTC trajectory clustering framework with a **34-parameter Variational Quantum Deep Q-Network (VQ-DQN)** — a 5-qubit, 3-layer hardware-efficient ansatz. The key research question: *does the quantum circuit provide any advantage over classical networks for this RL-driven sub-trajectory segmentation task?*

**Bottom line:** Under controlled gradient-free (SPSA) training with limited data, the VQ-DQN is competitive with — and often beats — classical networks that have up to **39× more parameters**. However, classical networks trained with SGD on 100× more data still achieve the best absolute clustering quality. The quantum advantage is **regime-specific**: parameter efficiency under constrained optimisation, not universal superiority.

---

## 2. What We Measure

### Competitive Ratio (ValCR)

$$\text{ValCR} = \frac{\text{OD}_{\text{segmented}}}{\text{OD}_{\text{baseline}}}$$

- **OD** = average over-distance (mean IED from each sub-trajectory to its cluster centre)
- **Lower ValCR is better**: < 1.0 means the agent produces tighter clusters than no segmentation
- ValCR = 1.0 is parity; > 1.0 is worse than not segmenting at all

### CUT%
Percentage of timesteps where the agent chooses to cut (create a new segment). Higher CUT% = more segments.

### Key Diagnostic Discovery
We discovered and fixed a **denominator bug** in ValCR: the baseline OD (`basesim`) was a static precomputed scalar that didn't change with the validation fold. This decoupled the metric from the actual data being evaluated, inflated cross-seed variance, and produced artificially high CR values (>1.0 everywhere). The fix: compute `basesim` dynamically per validation fold. All results below use the **corrected metric**.

---

## 3. Experiment Timeline & Result Directories

| Directory | Date | What it tests | Status |
|---|---|---|---|
| `thesis_pareto` | Feb 24 | D1 random sweep + 1-epoch E1 (4 models, 1 seed) | Early baseline, pre-metric fix reference |
| `thesis_qfix` | Feb 24 | First run with corrected metric (4 models, 1 seed, 2 epochs) | Confirmed fix works |
| `thesis_nvalcr` | Feb 24 | D1 diagnostic with nValCR/wValCR alternative metrics | Metric analysis |
| `thesis_multiseed` | Feb 25 | 5-seed multi-model run (9 models, 2 epochs, 30 trajs) — **pre-fix basesim** | Obsoleted by metric bug |
| `classical_baseline` | Feb 24 | RLSTCcode classical results (SGD, 1k–5k trajs, 5-fold CV) | Valid reference |
| `comparison` | Feb 24 | Direct head-to-head comparison report | Uses pre-fix Q-RLSTC numbers |
| **`thesis`** | **Feb 25** | **Definitive E1 run: 9 models × 5 seeds × 3 epochs × 50 trajectories, corrected basesim** | **Primary evidence** |

> **The `thesis/thesis_report_20260225_065747.md` is the canonical result.** All numbers in this document come from that run unless otherwise noted.

---

## 4. Models Tested (E1: Core Quantum Utility)

All 9 models share the same RL pipeline, replay buffer, epsilon schedule, and training data. Only the function approximator differs.

| Model | Params | Optimizer | Role |
|---|---|---|---|
| **VQ-DQN (5q×3L)** | **34** | **SPSA** | **Primary quantum model** |
| MLP-34 (SPSA) | 34 | SPSA | Iso-parameter classical control |
| MLP-34 (Adam) | 34 | Adam | Classical ceiling at same param count |
| Control A (linear) | 12 | SPSA | Minimal classical baseline |
| Control B (h=64) | 514 | SPSA | RLSTCcode-architecture classical control |
| Control C (h=32×32) | 1,314 | SPSA | Large classical control |
| Control D (Adam linear) | 12 | Adam | Adam classical baseline |
| Control E (Adam h=64) | 514 | Adam | Adam with RLSTCcode architecture |
| Control F (Adam h=32×32) | 1,314 | Adam | Adam large classical control |

---

## 5. Primary Results (Corrected E1, 5-Seed Summary)

### 5.1 Main Comparison Table

| Model | Params | ValCR (mean±std) | CUT% | #Segs | Wall Time | Q-margin |
|---|---|---|---|---|---|---|
| **VQ-DQN (5q×3L)** | **34** | **0.4126 ± 0.3039** | 6% ± 5% | 141 | 11,727s | +0.762 |
| MLP-34 (SPSA) | 34 | 0.7840 ± 0.2935 | 1% ± 3% | 36 | 45s | +0.540 |
| MLP-34 (Adam) | 34 | 0.4774 ± 0.1773 | 1% ± 1% | 22 | 57s | +0.090 |
| Control A (linear) | 12 | 0.4270 ± 0.3183 | 4% ± 6% | 107 | 48s | +0.507 |
| Control B (h=64) | 514 | 0.7185 ± 0.2567 | 0% ± 0% | 9 | 58s | +0.270 |
| Control C (h=32×32) | 1,314 | 0.5195 ± 0.1887 | 2% ± 3% | 50 | 69s | +0.148 |
| Control D (Adam linear) | 12 | 0.4125 ± 0.1612 | 1% ± 1% | 24 | 46s | +0.069 |
| Control E (Adam h=64) | 514 | 0.4696 ± 0.2872 | 3% ± 5% | 85 | 63s | +0.194 |
| Control F (Adam h=32×32) | 1,314 | 0.4777 ± 0.1590 | 1% ± 2% | 30 | 77s | +0.322 |

### 5.2 Key Observations

1. **VQ-DQN achieves the best mean ValCR (0.4126) among all SPSA-trained models**, beating:
   - MLP-34 SPSA (0.7840) — same parameter count, **47% worse**
   - Control B h=64 (0.7185) — 514 params (15× more), **74% worse**
   - Control C h=32×32 (0.5195) — 1,314 params (39× more), **26% worse**

2. **Adam-trained models close the gap**: Control D (Adam linear, 12 params) achieves 0.4125, essentially matching VQ-DQN. This tells us the quantum advantage is **optimizer-specific** — when you have access to gradient-based optimisation, classical networks do fine.

3. **High variance is real**: VQ-DQN's std of 0.3039 reflects genuine seed sensitivity. Some seeds (42, 2025) achieve excellent CR < 0.15, while seed 7 collapses into a never-cut policy (CR = 1.0). This is a known failure mode we document.

4. **Training time**: VQ-DQN is ~200× slower (11,727s vs ~50s) due to quantum circuit simulation. The value proposition is parameter footprint (34 params = 136 bytes), not speed.

---

## 6. Classical Baseline Context (RLSTCcode)

From the `classical_baseline` report, the original RLSTCcode system achieves:

| Configuration | Val CR | Training Data | Optimizer |
|---|---|---|---|
| Best single run (3k trajs) | **0.5892** | 3,000 trajectories | SGD |
| 5-fold CV mean | **0.7543 ± 0.0369** | ~4,000 trajectories | SGD |
| 1k trajectories | 0.7387 | 1,000 trajectories | SGD |
| 5k trajectories | 0.6710 | 5,000 trajectories | SGD |

**Important caveat**: RLSTCcode uses **SGD (backpropagation)** with **100× more data** (3,000 vs 50 trajectories). This is **not** a controlled comparison — it conflates optimizer, data scale, and model architecture.

### Cross-System Summary

| System | Best CR | Params | Data | Optimizer |
|---|---|---|---|---|
| RLSTCcode (best) | 0.5892 | 514 | 3,000 trajs | SGD |
| **Q-RLSTC VQ-DQN** | **0.4126** | **34** | **50 trajs** | **SPSA** |
| Q-RLSTC VQ-DQN (best seed) | ~0.13 | 34 | 50 trajs | SPSA |

The VQ-DQN's mean CR of 0.4126 is actually **better** than RLSTCcode's 0.5892, but the high variance (some seeds collapse) means this comparison must be interpreted carefully. The median is more robust.

---

## 7. Metric Degeneracy Discovery (D1 Diagnostic)

The D1 diagnostic revealed a fundamental problem with ValCR:

| Random CUT% | Actual CUT% | ValCR | #Segs |
|---|---|---|---|
| 0% | 0.0% | 9.2051 | 3 |
| 5% | 4.6% | 1.5032 | 67 |
| 50% | 32.3% | 1.4311 | 453 |
| 100% | 49.7% | 1.4193 | 696 |

**A random agent with no learning reduces ValCR from 9.2 to 1.4 just by cutting more.** This means ValCR is structurally biased toward over-segmentation. We mitigate this with:
- **Budget-constrained evaluation**: Compare models at matched CUT% thresholds
- **Reward shaping**: CUT_PENALTY (0.12) and L_MIN (3) in the reward function
- **Pareto analysis**: Plot each model's (CUT%, ValCR) against the random baseline curve

---

## 8. The Metric Bug and Its Fix

### What Went Wrong
`basesim` (the denominator of ValCR) was precomputed as a static scalar from the full dataset. It didn't change when the validation fold changed across seeds.

### What This Caused
- CR values > 1.0 everywhere in `thesis_multiseed` (7.17, 9.21, etc.)
- Quantized plateaus where many seeds reported identical CR
- Artificial variance masquerading as policy instability

### How We Fixed It
Compute `basesim` dynamically per validation fold by running an "always-extend" baseline policy on the fold's data. This restores ValCR as a genuine competitive ratio.

### Evidence the Fix Works
- `thesis_qfix` (single seed, corrected): VQ-DQN CR = 1.4811 (still above 1.0 because that run used older settings)
- `thesis` (5 seeds, corrected, 50 trajs, 3 epochs): VQ-DQN CR = **0.4126** — now below 1.0, meaning the agent genuinely improves clustering

---

## 9. What This Means for the Thesis

### Claims We Can Make
1. **C1 — Metric degeneracy**: ValCR has a structural fragmentation attractor. Budget-constrained evaluation is necessary for fair comparison. (Independent contribution, no quantum needed.)
2. **C2 — Framework**: Q-RLSTC is a modular hybrid quantum-classical RL framework with controlled substitution design.
3. **C3 — Regime-specific competitiveness**: Under SPSA + small data, VQ-DQN (34 params) is competitive with classical networks up to 39× larger. This is the core empirical contribution.
4. **C4 — NISQ feasibility**: The circuit design (5 qubits, ~15 depth) is within NISQ constraints.
5. **C5 — Metric autopsy**: The basesim bug discovery and fix demonstrates rigorous metric validation.

### Claims We Cannot Make
- ❌ "Quantum advantage" in a broad sense — Adam+classical networks match or beat VQ-DQN when gradient access is available
- ❌ Hardware demonstration — all experiments are simulation-based
- ❌ Generalisation beyond T-Drive GPS data

---

## 10. Open Questions for Discussion

1. **Seed sensitivity**: 2 of 5 seeds show never-cut collapse for VQ-DQN. Is this acceptable variance or a design problem?
2. **Adam parity**: The Adam-trained linear model (12 params!) matches VQ-DQN's CR. Does the SPSA-specific advantage carry enough weight?
3. **More epochs**: The current results use 3 epochs. Would more training time narrow the quantum-classical gap further?
4. **The comparison report** (`comparison/comparison_report.md`) uses pre-fix numbers. Should we regenerate it with corrected basesim?

---

## 11. File Reference

| File | Description |
|---|---|
| `results/thesis/thesis_report_20260225_065747.md` | **Primary E1 results** (corrected, 9 models, 5 seeds, 3 epochs) |
| `results/thesis/plots/` | Plots: ValCR comparison, CUT evolution, Pareto frontier |
| `results/classical_baseline/classical_report.md` | RLSTCcode baseline (SGD, 1k–5k trajs) |
| `results/comparison/comparison_report.md` | Head-to-head comparison (pre-fix numbers) |
| `results/thesis_nvalcr/` | D1 diagnostic with alternative metrics |
| `results/thesis_qfix/` | First corrected-metric run |
| `results/thesis_multiseed/` | Pre-fix multi-seed run (obsoleted by metric bug) |
| `results/thesis_pareto/` | Early Pareto + D1 baseline |
| `docs/thesis_outline/chapter4.md` | Experimental methodology chapter |
| `docs/thesis_outline/chapter5.md` | Conclusions and future work |
