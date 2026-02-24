# Thesis Outline: Quantum-Enhanced Reinforcement Learning for Sub-Trajectory Clustering

[← Back to README](../../README.md) · [Architecture](architecture.md) · **Thesis Outline** · [Experimental Design →](experimental_design.md)

---

> **Working Title**: *Parameter-Efficient Variational Quantum Policies for Sub-Trajectory Segmentation Under Budgeted Evaluation*

---

## 1. Introduction

### 1.1 Motivation
- Growth of spatiotemporal data (ride-sharing, logistics, urban mobility)
- Sub-trajectory clustering as a foundational operation for pattern mining, anomaly detection, and route optimization
- Classical RL-based segmentation (RLSTC) achieves adaptive, learned cut policies but scales in parameter count
- Opportunity: variational quantum circuits as ultra-compact policy approximators under NISQ constraints

### 1.2 Problem Statement
- Given a stream of trajectory points, learn a binary segmentation policy (EXTEND/CUT) that minimizes clustering distance while respecting explicit segmentation budget constraints (CUT ≤ α, AvgLen ≥ L)
- Core question: can a variational quantum policy with O(30) parameters match classical networks with O(1000) parameters on segmentation quality at comparable segmentation budgets?

### 1.3 Contributions
1. **VQ-DQN Architecture**: A 5-qubit, 3-layer variational quantum circuit with learnable affine head for trajectory segmentation, optimized via SPSA — achieving competitive quality with 38× fewer parameters than the best classical control
2. **ValCR Metric Diagnosis and Budgeted Evaluation Protocol**: We show that raw ValCR (mean segment-to-center IED) is length-sensitive under IED and can be trivially optimized by fragmentation; we propose budgeted evaluation (best ValCR at CUT ≤ α) as a standard protocol for trajectory clustering metrics, restoring meaningful comparisons
3. **Diagnostic Suite**: D1–D5 experiments exposing Q-value dynamics, policy basin formation, replay buffer drift, action distribution evolution, and the over-segmentation attractor — providing a reusable template for RL stability analysis
4. **NISQ Sensitivity Analysis**: Shot count and noise model impact on policy stability, quantifying when finite sampling degrades learned policies

### 1.4 Outline
- Overview of remaining chapters

---

## 2. Background and Related Work

### 2.1 Trajectory Data and Segmentation
- Trajectory representation (spatiotemporal point sequences)
- Sub-trajectory clustering objectives (compactness, separation)
- Classical segmentation approaches: sliding window, MDL-based, heuristic (speed/heading change), fixed-window
- RLSTC: reinforcement learning for sub-trajectory clustering (Lee et al.)

### 2.2 Reinforcement Learning for Sequential Decisions
- MDP formulation for segmentation
- Deep Q-Networks (DQN): experience replay, target networks, ε-greedy exploration
- SPSA (Simultaneous Perturbation Stochastic Approximation) as a gradient-free optimizer
- Why SPSA over backprop: gradient-free = compatible with quantum expectation values; identical optimizer across quantum/classical isolates the function approximator as the only experimental variable

### 2.3 Variational Quantum Computing
- Parameterized quantum circuits (PQCs) and variational algorithms
- Angle encoding for classical-to-quantum data embedding
- Hardware-efficient ansätze (HEA) for NISQ devices
- Expressibility, entanglement, and barren plateau considerations

### 2.4 Quantum Reinforcement Learning
- Variational Quantum Deep Q-Networks (VQ-DQN)
- Hybrid quantum-classical architectures
- Prior work on quantum RL for combinatorial tasks
- Open questions: parameter efficiency vs. training cost tradeoff

### 2.5 Distance Metrics for Trajectory Similarity
- Integrated Edit Distance (IED): definition, properties, length dependence
- Why IED matters: it is non-decreasing in segment length in expectation, and the min-over-centers aggregation amplifies this
- Alternative metrics: DTW, Fréchet, Hausdorff

---

## 3. Proposed Method

### 3.1 System Architecture
- Pipeline overview: observation → angle encoding → VQC → Q-values → ε-greedy action → environment step
- MDP specification: state (5D), actions {EXTEND, CUT}, termination conditions
- **Environment semantics**: on CUT, the completed segment is immediately assigned to its nearest cluster center (global, over all K clusters) and the center set is incrementally updated during the episode (online style). This means agent decisions affect subsequent distance computations within the same episode.

### 3.2 VQ-DQN Circuit Design
- 5-qubit circuit with RY angle encoding
- 3-layer hardware-efficient ansatz (CNOT entanglement, parameterized rotations)
- Readout: Pauli-Z expectation values → learnable affine head (scale, bias)
- Total parameters: 30 (circuit) + 4 (affine) = 34

### 3.3 Classical Controls and Baselines

**RL Baselines (identical training pipeline, SPSA optimizer):**

| Control | Architecture | Params | Purpose |
|---|---|---|---|
| A: Linear | 5→2 | 12 | Capacity lower bound |
| B: Medium MLP | 5→64→2 | 514 | Moderate capacity |
| C: Deep MLP | 5→32→32→2 | 1,314 | Classical ceiling |

**Non-RL Baselines (no training, deterministic):**

| Baseline | Method | Purpose |
|---|---|---|
| Uniform Window | Cut every W points (W chosen to hit target CUT%) | Matches segmentation budget without learning |
| Random Policy | Cut with probability p (from D1 sweep) | Lower bound — is learning needed at all? |

### 3.4 Training Pipeline
- SPSA optimizer with gradient clipping (max_grad_norm=10)
- Double DQN with target network soft updates
- Experience replay buffer (size 5,000)
- ε-greedy exploration with decay schedule

### 3.5 Reward Design and Stabilization

The reward is constructed incrementally. Each component is justified by an ablation:

| Component | Formula | Introduced to address | Ablated in |
|---|---|---|---|
| OD improvement (base) | `scale × (old_OD − new_OD)` | Primary learning signal | E6 (baseline) |
| CUT penalty | `−λ_cut` per cut action | Over-segmentation | E6 |
| EXTEND cost | `−λ_ext` per extend action | Idle extending | E6 |
| L_MIN = 3 | Hard action override | Micro-segments | E6 |
| Q-value clamping (±10) | Output clipping | Value explosion (observed: Q → 78M) | E6 |
| TD target clamping (±10) | Target clipping | Bootstrapping instability | E6 |

We introduce stabilization terms incrementally and quantify their effect in ablations (E6). Without clamping, Q-values diverge to ~78M within 2 epochs; without L_MIN, the agent produces 1-point segments.

### 3.6 Evaluation Framework

**Primary metric:**
- **ValCR** (raw): mean segment-to-center IED / base similarity

**Diagnostic variants (for pathology analysis):**
- **nValCR** (per-point): mean of (IED / segment_length) / base similarity
- **wValCR** (length-weighted): total_IED / total_points / base similarity

**Default reporting format:**
- Pareto-constrained table: best ValCR at CUT ≤ {5%, 10%, 20%, 30%, 40%}
- Pareto frontier plot: agent performance (ValCR, CUT%) overlaid on D1 random baseline curve

**Why budgeted evaluation?** Raw ValCR comparisons across agents with different CUT rates are meaningless because ValCR is strongly CUT-dependent (Chapter 4). Budgeted reporting controls for this confound.

---

## 4. ValCR Length Sensitivity and Budgeted Evaluation

### 4.1 The Length-Sensitivity Problem

ValCR is the mean of min_c IED(s, c) across all produced segments. This formulation has a structural incentive toward fragmentation:

1. ValCR averages _per segment_, not per point — a 3-point segment contributes equally to a 400-point segment
2. Cutting increases the number of segments and shifts mass toward shorter ones
3. IED has non-decreasing length dependence: for trajectory segments and representative centers, expected distance to the nearest center is lower for shorter sequences (fewer misalignment opportunities + min-over-centers cherry-picking)
4. Therefore, reducing segment length reduces ValCR even without better "semantic" clustering

**D1 confirmation:** A random policy sweep across CUT probabilities {0%–100%} shows ValCR drops sharply when CUT moves off 0% (9.21 → 1.50), then approaches a flat plateau in the short-segment regime (realized CUT ≈ 43–50%, AvgLen ≈ 3 points, ValCR ≈ 1.42). No learning is required for this improvement — segmentation rate alone drives the metric.

### 4.2 Normalized Variants as Diagnostic Tools

We introduce two length-aware variants to test whether the ValCR plateau is a length artifact:
- **nValCR**: mean of (IED / segment_length), removing per-segment length dependence
- **wValCR**: total_IED / total_points, weighting by segment length

D1 evaluation of these variants tests whether they produce a meaningful interior optimum (a CUT rate that balances quality and budget) or whether they overcorrect. This empirical result informs whether metric redesign or budgeted evaluation is the appropriate mitigation.

### 4.3 Budgeted Evaluation Protocol

Because no single normalization may fully correct the length-coupling, we propose **budgeted evaluation** as the primary reporting standard:

- Report best ValCR at matched CUT thresholds: CUT ≤ {5%, 10%, 20%, 30%, 40%}
- Overlay learned agents on the D1 random baseline Pareto curve
- This eliminates degenerate "wins" where an agent achieves lower ValCR simply by cutting more

**Defense of this choice:** Budgeted evaluation is constraint-agnostic — it works regardless of the metric's internal properties. Whether IED scaling is linear, sublinear, or superlinear, comparisons at matched CUT% are always fair.

### 4.4 Implications for Reward Design
- The over-segmentation attractor arises because the reward (Δoverdist) inherits ValCR's length sensitivity
- Q-margin analysis (D2) confirms learned agents develop negative Q-margins (Q(cut) > Q(extend)), consistent with drifting toward the high-CUT plateau
- Future work: Lagrangian formulation with explicit CUT budget in the optimization objective

---

## 5. Experimental Evaluation

### 5.1 Setup
- **Datasets**: T-Drive (Beijing taxi GPS), GeoLife (multi-mode transportation)
- **Hardware**: Local simulation via Qiskit Aer (statevector + shot-based + noise models)
- **Seeds**: 5 seeds per condition, reporting mean ± std with 95% CI
- **Hyperparameters**: PROTOCOL table (γ, ε schedule, penalties, buffer size)

### 5.2 Diagnostic Experiments (D1–D5)

#### D1: ValCR vs CUT% Sweep
- Random policy at CUT probabilities {0%, 5%, 10%, 20%, 30%, 50%, 80%, 100%}
- Reports raw ValCR, nValCR, wValCR, #segs, avg segment length
- **Key finding**: ValCR drops sharply from 9.21 (CUT=0%) to 1.50 (CUT≈5%), then plateaus around 1.42 at realized CUT ≈ 43–50% and AvgLen ≈ 3 points

#### D2: Q-Margin Evolution
- Q(extend) − Q(cut) tracked per epoch for all models
- **Key finding**: VQ-DQN develops negative Q-margin (prefers cutting); classical controls develop positive (prefer extending)

#### D3: Training Action Distribution
- Per-epoch CUT% in training actions
- Shows policy drift and action bias formation

#### D4: Policy Basin Test
- Forced all-cut / all-extend / alternating policies under drift mode
- Tests whether basin structure exists independent of learning

#### D5: Replay Buffer Histogram
- Buffer CUT% vs on-policy CUT% across epochs
- Quantifies replay distribution drift

### 5.3 Core Benchmarks (E1–E6)

#### E1: Core Quantum Utility (default Pareto format)

**Primary result table:** best ValCR under CUT budget constraints:

| CUT ≤ | Random (D1) | VQ-DQN (34p) | Control A (12p) | Control B (514p) | Control C (1,314p) | Heuristic |
|---|---|---|---|---|---|---|
| ≤ 5% | ... | ... | ... | ... | ... | ... |
| ≤ 10% | ... | ... | ... | ... | ... | ... |
| ≤ 20% | ... | ... | ... | ... | ... | ... |

Plus: parameter efficiency ratio, wall-clock comparison

**Key question answered:** At comparable segmentation budgets, does VQ-DQN match classical capacity with 38× fewer parameters?

#### E2: NISQ Viability
- Eagle and Heron noise models
- Impact of decoherence on policy quality

#### E3: Shot Sensitivity
- Shots ∈ {128, 512, 2048} vs noiseless (shots=0)
- Graceful degradation analysis
- **Key question:** What breaks first — policy preference, Q-margin stability, or ValCR?

#### E4: Drift Resilience
- **Operational definition**: Train on trajectories from temporal window A (e.g., first 70%), evaluate on temporally disjoint window B (last 30%). Distribution shift arises from time-of-day, route, and traffic pattern differences.
- Tests whether learned policies transfer across temporal distribution shifts

#### E5: Low-Data Generalization
- 10%, 25%, 50% data fractions
- Sample efficiency comparison

#### E6: Stabilization Ablation
- Incremental addition of each stability term (reward components, clamping, L_MIN)
- Baseline: raw Δoverdist reward only
- Shows which terms are load-bearing vs. cosmetic

### 5.4 Scalability (S1)
- Wall-clock timing at 250–1000 trajectories
- Quantum simulation overhead vs classical inference speed
- Honest accounting: VQ-DQN training is ~100× slower than classical controls due to circuit simulation; the value proposition is parameter footprint, not wall-clock

### 5.5 Multi-Seed Aggregation
- 5-seed results for E1 with mean ± std and 95% CI
- Paired comparisons across identical seeds/trajectory subsets for VQ-DQN vs each control

---

## 6. Results and Analysis

### 6.1 Parameter Efficiency Under Budget Constraints
- **Unconstrained:** VQ-DQN achieves lower raw ValCR than all classical controls but at substantially higher CUT rates (e.g., 40% vs 2% for Control C)
- **Budgeted comparison (main result):** at matched CUT ≤ α, VQ-DQN with 34 parameters achieves [comparable / better / worse] ValCR relative to Control C (1,314 params)
- Controls A/B lack representation capacity (extend-lock at CUT=0%)
- Table: model × params × best-ValCR-at-CUT≤10% × best-ValCR-at-CUT≤30%

### 6.2 ValCR Length Sensitivity (D1 Results)
- D1 sweep table with 3 ValCR variants
- Pareto frontier plot: agents overlaid on random baseline
- Empirical IED-vs-length relationship in the T-Drive/GeoLife data
- "Segment-weighting is the root issue: ValCR treats a 3-point segment as equally important as a 400-point segment"

### 6.3 Q-Learning Dynamics
- Q-value evolution (pre/post clamping): values diverged to ~78M, bounded to [−10, +10] after clamping
- Q-margin trends: VQ-DQN learns to differentiate actions; classical controls A/B collapse
- Replay buffer drift: BufCUT diverges from on-policy CUT by ~7–10%
- Over-segmentation attractor: negative Q-margin + high CUT% consistent with D1 plateau

### 6.4 NISQ and Shot Sensitivity
- Shot noise impact on ValCR and Q-margin stability
- Noise model degradation curves
- Practical circuit depth limits

### 6.5 Heuristic Baseline Comparison
- Do heading-change / fixed-window heuristics match the D1 random curve?
- If yes: evidence that ValCR is largely a segmentation-rate proxy
- If no: evidence that segment placement matters, strengthening the case for learned policies

### 6.6 Limitations
- Training wall-clock: VQ-DQN ~100× slower than classical due to circuit simulation; value proposition is parameter footprint, not training speed
- ValCR metric limitations (not quality-aligned without budget constraints)
- Ideal simulator results; no real quantum hardware validation
- Single dataset family (taxi GPS); generalization to diverse trajectory domains not yet demonstrated
- Backprop baselines not included (SPSA used for all agents to isolate approximator); future work should compare

---

## 7. Conclusions and Future Work

### 7.1 Summary of Contributions
- Compact VQ-DQN policy achieves competitive segmentation with extreme parameter compression under explicit CUT budget constraints
- ValCR length sensitivity identified, diagnosed with formal argument and D1 empirics, and mitigated via budgeted evaluation protocol — a community-relevant contribution independent of quantum
- Diagnostic suite (D1–D5) provides reusable template for RL stability analysis in sequential decision problems

### 7.2 Future Work
- **Constrained RL**: Lagrangian formulation with explicit CUT budget in optimization objective (not just evaluation)
- **Metric redesign**: length-weighted or per-point IED variants as primary metrics; downstream task evaluation (ARI/NMI, retrieval accuracy)
- **Standard backprop baselines**: classical DQN with Adam optimizer to quantify SPSA overhead
- **Real quantum hardware**: IBM Quantum execution with error mitigation
- **Multi-dataset validation**: T-Drive, GeoLife, Porto taxi, Foursquare check-ins
- **Stronger learned baselines**: Transformer-based policies, constrained RL methods (CPO, Lagrangian)
- **Communication/deployment story**: quantum policy as compact model for edge-device broadcasting (parameter footprint advantage)

---

## Appendices

### A. Hyperparameter Table (PROTOCOL)
- Full listing of all training hyperparameters and their justifications

### B. Quantum Circuit Diagrams
- VQC architecture visualization (5-qubit, 3-layer HEA)

### C. D1 Sweep Full Results
- Complete table with all 3 ValCR variants across CUT probabilities
- IED-vs-length empirical scaling plot

### D. Defense FAQ
1. **What does ValCR measure, and why should we care?** → ValCR is mean segment-to-nearest-center distance; it is the standard RLSTC evaluation metric. We demonstrate its limitations and propose mitigations.
2. **Why does cutting change the metric so much? Is it "cheating"?** → IED is length-dependent; per-segment averaging creates a structural incentive to fragment. This is a property of the evaluation protocol, not the agent. Chapter 4 formalizes this.
3. **If random cuts can get good ValCR, why do we need RL at all?** → At matched CUT budgets, learned agents [do/do not] outperform random — this is exactly what the Pareto analysis quantifies.
4. **Where is the "quantum" benefit?** → Parameter compression (34 vs 1,314), not training speed or asymptotic accuracy. The value proposition is compact policy broadcasting.
5. **Why SPSA? What happens with standard backprop?** → SPSA enables identical optimization across quantum and classical; future work should include backprop baselines.
6. **Does it survive finite shots / noise?** → E3 and E2 quantify this directly.

### E. Reproducibility
- Environment specification (Python, NumPy, Qiskit versions)
- Git commit hash for exact reproducibility
- Data preprocessing pipeline
- `--seeds` flag documentation

---

## References

*(RLSTC, VQ-DQN, SPSA, IED, T-Drive, GeoLife, quantum RL literature)*
