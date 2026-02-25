# Chapter 4: ValCR Length Sensitivity and Experimental Results

## 4.1 The Length-Sensitivity Problem

Before presenting benchmark results, we characterise a fundamental evaluation pathology affecting all RLSTC-family methods.

ValCR is the mean of $\min_c \text{IED}(s, c)$ across all produced segments. This formulation creates a systematic bias toward fragmentation through three interacting mechanisms:

1. **Per-segment averaging.** ValCR treats a 3-point segment identically to a 400-point segment. Cutting increases segment count and shifts mass toward shorter ones.
2. **Length dependence of IED.** Under per-segment averaging and typical sequence-distance behaviour, expected distance to the nearest cluster centre decreases with segment length — fewer misalignment opportunities combined with min-over-centres selection.
3. **Fragmentation attractor.** Reducing segment length reduces ValCR even without better "semantic" clustering. Segmentation rate alone, not placement quality, drives the metric.

Under per-segment averaging and typical sequence-distance behaviour, fragmentation systematically reduces expected segment-to-centre distance, creating a trivial high-CUT attractor. D1 empirically confirms this effect in our setting.

### Degeneracy Under Naive Reward

Under a naive reward that minimises OD without penalising segment count or enforcing minimum segment length:

1. Shrinking segments reduces the IED contribution per segment (shorter sequences have lower expected distance to their nearest centre).
2. An always-cut policy produces single-point segments, driving OD to an artificially low value regardless of trajectory structure.
3. This makes ValCR structurally degenerate: the unconstrained optimum is a trivial policy that performs no meaningful segmentation.

**Assumptions:** This argument requires (a) nonnegative distance, (b) expected distance non-increasing with segment length under nearest-centre selection, and (c) no segment-count penalty in the reward. Condition (b) is empirically verified for IED in our setting via the D1 sweep; it is not guaranteed for all distance metrics.

---

## 4.2 Denominator Pathology: basesim Sensitivity

ValCR = $\frac{OD}{\text{basesim}}$ introduces a second pathology beyond length sensitivity: **denominator instability**. If validation trajectories happen to be very close to their cluster centres (basesim ≈ 0), then CR explodes regardless of policy quality. This is distinct from the fragmentation attractor — it affects the evaluation metric's denominator rather than its numerator.

### Diagnostic Autopsy: The Static Denominator Bug
We initially observed quantized ValCR plateaus across seeds. Investigation showed the `basesim` denominator was loaded as a static scalar from the dataset's precomputed centres and did not depend on the specific validation fold assigned to the seed. Consequently, "CR" behaved like a globally-scaled OD rather than a fold-relative ratio, which inflated variance and complicated interpretation. We corrected this by computing a fold-specific `basesim` via an "always-extend" baseline policy executed strictly on the validation fold.

**Mitigation:**
- **Dynamic Fold-Specific BaseSim:** Dividing by the organically computed base distance of the active evaluation fold restores ValCR as a fully comparable relative ratio.
- **ε-stabilisation:** CR = OD / max(basesim, 10⁻⁸) prevents division by zero.
- **OD/basesim decomposition:** we log OD and basesim separately so anomalous denominators remain visible.
- **Median CR:** robust to single outlier trajectories with near-zero basesim.

---

## 4.3 D1 Diagnostic: ValCR vs CUT% Sweep

D1 runs a random policy at CUT probabilities {0%, 5%, 10%, 20%, 30%, 50%, 80%, 100%}:

- **CUT = 0%:** ValCR = 9.21 (entire trajectory is one segment)
- **CUT ≈ 5%:** ValCR drops to ~1.50
- **CUT ≈ 43–50%:** ValCR ≈ 1.42 (plateau, average segment length ≈ 3 points)

**No learning is required for this improvement — segmentation rate alone drives the metric.**

D1 also evaluates nValCR and wValCR to test whether length-aware normalisation restores a meaningful interior optimum. The empirical results inform whether metric redesign or budgeted evaluation is the appropriate mitigation.

---

## 4.4 Budget-Constrained Evaluation

Because no single normalisation fully corrects the length coupling, we adopt **budget-constrained evaluation** as the primary reporting standard:

- Report best ValCR at matched CUT thresholds: CUT ≤ {5%, 10%, 20%, 30%, 40%}
- Overlay learned agents on the D1 random baseline Pareto curve (the "money figure" — see §4.4)

**Why this works.** Budgeted evaluation is constraint-agnostic: comparisons at matched CUT% are fair regardless of IED's internal scaling properties. An agent that achieves lower ValCR than a random policy *at the same CUT budget* has learned meaningful segment placement, not just fragmentation.

**Implication for reward design.** The over-segmentation attractor arises because the reward ($\Delta OD$) inherits ValCR's length sensitivity. Q-margin analysis (D2) confirms learned agents develop negative Q-margins ($Q(\text{cut}) > Q(\text{extend})$), consistent with drift toward the high-CUT plateau. Future work should explore Lagrangian formulations with explicit CUT budget in the optimisation objective.

---

## 4.5 Failure Mode Taxonomy

Multi-seed evaluation reveals two distinct failure modes that must be reported, not hidden:

**Failure Mode A: Never-Cut Policy (Real Collapse)**
- Symptoms: Val CUT% ≈ 0%, very few segments, high OD
- Cause: Agent stuck in EXTEND-only basin (insufficient exploration or reward signal)
- Diagnosis: Q-margin is strongly negative from the start; agent never learns to differentiate actions

**Failure Mode B: CR Blowup (Denominator Pathology)**
- Symptoms: ValCR extremely high, but OD not proportionally high; basesim ≈ 0
- Cause: Validation trajectory split happens to include trajectories very close to cluster centres
- Diagnosis: OD/basesim decomposition reveals small denominator; median CR is substantially lower than mean CR

Across seeds, VQ-DQN typically improves rapidly by epoch 2, but exhibits occasional failure modes. We therefore report robust statistics (median CR) and analyze collapse cases explicitly rather than averaging over them.

---

## 4.6 Core Quantum Utility: E1 Results

E1 is the primary thesis experiment. All 9 RL models train under identical conditions (same trajectories, seeds, hyperparameters) with only the function approximator varying.

### The Money Figure: ValCR–CUT% Pareto Frontier

The central result is a scatter plot of each agent's (CUT%, ValCR) point overlaid on the D1 random baseline curve. An agent positioned **below** the D1 curve at any given CUT budget has learned segment placement beyond what random cutting achieves — this is the operational definition of "the policy has learned something useful."

### Primary Result Table

Best ValCR under CUT budget constraints (5-seed mean ± std; median CR in parentheses):

| CUT ≤ | Random (D1) | VQ-DQN (34p) | MLP-34 SPSA | Control B (514p) | Control C (1,314p) | Control F (Adam 1,314p) |
|---|---|---|---|---|---|---|
| ≤ 5% | ... | ... (med: ...) | ... | ... | ... | ... |
| ≤ 10% | ... | ... (med: ...) | ... | ... | ... | ... |
| ≤ 20% | ... | ... (med: ...) | ... | ... | ... | ... |
| ≤ 30% | ... | ... (med: ...) | ... | ... | ... | ... |
| ≤ 40% | ... | ... (med: ...) | ... | ... | ... | ... |

*(Values populated from multi-seed E1 run.)*

**OD/basesim decomposition table** (per seed, best epoch):

| Seed | Model | OD | basesim | CR (mean) | CR (median) | CUT% | #segs |
|---|---|---|---|---|---|---|---|
| 42 | VQ-DQN | ... | ... | ... | ... | ... | ... |
| 123 | VQ-DQN | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... | ... |

*(Seeds where basesim < 0.1 flagged as potential denominator pathology.)*

### Significance Testing

For each CUT budget threshold, paired VQ-DQN vs control comparisons report:
- Mann-Whitney U (nonparametric, 5 seeds per condition)
- Cohen's d with interpretation label
- Bootstrap 95% CI on mean ValCR difference

---

## 4.7 Q-Learning Dynamics

### Q-Value Stabilisation

Without Q-value clamping, values diverge to ~78M within 2 epochs. After ±10 clamping, values remain bounded and informative. This confirms clamping is load-bearing (E6 ablation).

### Q-Margin Analysis (D2)

Q-margin ($Q_{\text{extend}} - Q_{\text{cut}}$) tracked per epoch reveals policy bias formation. Comparison between VQ-DQN and classical controls shows whether the quantum circuit develops different preference dynamics.

### Replay Buffer Drift (D5)

Buffer CUT% diverges from on-policy CUT% by ~7–10% as training progresses, a standard RL phenomenon documented for completeness.

---

## 4.8 NISQ Viability and Shot Sensitivity

### E2: Hardware-Inspired Noise Models

VQ-DQN is evaluated under Eagle (heavy-hex, higher error rates) and Heron (square lattice, lower error rates) noise profiles from Qiskit Aer. These characterise expected degradation under hardware-motivated constraints, not physical hardware performance.

### E3: Shot Sensitivity

Shots ∈ {128, 512, 2,048} vs statevector (exact). Key questions:
- At what shot count does Q-margin become unreliable?
- Is degradation graceful or discontinuous?

### AB1: Entanglement Ablation

No-CNOT vs linear-CNOT circuit comparison. If entanglement contributes to policy quality, the no-CNOT circuit underperforms, providing mechanism evidence. If performance is equivalent, the efficiency may arise from generic architectural constraints (bounded outputs, structured parameterisation) rather than quantum correlations.

---

## 4.9 Stabilisation Ablation (E6)

Incremental addition of each reward component quantifies contribution:

| Configuration | Key Observation |
|---|---|
| Raw $\Delta OD$ only | Always-cut degeneracy emerges |
| + L_MIN = 3 | Micro-segments eliminated |
| + CUT_PENALTY | Over-segmentation reduced |
| + Q-clamping | Value explosion prevented |
| Full system | Stable training achieved |

This distinguishes load-bearing components from cosmetic ones.

---

## 4.10 Limitations

- **Training speed.** VQ-DQN ~100× slower than classical controls due to circuit simulation.
- **Metric.** ValCR, even budget-constrained, may not perfectly align with subjective clustering quality.
- **Simulation only.** No physical quantum hardware validation.
- **Single dataset.** T-Drive taxi GPS. Generalisation to diverse domains untested.
- **Small-data regime.** Results under larger data + gradient-based optimisation may differ.