# Chapter 1: Introduction

## 1.1 Motivation

The growth of spatiotemporal data from ride-sharing, logistics, and urban mobility systems has made trajectory analysis a foundational analytical operation. Significant structural similarity among moving objects often resides not in their complete paths, but within localised portions — motivating sub-trajectory clustering, which partitions trajectories into segments and groups those segments into spatially coherent clusters.

The central computational difficulty is a bilevel optimisation problem: optimal segmentation depends on cluster structure, yet cluster structure emerges only after segmentation. Classical approaches decouple these phases using rigid heuristics — most prominently TRACLUS, which uses the Minimum Description Length (MDL) principle for segmentation independent of clustering objectives. While mathematically elegant, MDL-based segmentation is static, deterministic, and domain-blind.

The RLSTC framework (Lee et al., VLDB 2024) replaced heuristic segmentation with a learned policy: a Deep Q-Network decides EXTEND or CUT at each trajectory point, guided by reward derived from downstream clustering quality. This consistently outperforms TRACLUS and related baselines. However, RLSTC's classical policy networks contain hundreds to thousands of trainable parameters — trivial for centralised deployments, but a constraint for bandwidth-limited federated edge learning, where model updates must traverse constrained wireless links.

This motivates the investigation of ultra-compact policy approximators that can match classical segmentation quality with radically fewer parameters.

## 1.2 Research Question

> Under a fixed small-data budget and gradient-free optimisation (SPSA), does a 5-qubit VQC policy achieve a better ValCR–CUT% Pareto frontier (lower ValCR at matched CUT budgets) than parameter-matched and larger classical MLP policies, when used as the sole substituted component in an RLSTC DQN agent?

This formulation:
- **Isolates the policy network** as the single experimental variable
- **Enables attribution**: differences arise from the function approximator, not confounded system changes
- **Is falsifiable against the Pareto reporting standard**, avoiding the ValCR degeneracy pitfall (Chapter 4)

## 1.3 Scope and Positioning

This work originated from a broader survey (Q-RLSTC Short Paper) identifying five candidate quantum subcomponents for the RLSTC pipeline. Feasibility analysis narrowed the scope to a single component: the policy network. Quantum clustering initialisation and distance estimation require circuit depths beyond current NISQ hardware; replacing multiple components simultaneously would entangle experimental variables, preventing attribution. This narrowing reflects standard scientific practice — broad vision → principled narrowing → rigorous test.

## 1.4 Contributions

### C1. Identification of a Structural Degeneracy in Trajectory Clustering Evaluation

We show that ValCR, the standard RLSTC evaluation metric, is structurally biased toward fragmentation: per-segment averaging combined with IED's length dependence creates a trivial high-CUT attractor. D1 diagnostics confirm this empirically — a random policy sweep reduces ValCR from 9.21 to ~1.42 based on segmentation rate alone, without learning. We propose budget-constrained evaluation (reporting ValCR at matched CUT thresholds) as a general mitigation applicable to all RL-based trajectory segmentation methods. This contribution is independent of the quantum component.

### C2. End-to-End Hybrid Quantum-Classical RL Framework

We design Q-RLSTC: a modular system replacing the classical DQN policy with a 34-parameter VQ-DQN (5-qubit, 3-layer HEA), while retaining identical classical components for distance computation, replay buffering, and cluster maintenance. The controlled substitution framework enables fair comparison.

### C3. Controlled Empirical Evidence of Regime-Specific VQC Competitiveness

Under matched training conditions, the 34-parameter VQ-DQN is evaluated against 8 classical RL controls — 4 SPSA-trained (parameter-matched, linear, medium, deep) and 4 Adam-trained (same architectures) — plus 2 non-RL baselines (uniform window, random policy). Causal attribution claims are restricted to the SPSA-matched cohort; Adam baselines contextualise performance ceilings. Multi-seed validation (5 seeds) with Mann-Whitney U tests, Cohen's d, and bootstrap 95% CI provides statistically grounded comparison.

### C4. NISQ-Feasible Circuit Design with Simulation-Based Robustness Evaluation

The VQ-DQN circuit operates within strict NISQ constraints: angle encoding (1 qubit per feature), shallow depth (~15), no mid-circuit measurement, and constant-cost SPSA optimisation (2 evaluations per gradient estimate). Robustness is characterised via shot sensitivity sweeps and hardware-inspired noise model simulations (IBM Eagle, Heron profiles). These experiments characterise expected degradation; they do not constitute a hardware demonstration.

## 1.5 Why Would a VQC Be Parameter-Efficient?

A VQC induces a bounded, highly structured hypothesis class: outputs are expectation values of low-depth unitaries composed with a linear readout, inherently constrained to $[-1, +1]$. In small-data and gradient-free regimes, this structure can behave like an implicit regulariser, potentially improving stability relative to overparameterised MLPs trained with noisy SPSA gradient estimates. The bounded output range provides natural value regularisation without explicit clamping (though clamping is still applied for safety). Our ablations — entanglement removal (AB1), shot sweeps (E3), noise sweeps (E2) — probe whether the observed parameter efficiency correlates with uniquely quantum resources (entanglement) or with generic architectural constraints (bounded outputs, structured parameterisation).

## 1.6 What This Work Does Not Claim

1. **No speedup.** VQ-DQN training is ~100× slower than classical controls due to circuit simulation. The value proposition is parameter footprint, not wall-clock time.
2. **No broad classical superiority.** Under Adam with sufficient data, classical networks achieve strong results. Competitiveness is specific to the SPSA + small-data regime.
3. **No generalisation beyond tested conditions.** The regime-specific competitiveness may reflect favourable SPSA landscape properties rather than quantum-mechanical effects per se.
4. **No theoretical claims.** Parameter efficiency here is an empirical observation (fewer params, competitive Pareto performance under matched training), not a statement about sample complexity or function query complexity.

## 1.7 Thesis Outline

- **Chapter 2**: Background on trajectory segmentation, RL, variational quantum computing, and the Q-RLSTC method (circuit design, 10 models, reward engineering, training pipeline).
- **Chapter 3**: Experimental setup — hardware, dataset, metrics, the full experiment matrix (D1–E6, AB1, A1, S1), multi-seed protocol, and reproducibility.
- **Chapter 4**: ValCR length sensitivity analysis and budget-constrained evaluation; core benchmark results (E1 Pareto analysis); Q-learning dynamics; NISQ and shot robustness; ablations.
- **Chapter 5**: Conclusions (C1–C4), limitations, and future work.