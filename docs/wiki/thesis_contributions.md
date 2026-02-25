# Thesis Contributions

## Positioning Statement

> The goal is not to demonstrate a quantum advantage, but to map a regime where a NISQ-feasible VQC policy is a competitive function approximator under constraints (small data, gradient-free optimization) that commonly arise in near-term quantum workflows.

**Research Question:**

> Under a fixed small-data budget and gradient-free optimization (SPSA), does a shallow 5-qubit VQC policy achieve lower validation Competitive Ratio than parameter-matched and larger classical MLP policies when used as the sole substituted component in an RLSTC DQN agent?

---

## Key Definition: Parameter Efficiency

**Parameter efficiency** (in this thesis) means: under a fixed optimizer (SPSA), fixed training budget (episodes × steps), and fixed data regime, the model with fewer trainable parameters achieves lower ValCR. This is a controlled empirical definition, not a universal statement about sample complexity or function query complexity.

---

## Contributions

### C1. End-to-End Hybrid Quantum-Classical RL Framework for Trajectory Segmentation

We design and implement Q-RLSTC: a complete system that replaces the classical DQN policy network with a 5-qubit, 3-layer hardware-efficient ansatz (34 trainable parameters), while retaining classical components for environment dynamics, distance estimation, replay buffering, and cluster maintenance. The framework is modular, allowing controlled substitution of the policy component for fair comparison.

### C2. Controlled Empirical Evidence of Regime-Specific VQC Competitiveness

Under matched training conditions — same optimizer (SPSA), same 30-trajectory dataset, same 2-epoch budget, same replay buffer, same reward function, same evaluation protocol — the 34-parameter VQ-DQN attains the lowest ValCR among the tested SPSA-trained function approximators, including parameter-matched and larger MLP controls. This is consistent with (but does not by itself prove) a favorable inductive bias of the chosen VQC ansatz in the small-data, gradient-free regime.

**Evidence status (update based on available data):**
- *If single-seed only*: This is **pilot evidence** from a single seed. Multi-seed validation is required before this claim can be considered established.
- *If multi-seed available*: Across N seeds, VQ-DQN achieves mean ValCR X ± Y versus the strongest classical SPSA baseline at A ± B (p = ..., bootstrap 95% CI = ...).

**Compute-budget parity**: All models receive identical numbers of environment steps, SPSA iterations, and forward evaluations (SPSA = 2 evaluations per gradient estimate; VQC vs MLP forward-pass cost is not relevant since we claim no speedup, only quality).

### C3. Reward Engineering with Anti-Gaming Constraints (Degeneracy Mitigation)

We identify and formally characterize a structural degeneracy in the competitive ratio metric.

**Lemma (CR Degeneracy under Naive Reward):**
Define CR = OD / basesim, where OD = (1/N) Σᵢ IED(sᵢ, centerᵢ) averages IED across N assigned segments. Under a naive reward that minimizes OD without penalizing segment count or enforcing minimum segment length:
1. Shrinking segments reduces the numerator of each IED term (shorter segments have lower expected distance to their nearest center).
2. The always-cut policy (CUT at every step, creating single-point segments) drives OD toward zero, achieving arbitrarily low CR regardless of the actual trajectory structure.
3. This makes CR structurally degenerate: the global optimum is a trivial policy that performs no meaningful segmentation.

**Mitigation:** Our shaped reward incorporates:
- **Action constraint** (`L_MIN = 3`): CUT actions within 3 steps of the last cut are overridden to EXTEND, enforcing minimum segment length.
- **Cut penalty** (`CUT_PENALTY = 0.12`): Explicit per-cut reward penalty creates a cost-benefit tradeoff.
- **Extend cost** (`EXTEND_COST = 0.01`): Small cost prevents pure always-extend policies.
- **Complexity regularizer** (`COMPLEXITY_LAMBDA = 0.03`): End-of-episode penalty proportional to cut rate.

**Empirical verification (D1):** The D1 diagnostic (random policies at varying cut rates) confirms that under the shaped reward, ValCR is NOT monotonically decreasing with cut frequency — there exists an interior optimum.

### C4. NISQ-Feasible Circuit Design with Empirical Robustness Evaluation

We separate two aspects of NISQ feasibility:

**a) Feasible circuit design constraints (architectural):**
- Angle encoding (5D state → 5-qubit, no exponential Hilbert space)
- Shallow circuits (3 layers, depth ~15 after transpilation)
- No mid-circuit measurement
- SPSA optimization (2 circuit evaluations per gradient estimate)

**b) Robustness to sampling and noise (empirically evaluated):**
- Shot sensitivity sweep (statevector vs 128/512/2048 shots) — E3
- Noise model sweep (Eagle/Heron hardware-inspired noise) — E2
- These experiments characterize degradation under realistic hardware constraints; they do not constitute a hardware demonstration.

**Note:** Statevector simulation is a valid proxy for noiseless circuit evaluation at 5 qubits (2^5 = 32-dimensional state space). It is NOT a substitute for noise/shot testing, which is handled separately.

---

## What We Do Not Claim

- We do not claim quantum speedup: wall-clock time is dominated by classical environment interaction.
- We do not claim that VQ-DQN outperforms well-tuned classical models with sufficient data and gradient access (Claim A from comparison report shows classical SGD on 3,000 trajectories achieves CR 0.59).
- We do not claim that the observed regime-specific competitiveness generalizes beyond the tested SPSA + small-data conditions.
- We do not claim that "parameter efficiency" under our operational definition implies superior function query complexity or generalization bounds.
- The VQC's advantage, if confirmed by multi-seed testing, may reflect favorable SPSA landscape properties of the ansatz rather than quantum-mechanical effects per se.
