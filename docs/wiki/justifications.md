# Classical vs. Quantum Justifications

[← Back to README](../../README.md) · [Distance & Clustering](distance_and_clustering.md) · **Justifications** · [Comparison →](comparison.md)

---

> A rigorous, thesis-grade justification for every architectural decision — why each component is classical, why one is quantum, and what it would concretely take to make the classical parts quantum.

## Decision Framework

Every component was assigned to classical or quantum execution based on three criteria:

| Criterion | Gate Question |
|---|---|
| **Algorithmic fit** | Does a known quantum algorithm provide a structural advantage here? |
| **Hardware feasibility** | Can this execute on a 5-qubit, depth-11 NISQ circuit within coherence time? |
| **Training loop frequency** | How many times per episode does this execute? Can NISQ latency absorb it? |

Components that fail _any_ criterion default to classical. The quantum budget is spent on a single, high-value component (the policy network) rather than spread thin.

---

## Component-by-Component Analysis

### Feature Extraction — CLASSICAL ✅

**What:** Converts raw trajectory coordinates into a 5D/8D state vector.

**Why classical:**
1. **No quantum subroutine exists.** Features require sequential geometric computation along an ordered trajectory — curvature estimation, running averages, MDL cost ratios.
2. **Executes ~50,000 times per epoch.** Even simulated quantum at 100ms/circuit would add 83 minutes. Classical: <1ms total.
3. **Trajectory structure is inherently sequential.** You cannot "look backward along a trajectory" in superposition.

**What it would take to make this quantum:**
- qRAM to hold trajectory points in superposition — theoretically exists, not built
- ~15 qubits just for data (100 points × 2D), before computation qubits
- Zero algorithmic gain for conditional geometric queries

---

### Policy Network (Q-Value Estimation) — QUANTUM 🔷

**What:** Maps 5D state → 2 Q-values.

**Why quantum is the research choice:**
1. **Parameter efficiency** — 20 params vs. ~450 classical (22.5× reduction). A 5-qubit circuit accesses a 2⁵ = 32-dimensional Hilbert space.
2. **Natural input mapping** — 5 features → 5 qubits, one-to-one via angle encoding. No padding, no truncation.
3. **NISQ-feasible** — 5 qubits, depth ~11, 8 CNOTs total. Fits within Eagle (~100μs T₂) and Heron (~200μs T₂).
4. **Expressivity** — Low-dimensional continuous function approximation is the regime where VQCs have theoretical advantages (Schuld et al., 2021).

**Classical control experiments:**
- Control A: 5→4→2 MLP (30 params — parameter-matched)
- Control B: 5→64→2 MLP (~450 params — architecture-matched)
- Control C: 5→2 linear (12 params — linearity test)
- **Critical:** All controls must use SPSA, not backprop.

---

### Distance Computation (IED) — CLASSICAL ✅

**What:** Integrated Euclidean Distance between sub-trajectories and centroids.

**Why classical:**
1. **Incremental O(1) updates.** No quantum analog exists for incrementally updating a running computation.
2. **Markov-safe.** RL requires reward that depends only on current state/action. Quantum swap test would require full re-encoding per step.
3. **Geometric, not algebraic.** IED measures "area between trajectories" — coordinate arithmetic, not inner products.

**Quantum cost analysis:**
- 100-point trajectory × 10 clusters = 1,000 circuit evaluations per episode
- At 100ms each: **100 seconds** per episode vs. <1ms classical
- 100,000× slowdown for identical output

---

### Clustering (K-Means) — CLASSICAL ✅

**What:** Groups sub-trajectory segments. Runs at episode-end only.

**Why classical:**
1. **No quantum centroid update.** K-means update = arithmetic mean. No quantum advantage for means.
2. **Runs infrequently.** Single call per epoch vs. ~50,000 VQ-DQN evaluations. <0.01% of runtime.
3. **Debugging requires determinism.** Cluster assignments must be inspectable and reproducible.

---

### Reward Computation — CLASSICAL ✅

**What:** `R_t = α·ΔOD + β·sharpness − penalty`

Pure floating-point arithmetic. Making this quantum: 10⁶× slowdown for identical output.

---

### Boundary Sharpness — CLASSICAL ✅

**What:** `arccos(v₁ · v₂ / (|v₁| × |v₂|)) / π` — approximately 10 FP operations.

Quantum version (swap test): ~100 circuit evaluations for less accuracy. 10,000× slower.

**Thesis significance:** Boundary sharpness is the key geometric signal. Its classical implementation preserves trajectory semantics in Euclidean space where they belong. The quantum component handles the _decision_ (should I cut?), not the _evidence_ (how sharp is this turn?).

---

## Summary Table

| Component | Assignment | One-Line Justification |
|---|---|---|
| Feature extraction | Classical | Sequential geometry on ordered trajectory; no quantum speedup exists |
| **Policy network** | **Quantum** | 22× parameter reduction via Hilbert space; clean 5→5 qubit mapping; NISQ-feasible |
| Distance (IED) | Classical | Incremental O(1) impossible in quantum; re-encoding costs 100,000× |
| K-means clustering | Classical | No quantum centroid update; runs once per epoch |
| Reward computation | Classical | Single FP subtraction; quantum 10⁶× slower |
| Boundary sharpness | Classical | 10 FP ops vs. 100+ circuits for same result |
| Swap test (optional) | Quantum | Verification probe only; does not affect training loop |

---

## The Argument in One Paragraph

Q-RLSTC uses quantum computation for exactly one component: the policy function approximator. This is the only component where quantum structure provides a demonstrable advantage (parameter efficiency via Hilbert space expressivity), where input dimensions naturally map to qubits (5 features → 5 qubits), and where NISQ constraints are satisfied (depth 11, 8 CNOTs). Every other component is either inherently sequential (trajectories), requires incremental updates (IED), or is trivially cheap (reward subtraction). Making these quantum would introduce orders-of-magnitude overhead for zero algorithmic improvement.

---

**Next:** [RLSTC vs. Q-RLSTC Comparison →](comparison.md)
