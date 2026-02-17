# Q-RLSTC: Classical vs. Quantum Justifications

> **Purpose**: A rigorous, thesis-grade justification for every architectural decision—why each component is classical, why one is quantum, what it would concretely take to make the classical parts quantum, and how to design experiments that produce cross-comparable results.

---

## 1. The Decision Framework

Every component in Q-RLSTC was assigned to classical or quantum execution based on three criteria:

| Criterion | Question |
|-----------|----------|
| **Algorithmic fit** | Does a known quantum algorithm provide a structural advantage here? |
| **Hardware feasibility** | Can this run on a 5-qubit, depth-11 NISQ circuit within coherence time? |
| **Training loop frequency** | How many times per episode does this execute? Can NISQ latency absorb it? |

Components that fail any criterion default to classical. The quantum budget is deliberately spent on a single, high-value component (the policy network) rather than spread thin.

---

## 2. Component-by-Component Analysis

### 2.1 Feature Extraction — CLASSICAL

**What it does**: Converts raw spatio-temporal trajectory coordinates into a 5-dimensional state vector:

| Feature | Meaning | Computation |
|---------|---------|-------------|
| `od_segment` | Projected OD if we split here | Running average update |
| `od_continue` | Projected OD if we extend | Centroid distance proxy |
| `baseline_cost` | TRACLUS-like MDL compression score | Log-ratio of reconstruction vs. compression cost |
| `len_backward` | Current segment length (normalized) | Point count / trajectory length |
| `len_forward` | Remaining trajectory (normalized) | Remaining count / trajectory length |

**Why classical is correct**:

1. **No quantum subroutine exists for this**. Feature extraction requires sequential geometric computation along a trajectory—curvature estimation, running averages, MDL cost ratios. These are inherently sequential, path-dependent operations. No quantum algorithm offers speedup for sequential path-following.

2. **Executes every step of every episode**. With ~100 points per trajectory and ~500 trajectories, feature extraction runs ~50,000 times per epoch. Even simulated quantum execution at 100ms/circuit would add 83 minutes per epoch. Classical execution is <1ms total.

3. **Spatio-temporal trajectories are semantic, not isolated vectors**. A trajectory is an ordered sequence of points with temporal dependencies. The features require looking backward (segment curvature) and forward (remaining length). Quantum state preparation would destroy this sequential structure—you cannot "look backward along a trajectory" in superposition. The semantic connection between consecutive points is fundamentally classical information processing.

**What it would take to make this quantum**:
- You would need a quantum random access memory (qRAM) to hold trajectory points in superposition — qRAM is theoretical and does not exist on any hardware.
- Even with qRAM, the features depend on *conditional* geometry (curvature from the *current split point*, not the whole trajectory). There is no known quantum advantage for conditional geometric queries over ordered sequences.
- **Qubit cost**: To encode a trajectory of 100 2D points, you would need log₂(100) ≈ 7 address qubits + precision qubits for coordinates. Even a toy version would require ~15 qubits just for data, before any computation qubits—tripling the circuit width for zero algorithmic gain.

> **Verdict**: Classical is the only defensible choice. Quantum feature extraction is not a simplification—it is a complication with no payoff.

---

### 2.2 Policy Network (Q-Value Estimation) — QUANTUM

**What it does**: Maps a 5-dimensional state vector to 2 Q-values (one per action: extend or cut).

**Why quantum is the deliberate research choice**:

1. **Parameter efficiency is the core research claim**. Classical RLSTC uses a 5→64→2 MLP with ~450 parameters. The VQ-DQN achieves comparable expressivity with **20 trainable parameters** (a 22.5× reduction). This is the central "quantum utility" argument: a 5-qubit circuit accesses a 2⁵ = 32-dimensional Hilbert space, providing representational capacity that scales exponentially with qubit count while parameter count scales only linearly (2 rotations × n_qubits × n_layers).

2. **The function approximation task fits quantum structure**. The policy maps ℝ⁵ → ℝ² (5 continuous features to 2 Q-values). This is a low-dimensional, continuous function approximation problem—exactly the regime where variational quantum circuits have theoretical expressivity advantages (see Schuld et al., 2021, "Effect of data encoding on the expressive power of variational quantum machine-learning models").

3. **The encoding maps naturally**. Five features → five qubits, one-to-one via angle encoding. No padding, no truncation, no normalization. The `arctan` scaling maps unbounded features to bounded rotation angles monotonically. This is the cleanest possible quantum encoding scenario.

4. **NISQ-feasible**. The circuit is 5 qubits, depth ~11, with 4 CNOT gates per layer (8 total). This fits within the coherence time of IBM Eagle (~100μs T₂) and Heron (~200μs T₂) processors with estimated fidelities of ~85% and ~95% respectively.

**What it would take to do this classically (the control experiment)**:
- Replace VQ-DQN with a 5→H→2 MLP, where H is tuned to match parameter count.
- A 5→2 linear model has 12 parameters. A 5→4→2 MLP has 30 parameters (matching quantum). A 5→64→2 MLP has ~450 parameters (original RLSTC).
- The experiment should test all three: (a) same parameter count, (b) same architecture class, (c) unconstrained classical.
- **Critical**: Use the same optimizer (SPSA), same loss (Huber), same replay buffer, same hyperparameters. Only the function approximator changes.

**What it would take to make this "more quantum"**:
- Increase to the RY-RZ-RY Euler decomposition (30 params at 2 layers) instead of RY-RZ (20 params). This is the circuit described in technical_reference_part1.
- Add entangling layers with full (not linear) connectivity — requires 10 CNOTs per layer instead of 4, roughly doubling depth. Feasible on Heron (longer T₂) but risky on Eagle.
- Use ZZFeatureMap-style encoding (as in qDINA) for pairwise feature entanglement — adds depth but may capture feature interactions classical networks handle trivially via hidden layers.

---

### 2.3 Distance Computation (IED) — CLASSICAL

**What it does**: Computes Integrated Euclidean Distance between sub-trajectory segments and cluster centroids. Used to compute the reward signal (OD improvement).

**Why classical is correct**:

1. **The hot path demands incremental O(1) updates**. The key optimization is that IED can be computed incrementally: when extending a segment by one point, the distance update is a single trapezoidal area addition. This reduces distance computation from O(n²) to O(1) per step. There is no quantum analog for incremental updates to a running computation—quantum circuits must re-encode and re-compute from scratch each time.

2. **The reward function must be Markov-safe**. The RL framework requires that reward depends only on the current state and action, not on a full re-evaluation. Incremental IED preserves this property. A quantum swap test would require re-encoding the entire segment at every step, destroying the MDP structure.

3. **Trajectory distances are geometric, not algebraic**. IED measures "area between trajectories"—a geometric integral approximated by trapezoidal summation. This is coordinate arithmetic, not inner product estimation. The swap test computes |⟨ψ|φ⟩|², which is algebraically useful but geometrically unrelated to how trajectories diverge in space-time.

**What it would take to make this quantum**:

The swap test path exists but has severe costs:

| Requirement | Cost |
|-------------|------|
| Amplitude encoding per segment | O(n) gates for n-dimensional vector |
| Re-encoding when segment grows | Full circuit rebuild per step |
| Qubit count | 2×log₂(d) + 1 ancilla (d = segment dimension) |
| Shots for precision | ≥4096 for ±0.01 distance accuracy |
| Circuit evaluations per episode | ~100 (one per point) instead of 0 incremental |

For a 100-point trajectory with 10 clusters, quantum distance would require ~1,000 circuit evaluations per episode (100 points × 10 cluster comparisons), each taking ~100ms on simulator. That's **100 seconds** per episode vs. **<1ms** classical. If the swap test is needed once per episode (a final comparison), the cost is tolerable; if needed per step, it is 100,000× slower.

**Where quantum distance IS used (and justified)**: As an optional verification path for research. The swap test runs at episode-end to compare the quantum distance estimate against the classical IED. This measures quantum fidelity without affecting training speed. This is the correct use: quantum as a validation probe, not as a training bottleneck.

---

### 2.4 Clustering (K-Means) — CLASSICAL

**What it does**: Groups sub-trajectory segments into k clusters. Runs at episode-end (not per-step). Computes the OD metric used for final evaluation.

**Why classical is correct**:

1. **Centroid update has no known quantum speedup**. K-means has two operations: assignment (distance computation) and update (mean computation). Even if quantum accelerated the assignment step (via swap test), the update step—computing the arithmetic mean of assigned points—has no quantum advantage. Quantum states cannot efficiently compute arithmetic means without measurement and classical post-processing.

2. **Runs infrequently**. K-means runs once per epoch-end for evaluation, not during the training loop. The computational bottleneck is the 50,000+ circuit evaluations for VQ-DQN per epoch, not the single k-means call. Optimizing k-means provides < 0.01% of total runtime improvement.

3. **Debugging requires classical transparency**. Cluster assignments must be inspectable, reproducible, and verifiable. Quantum clustering introduces measurement randomness into a component that should be deterministic for experimental control.

**What it would take to make this quantum (full quantum k-means)**:

Following the qmeans approach:
- **Assignment**: Swap test for distance estimation. Requires amplitude encoding of each segment (O(d) prep per segment) and each centroid. For k=10 clusters and 500 segments, this is 5,000 swap test circuits per iteration.
- **Update**: Still classical. No quantum mean algorithm exists.
- **Convergence check**: Classical (compare centroid shifts).
- **Qubits**: 2×log₂(d)+1 for swap test. For d=50 (reasonable segment representation), log₂(50) ≈ 6 → 13 qubits. Feasible on current hardware, but amplitude encoding circuits would be deep (~50 gates).
- **Total per-epoch cost**: ~5,000 circuits × ~10 iterations = 50,000 circuit evaluations, comparable to the VQ-DQN training overhead.

**Why we don't do this**: It doubles the quantum computational budget for a component that runs once per epoch and provides no training signal. The VQ-DQN already consumes the full coherence-time budget.

---

### 2.5 Reward Computation — CLASSICAL

**What it does**: Computes `R_t = OD_{t-1} - OD_t` (the incremental improvement in overall distance).

**Why classical is correct**: This is pure arithmetic subtraction of two floating-point numbers. Making this quantum would require encoding two scalars into quantum states to compute their difference, which is slower by orders of magnitude than a single CPU subtraction. There is no meaningful analysis to perform here—any claim that reward computation should be quantum would be scientifically indefensible.

**What it would take**: Encoding two floats as amplitude states and applying a controlled subtraction circuit. Minimum 3 qubits for ~8-bit precision. Runtime: ~1ms quantum vs. ~1 nanosecond classical. This is a 10⁶× slowdown for identical output.

---

### 2.6 Boundary Sharpness (Reward Component) — CLASSICAL

**What it does**: Computes the angle between movement vectors before and after a cut point. Higher angle = sharper boundary = better segmentation decision.

```
sharpness = arccos(v₁ · v₂ / (|v₁| × |v₂|)) / π
```

**Why classical is correct**: This is a dot product between two 2D vectors (4 multiplications, 1 addition, 1 division, 1 arccos). Total: ~10 floating point operations. The quantum inner product (swap test) requires:
- Amplitude encoding of two 2D vectors: ~4 qubits + ~8 gates
- Controlled swap: ~4 CNOT gates
- Measurement: ≥100 shots for ±0.1 precision
- Classical post-processing of counts

That's ~100 circuit evaluations to approximate what 10 FP operations compute exactly. The quantum version would be 10,000× slower and less accurate.

**Why this matters for the thesis**: Boundary sharpness is the key geometric signal in trajectory segmentation. It captures the semantic meaning of "this is where the person changed direction." Making this classical means the geometric semantics of trajectories stay in the domain where they are best represented—Euclidean space, not Hilbert space. Trajectories are spatially embedded, and their structure is captured by classical geometry. The quantum component handles the *decision* (should I cut here?), not the *evidence* (how sharp is this turn?).

---

## 3. The Cross-Comparability Problem

### 3.1 The Risk

If the quantum and classical implementations differ in anything beyond the function approximator, results are not attributable to the quantum component. The experiment measures the *system*, not the *circuit*.

### 3.2 What Must Be Identical

| Component | Classical Control | Quantum Experiment | Same? |
|-----------|------------------|--------------------|-------|
| Feature extraction | StateFeatureExtractor | StateFeatureExtractor | ✅ Identical |
| State representation | 5D vector | 5D vector | ✅ Identical |
| Action space | {extend, cut} | {extend, cut} | ✅ Identical |
| Reward function | OD improvement + boundary sharpness | OD improvement + boundary sharpness | ✅ Identical |
| Replay buffer | Size 5000, uniform sampling | Size 5000, uniform sampling | ✅ Identical |
| Exploration | ε-greedy, same schedule | ε-greedy, same schedule | ✅ Identical |
| Target network | Double DQN, same update freq | Double DQN, same update freq | ✅ Identical |
| **Function approximator** | **MLP (5→H→2)** | **VQ-DQN (5 qubits, 2 layers)** | ❌ The variable |
| Optimizer | SPSA (same hyperparameters) | SPSA (same hyperparameters) | ✅ Identical |
| Loss function | Huber (δ=1.0) | Huber (δ=1.0) | ✅ Identical |
| Dataset | Same trajectories, same seed | Same trajectories, same seed | ✅ Identical |
| Evaluation metrics | OD, F1, silhouette | OD, F1, silhouette | ✅ Identical |

### 3.3 Classical Baselines to Implement

To make the quantum advantage claim defensible, test against three classical controls:

**Control A: Parameter-matched MLP**
- Architecture: 5→4→2 (30 parameters, matching VQ-DQN)
- Purpose: Tests whether quantum gains come from *quantum structure* or simply from the *right number of parameters*
- If VQ-DQN outperforms: Quantum circuit provides richer representations than a classical network of equal size

**Control B: Architecture-matched MLP**
- Architecture: 5→64→2 (~450 parameters, matching original RLSTC)
- Purpose: Tests the ceiling—how well can classical do with no parameter budget constraint?
- If VQ-DQN approaches this: Quantum achieves comparable results with 22× fewer parameters

**Control C: Linear Model**
- Architecture: 5→2 (12 parameters)
- Purpose: Tests whether the problem is trivially linear
- If VQ-DQN significantly outperforms: The non-linearity provided by the quantum circuit is necessary

**Critical**: All three controls must use SPSA (not Adam or SGD with backpropagation), because the optimizer choice affects convergence independently of the function approximator. If classical controls use backprop, you cannot distinguish "quantum circuit is worse" from "SPSA is worse than backprop."

### 3.4 What Not To Optimize Differently

| Pitfall | Why It Breaks Comparability |
|---------|-----------------------------|
| Using Adam for classical, SPSA for quantum | Optimizer effects dominate function approximator effects |
| Different batch sizes | Affects gradient variance independently |
| Different learning rate schedules | Convergence rate changes are optimizer artifacts |
| Different shot counts per condition | Added noise in quantum path is an uncontrolled variable |
| Different random seeds | Trajectory order and exploration path differ |
| Running quantum with noise, classical without | You're measuring noise tolerance, not approximation quality |

---

## 4. What to Measure and Test

### 4.1 Primary Metrics

| Metric | What It Measures | How to Compare |
|--------|-----------------|----------------|
| **Overall Distance (OD)** | Clustering quality at convergence | Lower is better. Report mean ± std over 5 seeds |
| **Segmentation F1** | Boundary detection accuracy | Report against ground truth boundaries |
| **Convergence rate** | Episodes to reach 90% of final OD | Faster = more sample-efficient |
| **Parameter count** | Model complexity | Quantum advantage claim: same OD, fewer params |

### 4.2 Boundary Sharpness Analysis

Boundary sharpness is the strongest semantic signal for trajectory segmentation. Measure:

1. **Sharpness distribution at predicted vs. true boundaries**: The agent should learn to cut at high-sharpness points. Histogram comparison between quantum and classical reveals whether the VQ-DQN learns sharper cuts.

2. **Sharpness threshold sensitivity**: At what sharpness value does the agent switch from extend to cut? Plot Q(cut) - Q(extend) as a function of boundary sharpness for both classical and quantum agents. If the quantum agent develops a sharper decision boundary, it suggests the circuit captures non-linear sharpness-to-action mappings better.

3. **False positive analysis**: When the agent cuts at low-sharpness points, is this from noise (quantum) or overfitting (classical)? If quantum produces fewer low-sharpness cuts despite noisy Q-values, the Hilbert space regularization is providing value.

### 4.3 Trajectory Clustering Quality

1. **OD by trajectory complexity**: Segment trajectories by number of true boundaries (2, 4, 8, 16) and measure OD for each. Does quantum degrade faster on complex trajectories?

2. **Silhouette score**: Measures cluster cohesion and separation. Independent of boundary accuracy. Tests whether the segments produced by quantum policy are geometrically coherent.

3. **Cluster assignment stability**: Across 5 seeds, how consistent are the cluster assignments? Higher variance in quantum → noise dominates the policy.

### 4.4 Quantum-Specific Metrics

| Metric | What It Reveals |
|--------|----------------|
| **Noise resilience ratio** | `OD_noisy / OD_ideal` — how much noise degrades quality |
| **Parameter efficiency ratio** | `OD_quantum(30 params) / OD_classical(30 params)` |
| **Circuit fidelity** | Estimated from noise model, validates depth choice |
| **Shot count sensitivity** | OD vs. shots — identifies the noise floor |
| **Gradient variance** | SPSA gradient norm variance — quantum adds shot noise |

### 4.5 Experimental Matrix

For a **project**-level thesis (not coursework), run at minimum:

| Experiment | Variable | Fixed | Measures |
|------------|----------|-------|----------|
| **E1: Function approximator** | VQ-DQN vs. MLP(30) vs. MLP(450) vs. Linear | All else identical, SPSA | Core quantum utility claim |
| **E2: Noise impact** | Ideal vs. Eagle vs. Heron | VQ-DQN, same config | NISQ viability |
| **E3: Shot sensitivity** | 128, 256, 512, 1024, 4096 shots | VQ-DQN ideal | Noise floor identification |
| **E4: Circuit depth** | 1, 2, 3 layers | VQ-DQN, same qubit count | Expressivity vs. noise |
| **E5: Dataset complexity** | 2, 4, 8, 16 true boundaries per trajectory | All systems | Scaling behavior |

For a **thesis**-level contribution, add:

| Experiment | Variable | Measures |
|------------|----------|----------|
| **E6: Encoding comparison** | Angle vs. ZZFeatureMap vs. amplitude | Best encoding for RL state |
| **E7: Entanglement topology** | Linear vs. ring vs. full | Connectivity vs. noise tradeoff |
| **E8: Optimizer comparison** | SPSA vs. parameter-shift (both quantum) | Gradient estimation quality |
| **E9: Transfer learning** | Pre-train on ideal, fine-tune on noisy | Noise adaptation strategy |

---

## 5. Spatio-Temporal Trajectories in Quantum

### 5.1 Why Trajectories Are Hard for Quantum

A trajectory is not a vector—it is an **ordered sequence** with:
- **Temporal ordering**: Point 3 must come after point 2 (causality)
- **Variable length**: Trajectories have 50–500 points
- **Spatial continuity**: Consecutive points are spatially correlated
- **Semantic structure**: Phases (commute, shop, rest) emerge from point sequences, not individual points

Quantum circuits process fixed-width registers. There is no natural way to encode "a variable-length, temporally ordered sequence of 2D coordinates" into qubits without either:
- Truncating/padding to fixed length (losing information)
- Using exponentially many qubits (one per point—impractical)
- Encoding only summary statistics (what we do—the 5D state vector)

### 5.2 The State Vector as Quantum-Compatible Abstraction

The 5-dimensional state vector is the **bridge** between the classical trajectory domain and the quantum policy domain. It deliberately compresses the sequential spatio-temporal information into a fixed-size summary that:

1. Is **bounded** (via arctan encoding), making it safe for rotation angles
2. Is **Markov** (depends only on current position and accumulated statistics), enabling RL
3. Is **semantic** (captures curvature, segment cost, remaining length), preserving decision-relevant information
4. Is **low-dimensional** (5 features → 5 qubits), fitting NISQ constraints

The compression from trajectory → 5D state is inherently classical because it requires sequential geometric computation. The decision based on that state is where quantum provides value.

### 5.3 Contrast: Quantum for Trajectories vs. Quantum for Decisions

| Aspect | Trajectory Processing | Decision Making |
|--------|----------------------|-----------------|
| Data structure | Variable-length sequence | Fixed 5D vector |
| Required operations | Sequential geometry | Function approximation |
| Quantum advantage | None known | Hilbert space expressivity |
| NISQ feasibility | Requires qRAM | 5 qubits, depth 11 |
| Training frequency | Once per step (feature extraction) | Once per step (Q-value) |
| Quantum encoding | Would need amplitude (expensive) | Angle encoding (cheap) |

---

## 6. Summary of Justifications

| Component | Assignment | One-Line Justification |
|-----------|------------|----------------------|
| Feature extraction | Classical | Sequential geometric computation on ordered trajectory; no quantum speedup exists |
| Policy network | **Quantum** | 22× parameter reduction via Hilbert space expressivity; clean 5-feature → 5-qubit mapping; NISQ-feasible |
| Distance (IED) | Classical | Incremental O(1) updates impossible in quantum; re-encoding costs 100,000× overhead |
| K-means clustering | Classical | No quantum centroid update; runs once per epoch (not a bottleneck) |
| Reward computation | Classical | Single floating-point subtraction; quantum encoding overhead would be 10⁶× slower |
| Boundary sharpness | Classical | 10 FP operations vs. 100+ circuit evaluations for same result |
| Swap test (optional) | Quantum | Used only as verification probe at episode-end; does not affect training loop |

### The Argument in One Paragraph

Q-RLSTC uses quantum computation for exactly one component: the policy function approximator. This is the only component where quantum structure provides a demonstrable advantage (parameter efficiency via Hilbert space expressivity), where the input dimensions naturally map to qubits (5 features → 5 qubits), and where NISQ constraints are satisfied (depth 11, 8 CNOTs). Every other component—feature extraction, distance computation, clustering, reward—is either inherently sequential (trajectories), requires incremental updates (IED), or is trivially cheap (reward subtraction). Making these quantum would introduce orders-of-magnitude overhead for zero algorithmic improvement. The experimental design ensures this choice is testable: by comparing against classical controls with identical everything-except-the-approximator, any observed difference is attributable to the quantum circuit's representational properties, not to system-level confounding.
