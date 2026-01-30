# Q-RLSTC Technical Reference Document

## Part 4: Distance Estimation & Clustering

### 4.1 The Distance Computation Problem

Trajectory clustering requires comparing trajectories to cluster centroids repeatedly. Each training episode performs hundreds of distance computations. This creates two challenges:

1. **Computational cost**: Naive pairwise distance is expensive
2. **Quantum opportunity**: Distance estimation can use quantum algorithms

Q-RLSTC addresses both via incremental classical distance with optional quantum swap test.

### 4.2 Integrated Euclidean Distance (IED)

IED is the primary distance metric, inherited from classical RLSTC. It measures the area between two trajectories over their temporal overlap.

**Geometric interpretation**: 
Imagine two trajectories as paths through space-time. IED computes the integral of the spatial distance between them across their common time interval.

**Mathematical formulation**:
For two aligned segments with matching time intervals [t_s, t_e]:
- At time t_s: distance d₁ between the two starting points
- At time t_e: distance d₂ between the two ending points
- IED contribution = 0.5 × (d₁ + d₂) × (t_e - t_s)

This trapezoidal approximation efficiently captures the "area" between trajectory segments.

**Why IED over alternatives?**

| Metric | Pros | Cons | Use Case |
|--------|------|------|----------|
| IED | Fast incremental updates, handles different lengths | Requires temporal overlap | Real-time trajectory clustering |
| Fréchet | Shape-invariant, robust | O(n²) computation | Offline analysis |
| DTW | Handles speed variation | Expensive, not incremental | Time series alignment |
| Euclidean | Simple, fast | Requires same length | Fixed-size vectors only |

### 4.3 Incremental Distance Computation

The key insight: as the agent considers each point, it doesn't recompute the full trajectory distance. Instead, it incrementally updates from the previous computation.

**State maintained per cluster**:
- `mid_dist`: Distance for the temporally overlapping portion
- `real_dist`: Total distance including non-overlapping portions
- `lastp`: Last processed point in the cluster center
- `j`: Current index into the cluster center

**When extending a segment** (action = 0):
- Add the new point to the growing segment
- Update only the portion of IED affected by this point
- Amortized O(1) per point rather than O(n)

**When cutting** (action = 1):
- Finalize the current segment's distance
- Reset the incremental state
- Begin fresh computation for the new segment

This optimization is critical—without it, training would be 100× slower.

### 4.4 Quantum Distance via Swap Test

The swap test is a quantum algorithm that estimates the inner product between two amplitude-encoded quantum states.

**The algorithm**:
1. Prepare ancilla qubit in |0⟩
2. Apply Hadamard to ancilla: |+⟩
3. Controlled-swap between two data registers (conditioned on ancilla)
4. Apply Hadamard to ancilla
5. Measure ancilla

**Measurement statistics**:
- P(0) = 0.5 × (1 + |⟨ψ|φ⟩|²)
- P(1) = 0.5 × (1 - |⟨ψ|φ⟩|²)

**Converting to Euclidean distance**:
For normalized vectors x, y:
- |⟨ψ_x|ψ_y⟩|² = |x·y|² / (||x||² × ||y||²)
- d(x,y)² = ||x||² + ||y||² - 2×||x||×||y||×|⟨ψ_x|ψ_y⟩|

**When Q-RLSTC uses swap test**:
- Optional mode for clustering distance estimation
- Primarily for research validation, not performance
- Requires amplitude encoding (expensive state preparation)
- Provides quantum computational comparison point

### 4.5 Hybrid K-Means Clustering

Q-RLSTC uses k-means for clustering sub-trajectories, but with a hybrid design:

**Quantum component** (optional):
- Distance estimation via swap test
- Provides logarithmic qubit scaling for high-dimensional data

**Classical component** (always):
- Centroid computation as arithmetic mean
- Time-aligned averaging for representative trajectories
- Point filtering by temporal coverage threshold

**Why not full quantum k-means?**
Pure quantum k-means would require quantum centroid updates, which don't have known efficient algorithms. The hybrid approach uses quantum where it helps (distance) and classical where it's efficient (means).

### 4.6 Cluster Center Computation

Representative trajectories are computed differently from vector centroids:

**Time-aligned averaging**:
1. Collect all points from sub-trajectories in the cluster
2. Group points by timestamp
3. Filter timestamps with sufficient trajectory coverage
4. Merge nearby timestamps within a threshold
5. Compute arithmetic mean of coordinates at each time slot

This produces a "consensus trajectory" that represents the cluster's movement pattern.

**Thresholds**:
- `threshold_num`: Minimum trajectories required at a timestamp (default: 3)
- `threshold_t`: Time window for merging nearby timestamps (default: 0.095 normalized units)

---

## Part 5: RL Training & Optimization

### 5.1 The MDP Structure

The trajectory segmentation problem is formulated as a Markov Decision Process:

**State**: 5 features describing the current decision context
- `overall_sim`: Current clustering quality (lower = better)
- `minsim` or `split_overdist`: Projected quality if we split here
- `scaled_od`: Amplified quality signal for gradient learning
- `segment_ratio`: How much of trajectory is in current segment
- `remaining_ratio`: How much trajectory remains

**Actions**: Binary
- 0 = Extend (continue current sub-trajectory)
- 1 = Cut (end current sub-trajectory, start new one)

**Reward**: Immediate improvement in overall distance
- R_t = OD_{t-1} - OD_t
- Positive reward when splitting improves clustering
- Zero reward when extending
- This creates sparse rewards, requiring careful exploration

**Transitions**: Deterministic
- Moving to next point along trajectory
- Final point automatically ends episode

### 5.2 Why SPSA Instead of Parameter-Shift?

Q-RLSTC can use two gradient computation methods:

**Parameter-Shift Rule** (used in TheFinalQRLSTC):
- Exact quantum gradients
- 2 circuit evaluations per parameter
- Total: 2 × 30 = 60 circuits per sample
- Mathematically precise

**SPSA** (used in DINA, available in Q-RLSTC):
- Stochastic gradient estimate
- 2 circuit evaluations total (regardless of parameter count)
- Much faster for larger circuits
- Approximate but unbiased

**Decision logic in Q-RLSTC**:
- For the 30-parameter VQ-DQN: parameter-shift is feasible
- For larger circuits or real hardware: SPSA recommended
- Trade-off: precision vs. computational cost

### 5.3 SPSA Algorithm

Simultaneous Perturbation Stochastic Approximation:

**Per optimization step**:
1. Current parameters θ
2. Random perturbation Δ (each component ±1 with equal probability)
3. Perturbed parameters: θ₊ = θ + c·Δ and θ₋ = θ - c·Δ
4. Evaluate loss at both: L₊ = loss(θ₊), L₋ = loss(θ₋)
5. Gradient estimate: g ≈ (L₊ - L₋) / (2c·Δ)
6. Update: θ_new = θ - α·g

**SPSA hyperparameters**:
- `a` (learning rate base): Controls step size
- `c` (perturbation size): Controls gradient estimate quality
- `A` (stability constant): Prevents early divergence
- Learning rate schedule: α_k = a / (k + A)^0.602
- Perturbation schedule: c_k = c / k^0.101

**Recommended values for Q-RLSTC**:
- A = 20 (larger for more stable early training)
- a = 0.12
- c = 0.08

### 5.4 Experience Replay

Q-RLSTC uses standard experience replay to decorrelate samples:

**Buffer structure**:
- Fixed-size deque (default: 5000 transitions)
- Stores: (state, action, reward, next_state, done)
- Uniform random sampling for minibatches

**Training procedure**:
1. Sample batch_size transitions from buffer
2. For each transition:
   - Compute TD target using Double DQN
   - Compute gradient via parameter-shift or SPSA
3. Average gradients across batch
4. Clip gradient norm (max = 1.0)
5. Apply Adam update

### 5.5 Double DQN for Stability

Standard DQN can overestimate Q-values. Double DQN decouples action selection from evaluation:

**Standard DQN target**:
- target = r + γ × max_a Q_target(s', a)

**Double DQN target**:
- best_action = argmax_a Q_online(s', a)
- target = r + γ × Q_target(s', best_action)

This prevents the target network from reinforcing its own overestimations.

### 5.6 Target Network Updates

Q-RLSTC supports two update strategies:

**Soft update** (default):
- θ_target ← τ × θ_online + (1 - τ) × θ_target
- Apply after every training step
- τ = 0.01 (slow tracking)

**Hard update** (alternative):
- θ_target ← θ_online
- Apply every N episodes
- N = 10 (default)

Soft updates provide smoother training dynamics.

### 5.7 Epsilon-Greedy Exploration

Exploration strategy:
- Start with ε = 1.0 (pure exploration)
- Decay multiplicatively: ε_new = ε × decay_rate
- Minimum ε = 0.1 (always 10% exploration)
- Decay rate = 0.995 per step

The slow decay allows extensive exploration early in training when the policy is random, transitioning to exploitation as it improves.

### 5.8 Outer Training Loop

The complete training algorithm:

```
Initialize cluster centers via k-means++
For each outer iteration:
    For each trajectory in training set:
        Reset MDP environment
        For each point in trajectory:
            Observe state
            Select action (ε-greedy)
            Execute action, receive reward
            Store transition in replay buffer
            If buffer sufficiently full:
                Train on minibatch
                Update target network
        Final point: end episode
    After all trajectories:
        Recompute cluster centers
    If centers converged:
        Break
Output: trained VQ-DQN parameters + cluster assignments
```

---

## Part 6: Comparative Analysis

### 6.1 System Comparison Matrix

| Aspect | RLSTCcode | TheFinalQRLSTC | Q-RLSTC | qDINA | qmeans |
|--------|-----------|----------------|---------|-------|--------|
| **Domain** | Trajectory clustering | Trajectory clustering | Trajectory clustering | Database indexing | General clustering |
| **Problem** | Segmentation | Segmentation | Segmentation | Index selection | Point clustering |
| **RL Framework** | DQN | VQ-DQN | VQ-DQN | DQN/QNN | None (unsupervised) |
| **Classical Network** | 5→64→2 | — | — | 5→128→n_actions | — |
| **Quantum Circuit** | — | 5 qubits, 2 layers | 5 qubits, 2 layers | 8-11 qubits, TwoLocal/BQN | 2-4 qubits per distance |
| **Parameters** | ~450 | ~30 | ~30 | ~100-300 | ~0 (no learning) |
| **Optimizer** | SGD | Adam + param-shift | Adam or SPSA | SPSA | N/A |
| **Distance Metric** | IED | IED | IED + optional swap test | Query cost | Swap test |
| **NISQ Noise** | N/A | Yes | Yes | No | Yes |

### 6.2 RLSTCcode (Classical Baseline)

**Architecture**:
- Two-layer MLP: input(5) → hidden(64) → output(2)
- ReLU activation, Huber loss
- TensorFlow 2.x / Keras implementation

**Key characteristics**:
- Proven approach with published results
- Fast training (no quantum overhead)
- ~450 parameters
- CPU/GPU execution

**What Q-RLSTC inherits**:
- MDP formulation (states, actions, rewards)
- Incremental IED computation
- Representative trajectory centroids
- Training loop structure

**What Q-RLSTC changes**:
- Network architecture (VQC replaces MLP)
- Gradient computation (parameter-shift instead of backprop)
- Optimizer (Adam with gradient clipping instead of SGD)

### 6.3 TheFinalQRLSTC (Direct Predecessor)

**Architecture**:
- Same VQ-DQN as Q-RLSTC but fewer code abstractions
- Hardcoded configurations in some places
- Less modular structure

**Key characteristics**:
- First working quantum implementation
- Validated against RLSTCcode
- Limited noise model support
- Monolithic codebase

**What Q-RLSTC improves**:
- Modular architecture with config dataclasses
- Multiple backend options
- Comprehensive noise model support
- SPSA optimizer option
- Better test coverage
- Clearer documentation

### 6.4 qDINA (Different Domain, Similar Quantum RL)

**Domain**: Database index tuning for replicated PostgreSQL clusters

**Key innovations**:
- Divergent indexing (different replicas have different indexes)
- Query routing based on index configuration
- TwoLocal and Bayesian Quantum Network (BQN) ansätze
- SPSA-only optimization (no parameter-shift)

**Similarities to Q-RLSTC**:
- DQN framework with quantum function approximator
- Binary/discrete action space
- Epsilon-greedy exploration
- Experience replay

**Differences from Q-RLSTC**:
- Much larger action space (2 × replicas × candidates)
- State encoded via ZZFeatureMap (pairwise entanglement)
- TorchConnector for Qiskit-PyTorch integration
- SPSA mandatory (too many parameters for param-shift)
- Gymnasium environment interface

**Lessons for Q-RLSTC**:
- SPSA scales better for larger circuits
- BQN ansatz could improve expressivity
- TorchConnector enables hybrid classical-quantum training

### 6.5 qmeans (Pure Quantum Distance)

**Purpose**: Quantum k-means clustering with swap test distance

**Key innovations**:
- Swap test for inner product estimation
- Amplitude encoding for vectors
- Job batching for hardware execution
- Scikit-learn compatible API

**Relationship to Q-RLSTC**:
- Q-RLSTC can use qmeans-style swap test for clustering
- Same amplitude encoding technique
- Different purpose: qmeans is unsupervised, Q-RLSTC has RL policy

**What Q-RLSTC can import**:
- Swap test circuit design
- Amplitude encoding utilities
- Hardware execution patterns

---

## Part 7: Engineering Tradeoffs & Expected Results

### 7.1 Circuit Depth vs. Noise Resilience

**The tradeoff**:
- Deeper circuits can represent more complex functions
- Deeper circuits accumulate more noise on NISQ hardware
- Each additional layer adds ~4 layers of gate depth

**Q-RLSTC choice**: 2 variational layers
- Total depth ~11 (encoding + 2×layers + measurement)
- Estimated fidelity: ~85% on IBM Eagle, ~95% on IBM Heron
- Sufficient expressivity for 5-dimensional state space

**If you need more expressivity**:
- Increase shots (reduces statistical noise)
- Use data re-uploading (expressivity without depth)
- Accept higher error rates

### 7.2 Parameter Count vs. Trainability

**The tradeoff**:
- More parameters = more expressive function
- More parameters = harder to train (gradient vanishing, barren plateaus)
- More parameters = more gradient computations

**Q-RLSTC choice**: 30 parameters
- 5 qubits × 3 rotations × 2 layers
- 93% fewer than classical DQN (450 params)
- Comparable task performance in benchmarks

**If you need more parameters**:
- Add layers (15 params each)
- Use BQN ansatz (more expressive per parameter)
- Switch to SPSA (avoids parameter-shift overhead)

### 7.3 Shots vs. Statistical Precision

**The tradeoff**:
- More shots = lower variance in expectation values
- More shots = slower execution
- Shot noise affects gradient estimates

**Q-RLSTC defaults**:
- Training: 512 shots (faster, noisier)
- Evaluation: 1024 shots (more precise)
- With noise: 4096+ shots (compensate for decoherence)

**Rule of thumb**:
- Expect √(shots) improvement in precision
- Doubling shots halves the variance
- Diminishing returns above ~8192 shots

### 7.4 Angle Encoding vs. Amplitude Encoding

**Angle encoding**:
- Linear qubit count: n features → n qubits
- Simple gradient computation
- No normalization required
- Fast state preparation

**Amplitude encoding**:
- Logarithmic qubit count: n features → log₂(n) qubits
- Complex state preparation
- Requires normalized data
- Enables swap test distance

**Q-RLSTC choice**: Angle for VQ-DQN, Amplitude for swap test
- VQ-DQN needs fast, repeated encoding during training
- Swap test needs inner product structure (amplitude)
- Different tools for different jobs

### 7.5 Classical vs. Quantum Training

**Classical training** (simulate on CPU):
- Fast iteration (~10ms per circuit)
- Exact expectation values (infinite shots)
- No noise modeling
- Best for development and debugging

**Quantum simulation** (Qiskit Aer):
- Realistic shot statistics
- Configurable noise models
- ~100ms per circuit
- Best for NISQ validation

**Hardware execution** (IBM Quantum):
- Real device noise and errors
- ~1-10s per circuit (queue + execution)
- Limited qubit connectivity
- Best for final validation

### 7.6 Expected Results

**Clustering quality** (Overall Distance):
- Classical RLSTC achieves OD reduction of 20-40% vs. baseline
- Q-RLSTC targets comparable OD within 5% of classical
- Noise degrades performance by ~10-15% on current hardware

**Training time**:
- Classical: ~5 minutes for 500 trajectories
- Simulated quantum: ~30 minutes (parameter-shift)
- Simulated quantum with SPSA: ~10 minutes
- Real hardware: impractical for full training (use pre-trained)

**Parameter efficiency**:
- 93% reduction in parameters vs. classical
- Similar convergence behavior
- Potentially better generalization (regularization effect)

### 7.7 Known Limitations

**Current limitations of Q-RLSTC**:

1. **No proven quantum advantage**: Parameter efficiency is not speedup
2. **Simulation overhead**: Quantum simulation is slower than classical NN
3. **Hardware constraints**: Current NISQ can't run full training loop
4. **Shot noise**: Requires multiple runs for stable gradients
5. **Barren plateaus**: Deep circuits may become untrainable

**When Q-RLSTC makes sense**:
- Research into quantum RL architectures
- Validation of VQC expressivity for RL
- Preparing for fault-tolerant quantum era
- Benchmarking quantum vs. classical approaches

**When classical RLSTC is better**:
- Production deployment
- Speed-critical applications
- Hardware not available
- Larger state/action spaces

### 7.8 Future Directions

**Near-term improvements**:
- BQN ansatz integration from qDINA
- Hardware-efficient noise-adapted training
- Quantum error mitigation beyond readout
- Adaptive circuit depth

**Medium-term research**:
- Quantum-native distance metrics
- End-to-end differentiable quantum clustering
- Hybrid classical-quantum transfer learning
- Noise-aware reward shaping

**Long-term possibilities**:
- Fault-tolerant quantum speedup
- Quantum advantage in trajectory similarity
- Quantum-enhanced spatiotemporal learning
