# Q-RLSTC Technical Reference Document

## Part 1: System Architecture & Philosophy

### 1.1 What is Q-RLSTC?

Q-RLSTC (Quantum-enhanced Reinforcement Learning for Sub-Trajectory Clustering) is a hybrid quantum-classical system that learns to segment GPS trajectories into semantically meaningful sub-trajectories. The system uses a Variational Quantum Circuit as the policy network in a Deep Q-Learning framework.

The fundamental insight is that trajectory segmentation can be framed as a sequential decision problem: at each point along a trajectory, the agent must decide whether to **extend** the current segment or **cut** to start a new one. This binary decision, repeated for every point, produces a complete segmentation.

### 1.2 The Problem Domain

**Sub-trajectory clustering** groups portions of trajectories that share similar movement patterns. Unlike whole-trajectory clustering, this captures the reality that a single journey often contains multiple phases—commuting, shopping, recreation—each with distinct characteristics.

The challenge is determining where to place boundaries. Too few produces oversimplified segments that miss behavioral changes. Too many creates noise-sensitive fragments that don't generalize. The optimal balance depends on:

- Movement characteristics (speed, direction, acceleration)
- Spatial proximity to other trajectories
- The goal of minimizing overall distance to cluster centroids

### 1.3 Why Reinforcement Learning?

Traditional segmentation uses hand-crafted heuristics (angular threshold, distance-based splits) or exhaustive search (dynamic programming). RL offers an alternative: learn a policy that makes locally optimal decisions that lead to globally optimal segmentations.

The MDP (Markov Decision Process) structure:
- **State**: Features describing the current decision context
- **Action**: Binary—extend current segment or cut
- **Reward**: Improvement in clustering quality (reduction in overall distance)
- **Transition**: Move to next point along trajectory

This framing allows the agent to learn complex patterns without explicit programming of segmentation rules.

### 1.4 Why Quantum?

The classical RLSTC uses a Deep Q-Network with approximately 400 trainable parameters. Q-RLSTC replaces this with a Variational Quantum Circuit containing approximately 30 parameters while maintaining comparable expressivity.

**Rationale for quantum approach:**

1. **Parameter Efficiency**: Quantum circuits can represent functions in high-dimensional Hilbert space with fewer parameters than classical networks. A 5-qubit circuit accesses a 32-dimensional state space.

2. **NISQ Hardware Alignment**: The VQ-DQN design targets near-term quantum computers. Shallow circuits (depth 2), limited qubit count (5), and noise-aware training make execution feasible on current hardware.

3. **Research Contribution**: Demonstrating quantum advantage (or quantum utility) in RL applications is an active research area. Q-RLSTC provides a concrete testbed.

**Important caveat**: Q-RLSTC does not claim quantum speedup. The goal is parameter efficiency and validation of quantum RL architectures, not raw performance gains.

### 1.5 System Overview

The system operates in three integrated layers:

**Layer 1: Feature Extraction (Classical)**
Converts raw trajectory coordinates into a 5-dimensional state vector:
- OD Proximity: Similarity to cluster origin/destination patterns
- TRACLUS Baseline: Classical distance-based segmentation signal
- Segment Length: Normalized length of current segment
- Running OD: Current overall distance metric
- Segment Count: Normalized number of segments created

**Layer 2: Policy Network (Quantum)**
The VQ-DQN circuit processes the 5-dimensional state and outputs Q-values for each action (extend, cut). The circuit uses angle encoding to map classical features to qubit rotations, followed by variational layers that learn the optimal policy.

**Layer 3: Clustering (Classical)**
Segments are grouped using k-means clustering with representative trajectory centroids. Distance computations can use either classical Euclidean distance or quantum swap test estimation.

### 1.6 Design Philosophy

**Hybrid First**: Pure quantum solutions are not viable for NISQ. Q-RLSTC uses quantum computation only where it provides value (policy network), keeping everything else classical.

**NISQ Awareness**: Every circuit design decision prioritizes noise resilience. This means:
- Shallow depth (errors compound with depth)
- Limited qubit count (fewer qubits = fewer error sources)
- Repeated measurements (statistical averaging reduces shot noise)

**Modularity**: Components are designed for independent testing and replacement. The VQ-DQN can be swapped for a classical DQN. The clustering can use purely classical distance. This enables controlled experiments.

**Reproducibility**: Random seeds are exposed at every level. Circuit construction is deterministic given parameters. Results should be reproducible across runs.

---

## Part 2: Quantum Encoding Dichotomy

### 2.1 The Two Encoding Strategies

Q-RLSTC uses two distinct quantum encoding approaches for different purposes:

| Component | Encoding | Qubits | Purpose |
|-----------|----------|--------|---------|
| VQ-DQN Policy | Angle Encoding | 5 | Map state to Q-values |
| Distance Estimation | Amplitude Encoding | log₂(n) | Estimate Euclidean distance |

This separation is intentional and reflects fundamental tradeoffs in quantum data encoding.

### 2.2 Angle Encoding (VQ-DQN)

**Mechanism**: Each feature value θᵢ is mapped to a rotation angle applied to qubit i:

```
RY(2·arctan(xᵢ)) applied to qubit i
```

**Why arctan scaling?**
- Unbounded features (any real number) map to bounded angles (-π, π)
- Preserves sign: positive features → positive angles
- Monotonic: larger features → larger angles
- Graceful saturation: extreme values don't cause numerical issues

**Properties of angle encoding:**
- Linear qubit count: n features → n qubits
- Efficient preparation: Single layer of single-qubit gates
- No normalization required: Each feature encoded independently
- Simple gradient computation: Parameter-shift rule applies directly

**Why not amplitude encoding for VQ-DQN?**
Amplitude encoding would require:
- Normalizing the state vector (losing magnitude information)
- More complex state preparation circuits
- Only 3 qubits for 5 features (log₂(8) = 3), but entanglement overhead

Angle encoding's simplicity and direct parameter-shift gradient computation make it superior for the RL training loop.

### 2.3 Amplitude Encoding (Swap Test Distance)

**Mechanism**: A normalized vector x is encoded into the amplitudes of a quantum state:

```
|ψ⟩ = Σᵢ xᵢ|i⟩ where x is normalized (||x||=1)
```

**Why amplitude encoding for distance?**
The swap test estimates inner product between two amplitude-encoded states. Combined with known norms, this yields Euclidean distance:

```
d(x,y) = √(||x||² + ||y||² - 2·||x||·||y||·⟨ψ_x|ψ_y⟩)
```

This is a fundamentally quantum algorithm—no efficient classical circuit can compute inner products in log(n) time.

**Preparation complexity:**
Amplitude encoding requires O(n) gates for arbitrary n-dimensional vectors (state preparation is generally expensive). However:
- Trajectory segments have fixed structure
- State preparation circuits can be cached and reused
- For clustering, the same centroids are compared against many points

### 2.4 The Dichotomy's Purpose

Using different encodings for different tasks optimizes each component:

**VQ-DQN needs**:
- Fast encoding (executes thousands of times during training)
- Direct gradient access (parameter-shift rule)
- No normalization (preserve feature magnitudes)
- Fixed qubit count (5 features = 5 qubits)

**Distance estimation needs**:
- Quantum inner product (the algorithmic advantage)
- Logarithmic qubit scaling (handle high-dimensional trajectory data)
- Amplitude structure (mathematically required for swap test)

A single encoding scheme would force compromises in both use cases.

---

## Part 3: VQ-DQN Circuit Design

### 3.1 Circuit Architecture Overview

The VQ-DQN circuit follows a layered structure:

```
[Angle Encoding] → [Variational Layer 1] → [Re-encoding] → [Variational Layer 2] → [Measurement]
```

Each component serves a specific purpose in the policy function approximation.

### 3.2 Input Layer: Angle Encoding

The 5-dimensional state vector is encoded using RY rotations:

```
For each qubit i in [0,4]:
    Apply RY(2·arctan(state[i])) to qubit i
```

This creates an initial state that encodes the decision context. All encoding gates commute, so execution order doesn't matter.

### 3.3 Variational Layers: Hardware-Efficient Ansatz

Each variational layer applies:

**Rotation block**: Three parameterized rotations per qubit
```
For each qubit i:
    RY(θ₁) → RZ(θ₂) → RY(θ₃)
```

The RY-RZ-RY sequence can represent any single-qubit rotation (Euler decomposition). This provides maximum expressivity per qubit.

**Entanglement block**: Linear CNOT chain
```
CNOT(0,1) → CNOT(1,2) → CNOT(2,3) → CNOT(3,4)
```

Linear entanglement (vs. full or ring) is chosen for:
- Lower circuit depth (gates execute sequentially, not in parallel)
- Better noise resilience (fewer two-qubit gates)
- Sufficient expressivity for 5-qubit systems

### 3.4 Data Re-uploading

Between variational layers, the input state is re-encoded:

```
After Layer 1:
    For each qubit i: Apply RY(2·arctan(state[i]))
Then Layer 2
```

**Why re-upload?**
Standard VQCs suffer from the "barren plateau" problem and limited expressivity for input dependence. Data re-uploading:
- Increases circuit expressivity without increasing depth
- Allows each layer to process input differently
- Empirically improves trainability

This technique comes from "Data re-uploading for a universal quantum classifier" (Pérez-Salinas et al., 2020).

### 3.5 Parameter Count

Total trainable parameters:
```
Parameters per layer = 5 qubits × 3 rotations = 15
Total = 15 × 2 layers = 30 parameters
```

Compare to classical RLSTC DQN: ~400 parameters (two hidden layers of 32 neurons).

The 13× reduction in parameters is the primary quantum advantage claim.

### 3.6 Q-Value Extraction

After circuit execution, Z-expectation values are computed for the first 2 qubits:

```
Q(extend) = ⟨Z₀⟩ × scale₀ + bias₀
Q(cut) = ⟨Z₁⟩ × scale₁ + bias₁
```

The expectation ⟨Zᵢ⟩ ranges from -1 to +1, scaled and biased to produce realistic Q-values.

**Why Z-expectation?**
- Direct measurement in computational basis gives bitstrings
- Expectation computed as: ⟨Z⟩ = (n₀ - n₁) / (n₀ + n₁)
- No additional rotation needed (unlike Pauli-X or Y)

### 3.7 Circuit Depth Analysis

Total depth breakdown:
- Encoding layer: 1 (parallel RY gates)
- Variational layer 1: ~4 (3 rotations + CNOT chain)
- Re-encoding: 1
- Variational layer 2: ~4
- Measurement: 1

**Total: ~11 layers**

This shallow depth is critical for NISQ execution where decoherence limits circuit duration.
