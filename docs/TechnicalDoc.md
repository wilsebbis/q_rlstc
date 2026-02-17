# Q-RLSTC Technical Documentation

> **Version**: 1.0  
> **Last Updated**: 2026-02-06  
> **Architecture**: Hybrid Quantum-Classical RL for Sub-Trajectory Clustering

---

## Table of Contents

1. [System Overview](#system-overview)
2. [MDP Formulation](#mdp-formulation)
3. [VQ-DQN Circuit Architecture](#vq-dqn-circuit-architecture)
4. [Training Pipeline](#training-pipeline)
5. [Classical Clustering](#classical-clustering)
6. [File Reference](#file-reference)

---

## System Overview

Q-RLSTC is a hybrid quantum-classical framework for learning optimal sub-trajectory segmentation policies. The system uses a Variational Quantum Deep Q-Network (VQ-DQN) to learn when to segment trajectories, optimizing for clustering quality.

### Quantum vs Classical Boundary

| Component | Implementation | Rationale |
|-----------|----------------|-----------|
| Q-value estimation | **Quantum** (VQ-DQN) | Variational expressivity for policy |
| State encoding | **Quantum** (Angle) | Bounded features → rotation angles |
| Distance computation | **Classical** | Avoids amplitude encoding overhead |
| Clustering (K-Means) | **Classical** | Efficient, debuggable, episode-end only |
| Reward computation | **Classical** | Incremental, Markov-safe |

### Data Flow

```
Trajectory → StateFeatureExtractor → VQ-DQN Circuit → Q-values → Action
                                                          ↓
                                          MDPEnvironment.step() → Reward
                                                          ↓
                                          ReplayBuffer → SPSA Update
```

---

## MDP Formulation

The segmentation problem is formulated as a Markov Decision Process where an agent traverses a trajectory point-by-point, deciding whether to extend the current segment or cut.

### State Space (5 dimensions)

Defined in `q_rlstc/data/features.py` → `StateFeatureExtractor.extract_features()`:

```python
state = [
    od_segment,       # OD if we split here
    od_continue,      # OD if we extend
    baseline_cost,    # TRACLUS-like MDL score
    len_backward,     # Current segment length (normalized)
    len_forward       # Remaining trajectory (normalized)
]
```

**Encoding**: All features are bounded via `θ = 2·arctan(x)` before quantum encoding.

### Action Space

| Action | Value | Effect |
|--------|-------|--------|
| EXTEND | 0 | Add next point to current segment |
| CUT | 1 | End current segment, start new one |

### Anti-Gaming Constraints

Defined as class constants in `MDPEnvironment`:

```python
MIN_SEGMENT_LEN = 3   # CUT disallowed if segment < 3 points
MAX_SEGMENTS = 50     # Episode terminates if exceeded
SEGMENT_PENALTY = 0.1 # λ: penalty per new segment (prevents over-cutting)
```

**Enforcement**: If `action == CUT` but `current_segment_len < MIN_SEGMENT_LEN`, the action is forced to EXTEND.

### Reward Function

The reward is computed incrementally (no full k-means per step):

```python
def compute_reward():
    reward = 0.0
    
    if action == CUT:
        # 1. Boundary sharpness bonus (direction change at cut point)
        reward += boundary_sharpness * 0.5
        
        # 2. Segment penalty (anti-gaming)
        reward -= SEGMENT_PENALTY
    
    # 3. Local variance improvement
    variance_delta = old_variance - new_variance
    reward += variance_delta * 0.1
    
    return reward
```

**Boundary Sharpness**: Angle between vectors before/after boundary, normalized to [0, 1]:
```python
cos_angle = dot(v1, v2) / (|v1| * |v2|)
sharpness = arccos(cos_angle) / π  # Higher = sharper turn
```

### Termination Conditions

1. **End of trajectory**: All points consumed
2. **Max segments exceeded**: `n_segments >= MAX_SEGMENTS`

---

## VQ-DQN Circuit Architecture

Defined in `q_rlstc/quantum/vqdqn_circuit.py`.

### Circuit Structure

```
┌──────────────────────────────────────────────────────────────────┐
│ ENCODING: RY(2·arctan(xᵢ)) on each qubit                        │
├──────────────────────────────────────────────────────────────────┤
│ LAYER 1: Variational (RY-RZ) + CNOT chain                       │
├──────────────────────────────────────────────────────────────────┤
│ REUPLOADING: RY(2·arctan(xᵢ)) repeated                          │
├──────────────────────────────────────────────────────────────────┤
│ LAYER 2: Variational (RY-RZ) + CNOT chain                       │
├──────────────────────────────────────────────────────────────────┤
│ MEASUREMENT: All qubits in computational basis                  │
└──────────────────────────────────────────────────────────────────┘
```

### Angle Encoding

```python
def angle_encode(features, scaling='arctan'):
    """Map features to rotation angles."""
    return 2 * np.arctan(features)  # Maps (-∞, ∞) → (-π, π)
```

### Variational Layer

Each layer applies:
1. **RY(θ₂ᵢ)** and **RZ(θ₂ᵢ₊₁)** on each qubit i
2. **CNOT chain**: 0→1→2→3→4 (linear, NO ring closure)

```
Qubit 0: ─RY(θ₀)─RZ(θ₁)─●────────────────────────
                        │
Qubit 1: ─RY(θ₂)─RZ(θ₃)─X──●─────────────────────
                           │
Qubit 2: ─RY(θ₄)─RZ(θ₅)────X──●──────────────────
                              │
Qubit 3: ─RY(θ₆)─RZ(θ₇)───────X──●───────────────
                                 │
Qubit 4: ─RY(θ₈)─RZ(θ₉)──────────X───────────────
```

### Parameter Count

| Component | Count |
|-----------|-------|
| Qubits | 5 |
| Rotations per qubit | 2 (RY, RZ) |
| Layers | 2 |
| **Total trainable** | **20** |

Encoding angles are fixed from state, not trained.

### Q-Value Extraction

```python
def evaluate_q_values(state, params, backend, shots):
    # 1. Build circuit with encoded state and variational params
    circuit = build_vqdqn_circuit(state, params)
    
    # 2. Execute on backend
    counts = backend.run(circuit, shots=shots).result().get_counts()
    
    # 3. Extract Z-expectations for qubits 0 and 1
    Q_extend = expectation_Z(counts, qubit=0)  # ∈ [-1, 1]
    Q_cut = expectation_Z(counts, qubit=1)     # ∈ [-1, 1]
    
    # 4. Apply linear scaling (optional)
    return [Q_extend * scale[0] + bias[0],
            Q_cut * scale[1] + bias[1]]
```

**Expectation Computation** (from measurement bitstrings):
```python
def compute_expectation_from_counts(counts, shots, qubit_idx, n_qubits):
    expectation = 0.0
    for bitstring, count in counts.items():
        bit = bitstring[-(qubit_idx + 1)]  # Little-endian
        sign = 1 if bit == '0' else -1
        expectation += sign * count
    return expectation / shots
```

---

## Training Pipeline

Defined in `q_rlstc/rl/train.py` → `Trainer` class.

### Training Loop

```python
for epoch in range(n_epochs):
    for trajectory in dataset.trajectories:
        # 1. Run episode
        state = env.reset()
        while not done:
            # Get Q-values (512 shots)
            q_values = agent.get_q_values(state)
            
            # ε-greedy action selection
            action = epsilon_greedy(q_values, epsilon)
            
            # Environment step
            next_state, reward, done = env.step(action)
            
            # Store transition
            replay_buffer.push(state, action, reward, next_state, done)
            
            # SPSA update from batch
            if buffer_ready:
                batch = replay_buffer.sample(batch_size)
                agent.update(batch)
            
            state = next_state
        
        # Decay epsilon after each episode
        agent.decay_epsilon()
    
    # Episode-end: log metrics (optional k-means evaluation)
```

### SPSA Optimizer

Defined in `q_rlstc/rl/spsa.py`:

```python
class SPSAOptimizer:
    """Simultaneous Perturbation Stochastic Approximation."""
    
    def step(self, loss_fn, params):
        # Sample random direction
        delta = np.random.choice([-1, 1], size=len(params))
        
        # Finite difference gradient estimate (2 circuit evals)
        loss_plus = loss_fn(params + c * delta)
        loss_minus = loss_fn(params - c * delta)
        
        grad_estimate = (loss_plus - loss_minus) / (2 * c * delta)
        
        # Update parameters
        new_params = params - lr * grad_estimate
        return new_params, grad_estimate
```

**Why SPSA**: Cannot use backpropagation through quantum circuits. SPSA estimates gradients with only 2 function evaluations regardless of parameter count.

### TD Loss

```python
def compute_td_loss(state, action, target):
    q_value = get_q_values(state)[action]
    td_error = target - q_value
    
    # Huber loss (robust to outliers)
    delta = 1.0
    if abs(td_error) <= delta:
        return 0.5 * td_error ** 2
    else:
        return delta * (abs(td_error) - 0.5 * delta)
```

### Target Computation (Double DQN)

```python
def compute_target(reward, next_state, done):
    if done:
        return reward
    
    # Online network selects best action
    online_q = get_q_values(next_state, use_target=False)
    best_action = argmax(online_q)
    
    # Target network evaluates
    target_q = get_q_values(next_state, use_target=True)
    
    return reward + gamma * target_q[best_action]
```

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 0.99 | Discount factor |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_min` | 0.1 | Minimum exploration rate |
| `epsilon_decay` | 0.99 | Decay per episode |
| `shots` | 512 | Measurement shots (training) |
| `batch_size` | 32 | Replay batch size |
| `memory_size` | 10000 | Replay buffer capacity |
| `target_update_freq` | 10 | Episodes between target sync |

---

## Classical Clustering

Defined in `q_rlstc/clustering/classical_kmeans.py`.

### When Used

- **Episode-end evaluation**: Compute final OD and F1 metrics
- **NOT per-step**: Reward uses local incremental proxies

### Algorithm

Standard Lloyd's algorithm with k-means++ initialization:

```python
class ClassicalKMeans:
    def fit(self, data):
        # 1. Initialize centroids (k-means++)
        centroids = self._initialize_centroids(data)
        
        for iteration in range(max_iter):
            # 2. Assign points to nearest centroid
            labels = self._assign_clusters(data, centroids)
            
            # 3. Update centroids as cluster means
            new_centroids = self._update_centroids(data, labels)
            
            # 4. Check convergence
            if max_shift < threshold:
                break
            
            centroids = new_centroids
        
        return KMeansResult(centroids, labels, objective)
```

### Vectorized Distance Computation

```python
def _distance_matrix(self, data, centroids):
    # (n, 1, d) - (1, k, d) -> (n, k, d) -> (n, k)
    diff = data[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)
```

---

## File Reference

### Core Modules

| File | Purpose |
|------|---------|
| `q_rlstc/quantum/vqdqn_circuit.py` | VQ-DQN circuit builder, angle encoding |
| `q_rlstc/quantum/backends.py` | Aer simulator factory |
| `q_rlstc/quantum/mitigation.py` | Readout error mitigation |
| `q_rlstc/rl/vqdqn_agent.py` | Agent wrapper (ε-greedy, target network) |
| `q_rlstc/rl/spsa.py` | SPSA optimizer implementation |
| `q_rlstc/rl/train.py` | Training loop, MDP environment |
| `q_rlstc/rl/replay_buffer.py` | Experience replay |
| `q_rlstc/clustering/classical_kmeans.py` | K-means for evaluation |
| `q_rlstc/clustering/metrics.py` | OD, silhouette, F1 metrics |
| `q_rlstc/data/features.py` | State feature extraction |
| `q_rlstc/data/synthetic.py` | Trajectory generation |
| `q_rlstc/config.py` | Configuration dataclasses |

### Key Classes

| Class | File | Responsibility |
|-------|------|----------------|
| `VQDQNCircuitBuilder` | vqdqn_circuit.py | Build parameterized circuits |
| `VQDQNAgent` | vqdqn_agent.py | RL agent with Q-learning |
| `SPSAOptimizer` | spsa.py | Gradient-free optimization |
| `MDPEnvironment` | train.py | Segmentation MDP simulation |
| `Trainer` | train.py | Orchestrate training |
| `ClassicalKMeans` | classical_kmeans.py | Episode-end evaluation |
| `StateFeatureExtractor` | features.py | 5-dim state computation |

### Key Functions

| Function | File | Purpose |
|----------|------|---------|
| `angle_encode()` | vqdqn_circuit.py | Features → rotation angles |
| `evaluate_q_values()` | vqdqn_circuit.py | Circuit execution → Q-values |
| `build_vqdqn_circuit()` | vqdqn_circuit.py | Convenience circuit builder |
| `get_backend()` | backends.py | Get Aer simulator |
| `kmeans_fit()` | classical_kmeans.py | Convenience K-means |
| `od_improvement_reward()` | metrics.py | Reward signal |
| `segmentation_f1()` | metrics.py | Boundary F1 score |

---

## NISQ Constraints Summary

| Constraint | Value | Rationale |
|------------|-------|-----------|
| Qubits | 5 | Matches state dimensionality |
| Circuit depth | 2 layers | Minimize decoherence |
| Trainable parameters | 20 | Below barren plateau threshold |
| Shots (training) | 512 | Balance noise vs speed |
| Shots (inference) | 1024 | Lower variance for metrics |
| Entanglement | Linear chain | Simpler transpilation |
| Optimization | SPSA | Gradient-free, shot-noise robust |

---

*See [TechnicalDoc2.md](TechnicalDoc2.md) for implementation deep-dives and debugging guides.*
