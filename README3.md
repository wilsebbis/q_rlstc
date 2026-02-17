# Q-RLSTC: Quantum-Enhanced RL for Sub-Trajectory Clustering

Hybrid quantum-classical framework for sub-trajectory clustering using Variational Quantum Deep Q-Networks (VQ-DQN).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Q-RLSTC System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐     ┌─────────────────┐     ┌──────────────┐  │
│  │  Trajectory │────▶│   MDP Env       │────▶│   VQ-DQN     │  │
│  │    Data     │     │  (Segmentation) │     │  (5 qubits)  │  │
│  └─────────────┘     └────────┬────────┘     └──────┬───────┘  │
│                               │                      │          │
│                               ▼                      ▼          │
│  ┌─────────────┐     ┌─────────────────┐     ┌──────────────┐  │
│  │  Classical  │◀────│     Reward      │◀────│   Action:    │  │
│  │   K-Means   │     │  (Local Delta)  │     │  Extend/Cut  │  │
│  └─────────────┘     └─────────────────┘     └──────────────┘  │
│                                                                 │
│  Quantum: Policy/Q-value only │ Classical: Clustering/Reward   │
└─────────────────────────────────────────────────────────────────┘
```

## Quantum Scope (What's Actually Quantum)

**Only the VQ-DQN policy network uses quantum circuits.**

| Component | Implementation | Rationale |
|-----------|----------------|-----------|
| Q-value estimation | **Quantum** (VQ-DQN) | Variational expressivity |
| State encoding | **Quantum** (Angle) | Bounded 5-dim → rotation angles |
| Clustering (K-Means) | **Classical** | Efficient, debuggable |
| Distance computation | **Classical** | Avoids amplitude encoding cost |
| Reward computation | **Classical** | Markov-safe, incremental |

> **Note**: Swap-test amplitude encoding was considered but rejected. It requires deep state-prep circuits and adds shot noise per distance call—impractical for per-step RL.

## MDP Specification

### State Vector (5 dimensions)

```python
state = [
    segment_length,      # Current segment point count (normalized)
    local_variance,      # Within-segment feature variance
    centroid_distance,   # Distance to nearest cluster centroid
    trajectory_progress, # Position in trajectory [0, 1]
    segment_count        # Number of segments so far (normalized)
]
# All features bounded via arctan normalization → [−π, π]
```

### Action Space

| Action | Effect |
|--------|--------|
| `EXTEND` (0) | Add next point to current segment |
| `CUT` (1) | End current segment, start new one |

### Termination

- **End of trajectory**: Episode ends when all points consumed
- **Max segments**: Episode terminates if `segment_count > max_segments`

### Constraints (Anti-Gaming)

- **Minimum segment length**: Cut disallowed if segment has < 3 points
- **Segment count penalty**: Reward -= λ per segment (prevents "cut every step")

### Reward (Local Delta, Not Full Re-Clustering)

```python
def compute_reward(action, old_state, new_state):
    """Incremental reward - NO full k-means per step."""
    
    # Local distortion change (main signal)
    delta_variance = old_state.local_variance - new_state.local_variance
    
    # Segment boundary quality (on CUT only)
    if action == CUT:
        boundary_score = compute_boundary_sharpness(segment)
    else:
        boundary_score = 0
    
    # Segment count penalty (regularization)
    segment_penalty = -LAMBDA * (new_state.segment_count - old_state.segment_count)
    
    return delta_variance + boundary_score + segment_penalty

# Full k-means evaluation: episode end only (for metrics, not training)
```

## VQ-DQN Architecture

### Circuit Template

```
Input Encoding (Angle):
──RY(2·arctan(x₀))──RY(2·arctan(x₁))──...──RY(2·arctan(x₄))──

Variational Layer (×2 layers, data re-uploading):
┌────────────────────────────────────────────────┐
│ RY(θ₀)─RZ(θ₁)─●───────────────────────────────│
│               │                                │
│ RY(θ₂)─RZ(θ₃)─X──●────────────────────────────│
│                  │                             │
│ RY(θ₄)─RZ(θ₅)────X──●─────────────────────────│
│                     │                          │
│ RY(θ₆)─RZ(θ₇)───────X──●──────────────────────│
│                        │                       │
│ RY(θ₈)─RZ(θ₉)──────────X──────────────────────│
└────────────────────────────────────────────────┘
Linear entanglement: CNOT chain (0→1→2→3→4), NO ring closure.

Measurement:
Q(s, EXTEND) = ⟨Z₀⟩,  Q(s, CUT) = ⟨Z₁⟩
```

### Parameter Count

- **Encoding**: 5 angles (fixed from state, not trained)
- **Per layer**: 5 qubits × 2 rotations (RY, RZ) = 10 params
- **Layers**: 2
- **Total trainable**: 20 parameters

### Q-Value Extraction

```python
def get_q_values(circuit, state, params, shots=512):
    """Map measurement expectations to Q-values."""
    # Bind state → encoding angles, params → variational angles
    bound_circuit = circuit.assign_parameters({...})
    
    # Execute with Aer
    result = backend.run(bound_circuit, shots=shots).result()
    counts = result.get_counts()
    
    # Extract Z expectations for qubits 0 and 1
    q_extend = expectation_Z(counts, qubit=0)  # ∈ [-1, 1]
    q_cut = expectation_Z(counts, qubit=1)     # ∈ [-1, 1]
    
    return [q_extend, q_cut]
```

## Training Pipeline

### Loss Function (TD Error)

```python
# Standard DQN temporal difference loss
loss = (r + γ * max_a' Q_target(s', a') - Q(s, a))²

# Gradient: Cannot use backprop through quantum circuit
# Solution: SPSA finite differences
```

### Optimizer (SPSA)

```python
class SPSAOptimizer:
    """Simultaneous Perturbation Stochastic Approximation."""
    
    def step(self, params, loss_fn):
        # Sample random direction
        delta = np.random.choice([-1, 1], size=len(params))
        
        # Finite difference gradient estimate (2 circuit evals)
        loss_plus = loss_fn(params + c * delta)
        loss_minus = loss_fn(params - c * delta)
        
        grad_estimate = (loss_plus - loss_minus) / (2 * c * delta)
        
        # Update
        return params - lr * grad_estimate
```

### Training Loop

```python
for episode in range(num_episodes):
    state = env.reset()
    
    while not done:
        # 1. Get Q-values from VQ-DQN (512 shots)
        q_values = agent.get_q_values(state)
        
        # 2. ε-greedy action selection
        action = epsilon_greedy(q_values, epsilon)
        
        # 3. Environment step (incremental reward)
        next_state, reward, done = env.step(action)
        
        # 4. Store transition
        replay_buffer.push(state, action, reward, next_state, done)
        
        # 5. Sample batch and compute TD loss
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            loss = compute_td_loss(batch)
            
            # 6. SPSA gradient step
            agent.params = spsa.step(agent.params, loss)
        
        state = next_state
    
    # Episode end: Full k-means evaluation (metrics only)
    if episode % eval_interval == 0:
        segments = extract_segments(trajectory)
        od_score = classical_kmeans_objective(segments)
        log_metrics(od_score)
```

## NISQ Constraints

| Constraint | Value | Rationale |
|------------|-------|-----------|
| Qubits | 5 | Matches state dimensionality |
| Depth | 2 layers | Minimize decoherence |
| Parameters | 20 | Below barren plateau threshold |
| Shots (train) | 512 | Balance noise vs speed |
| Shots (eval) | 1024 | Lower variance for metrics |
| Entanglement | Linear chain | No ring; simpler transpilation |

## Classical Components

| Component | Implementation |
|-----------|----------------|
| Replay buffer | Standard circular buffer, capacity 10k |
| Feature extraction | NumPy-based trajectory stats |
| Centroid update | Classical mean (Lloyd's algorithm) |
| Distance computation | Euclidean (NumPy) |
| K-Means clustering | Scikit-learn (episode-end eval only) |

## Installation

```bash
cd q_rlstc
pip install -e ".[dev]"
```

## Quick Start

```bash
# Run demo
python experiments/run_synth_demo.py

# Run tests
pytest tests/ -v
```

## Project Structure

```
q_rlstc/
├── pyproject.toml
├── README.md
├── q_rlstc/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── synthetic.py      # Trajectory generation
│   │   └── features.py       # State vector extraction
│   ├── quantum/
│   │   ├── vqdqn_circuit.py  # VQ-DQN circuit builder
│   │   ├── backends.py
│   │   └── mitigation.py
│   ├── rl/
│   │   ├── replay_buffer.py
│   │   ├── vqdqn_agent.py
│   │   ├── spsa.py
│   │   └── train.py
│   └── clustering/
│       ├── classical_kmeans.py
│       └── metrics.py
├── experiments/
│   └── run_synth_demo.py
└── tests/
    ├── test_angle_encoding.py
    ├── test_hea_depth.py
    ├── test_kmeans_update.py
    └── test_training_smoke.py
```

## MCP Qiskit Tool Inventory

| Tool | Purpose |
|------|---------|
| `analyze_circuit_tool` | Circuit metrics (depth, gates) |
| `transpile_circuit_tool` | Transpilation with optimization |
| `compare_optimization_levels_tool` | Compare O0-O3 transpilation |
| `convert_qasm3_to_qpy_tool` | QASM3 → QPY conversion |
| `convert_qpy_to_qasm3_tool` | QPY → QASM3 conversion |

> **Note**: Circuit execution uses local `qiskit-aer` simulator. MCP provides analysis/transpilation.

## References

1. Liang et al. — "Sub-trajectory clustering with deep reinforcement learning"
2. Chen et al. — "Variational Quantum Circuits for Deep Reinforcement Learning"
3. Schuld et al. — "Evaluating analytic gradients on quantum hardware"

## License

MIT License - Research code for academic use.
