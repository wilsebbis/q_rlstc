# Q-RLSTC Technical Documentation - Part 2

> Deep-dives, implementation details, and debugging guides.

---

## Table of Contents

1. [State Feature Computation Details](#state-feature-computation-details)
2. [Reward Engineering Deep-Dive](#reward-engineering-deep-dive)
3. [SPSA Implementation Details](#spsa-implementation-details)
4. [Replay Buffer Design](#replay-buffer-design)
5. [Backend and Noise Models](#backend-and-noise-models)
6. [Debugging Guide](#debugging-guide)
7. [Extension Points](#extension-points)

---

## State Feature Computation Details

Located in `q_rlstc/data/features.py`.

### Feature 0-1: OD Proxies

These estimate the overall distance (OD) objective under different actions without running full k-means:

```python
def _compute_od_proxy(segment_points, current_od, n_segments):
    """Lightweight proxy for OD after adding segment."""
    # Segment "diameter" = mean distance to centroid
    coords = np.array([[p.x, p.y] for p in segment_points])
    centroid = coords.mean(axis=0)
    segment_cost = np.linalg.norm(coords - centroid, axis=1).mean()
    
    # Running average update
    return (current_od * n_segments + segment_cost) / (n_segments + 1)
```

- **od_segment**: OD if we cut at current point
- **od_continue**: OD if we extend by one more point

### Feature 2: TRACLUS Baseline

MDL-inspired cost measuring how well a straight line approximates the segment:

```python
def _compute_traclus_baseline(points, split_idx):
    """Sum of perpendicular distances to start-end line."""
    start, end = points[0], points[-1]
    line_vec = end - start
    line_unit = line_vec / norm(line_vec)
    
    total_perp_dist = 0.0
    for p in points[1:-1]:
        vec_to_p = p - start
        projection = clip(dot(vec_to_p, line_unit), 0, line_len)
        proj_point = start + projection * line_unit
        perp_dist = norm(p - proj_point)
        total_perp_dist += perp_dist
    
    return log2(line_len) + total_perp_dist
```

### Features 3-4: Position Features

Simple normalized position indicators:

```python
len_backward = (current_idx - split_point + 1) / total_length
len_forward = (total_length - current_idx - 1) / total_length
```

---

## Reward Engineering Deep-Dive

### Why Not Use Raw OD Improvement?

The naive reward `reward = old_od - new_od` has problems:

1. **Delayed signal**: OD only changes meaningfully on CUT
2. **Sparse rewards**: EXTEND actions get near-zero reward
3. **Not Markov-safe**: Depends on global clustering state

### Current Reward Design

```
reward = boundary_sharpness_bonus + variance_delta - segment_penalty
```

| Component | Range | When Applied |
|-----------|-------|--------------|
| `boundary_sharpness * 0.5` | [0, 0.5] | On CUT only |
| `variance_delta * 0.1` | Variable | Every step |
| `-SEGMENT_PENALTY` | -0.1 | On CUT only |

### Boundary Sharpness Calculation

Measures direction change at cut point (sharper turn = better boundary):

```python
def _compute_boundary_sharpness(boundary_idx):
    v1 = points[boundary_idx] - points[boundary_idx - 1]  # Before
    v2 = points[boundary_idx + 1] - points[boundary_idx]  # After
    
    cos_angle = dot(v1, v2) / (norm(v1) * norm(v2))
    angle = arccos(clip(cos_angle, -1, 1))  # [0, π]
    
    return angle / π  # Normalized to [0, 1]
```

### Variance Delta

Provides per-step feedback for EXTEND actions:

```python
old_variance = var(segment[:current_idx])
new_variance = var(segment[:current_idx + 1])
variance_delta = old_variance - new_variance  # Positive if variance decreased
```

---

## SPSA Implementation Details

Located in `q_rlstc/rl/spsa.py`.

### Algorithm

SPSA approximates gradients using only 2 function evaluations:

```
g_k ≈ (L(θ + c_k Δ) - L(θ - c_k Δ)) / (2 c_k Δ)
θ_{k+1} = θ_k - a_k g_k
```

Where:
- `Δ` is a random perturbation vector with ±1 entries
- `a_k = a / (k + A + 1)^α` is the step size
- `c_k = c / (k + 1)^γ` is the perturbation size

### Hyperparameters

```python
class SPSAOptimizer:
    def __init__(
        self,
        a: float = 0.1,        # Initial step size
        c: float = 0.1,        # Initial perturbation size
        A: float = 10.0,       # Step size offset
        alpha: float = 0.602,  # Step size decay rate
        gamma: float = 0.101,  # Perturbation decay rate
    ):
```

### Why These Defaults?

- **α = 0.602, γ = 0.101**: Standard SPSA theory values
- **A = 10**: Stabilizes early training
- **a, c = 0.1**: Conservative for noisy quantum objectives

### Shot Noise Interaction

SPSA is robust to shot noise because:
1. It only needs function values, not exact gradients
2. Random perturbations average out noise over iterations
3. Decaying perturbation size reduces noise impact over time

---

## Replay Buffer Design

Located in `q_rlstc/rl/replay_buffer.py`.

### Structure

```python
@dataclass
class Transition:
    state: np.ndarray      # 5-dim
    action: int            # 0 or 1
    reward: float
    next_state: np.ndarray
    done: bool

class ReplayBuffer:
    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
```

### Sampling Strategy

Uniform random sampling (no prioritized replay):

```python
def sample_batch(self, batch_size: int):
    indices = np.random.choice(len(self.buffer), batch_size, replace=False)
    batch = [self.buffer[i] for i in indices]
    
    states = np.array([t.state for t in batch])
    actions = np.array([t.action for t in batch])
    rewards = np.array([t.reward for t in batch])
    next_states = np.array([t.next_state for t in batch])
    dones = np.array([t.done for t in batch])
    
    return states, actions, rewards, next_states, dones
```

### Warm-up Period

Training only starts when buffer has enough samples:

```python
def is_ready(self, min_size: int) -> bool:
    return len(self.buffer) >= min_size
```

---

## Backend and Noise Models

Located in `q_rlstc/quantum/backends.py`.

### Backend Factory

```python
def get_backend(mode: str, noise_model_name: str = None):
    if mode == "ideal":
        return AerSimulator()
    elif mode == "noisy_sim":
        noise_model = get_noise_model(noise_model_name)
        return AerSimulator(noise_model=noise_model)
```

### Available Noise Models

| Name | Description |
|------|-------------|
| `depolarizing` | Uniform depolarizing on all gates |
| `thermal` | Thermal relaxation (T1, T2) |
| `ibm_fake` | Fake backend mimicking IBM hardware |

### Readout Error Mitigation

In `q_rlstc/quantum/mitigation.py`:

```python
class ReadoutMitigator:
    """Correct measurement errors using calibration matrix."""
    
    def calibrate(self, backend, shots=8192):
        """Build calibration matrix by measuring |0...0⟩ and |1...1⟩."""
        # Run calibration circuits
        # Build M matrix where M[i,j] = P(measure i | prepared j)
        self.calibration_matrix = M
    
    def apply(self, counts):
        """Apply inverse calibration to correct raw counts."""
        # Solve linear system to get corrected probabilities
        return corrected_counts
```

---

## Debugging Guide

### Common Issues

#### 1. All Q-values are the same

**Symptom**: `Q_extend ≈ Q_cut` for all states

**Causes**:
- Circuit not expressive enough
- SPSA learning rate too low
- Too few shots (high variance masks signal)

**Fixes**:
- Increase `n_layers` to 3
- Increase `a` parameter in SPSA
- Increase `shots` to 1024

#### 2. Agent always chooses same action

**Symptom**: 100% EXTEND or 100% CUT

**Causes**:
- Epsilon not decaying properly
- Reward imbalance (segment penalty too high/low)
- Initial Q-values biased

**Fixes**:
- Check `epsilon_decay` < 1.0
- Tune `SEGMENT_PENALTY` (try 0.05 or 0.2)
- Initialize output_scale and output_bias

#### 3. Training loss doesn't decrease

**Symptom**: TD loss flat or oscillating

**Causes**:
- Target network not updating
- SPSA perturbation too large
- Reward scale mismatch

**Fixes**:
- Check `target_update_freq` is being hit
- Reduce `c` parameter in SPSA
- Normalize rewards to [-1, 1] range

#### 4. Circuit execution is slow

**Symptom**: >1 second per Q-value evaluation

**Causes**:
- Too many shots
- No caching of transpiled circuits
- Full-state simulation overhead

**Fixes**:
- Reduce shots to 256-512 for training
- Pre-transpile circuit templates
- Use matrix_product_state method for deeper circuits

### Diagnostic Functions

```python
# Check circuit structure
info = agent.get_circuit_info()
print(f"Depth: {info.depth}, Params: {info.n_params}")
print(f"Gates: {info.gate_counts}")

# Check Q-value distribution
states = [env.reset() for _ in range(10)]
q_values = [agent.get_q_values(s) for s in states]
print(f"Q-range: {np.min(q_values)} to {np.max(q_values)}")

# Check replay buffer
print(f"Buffer size: {len(trainer.buffer)}")
print(f"Recent rewards: {[t.reward for t in list(trainer.buffer)[-10:]]}")
```

---

## Extension Points

### Adding New State Features

1. Modify `StateFeatureExtractor.extract_features()` in `features.py`
2. Update `n_qubits` in config to match new dimension
3. Rebuild circuit with new qubit count

### Custom Reward Functions

1. Modify `MDPEnvironment.step()` in `train.py`
2. Access trajectory geometry via `self.trajectory.points`
3. Keep reward bounded and incremental

### Alternative Entanglement Patterns

In `VQDQNCircuitBuilder.__init__()`:

```python
entanglement: str = 'linear'  # Options: 'linear', 'circular', 'full'
```

To add new patterns, modify `_add_variational_layer()`:

```python
elif self.entanglement == 'custom':
    # Your custom CNOT pattern
    circuit.cx(qr[0], qr[2])
    circuit.cx(qr[1], qr[3])
    # ...
```

### Different Optimizers

Replace SPSA with Adam-SPSA or quantum natural gradient:

```python
# In vqdqn_agent.py
from .adam_spsa import AdamSPSAOptimizer

self.optimizer = AdamSPSAOptimizer(
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
)
```

---

## Appendix: Configuration Dataclasses

Located in `q_rlstc/config.py`:

```python
@dataclass
class VQDQNConfig:
    n_qubits: int = 5
    n_layers: int = 2
    shots_train: int = 512
    shots_eval: int = 1024

@dataclass
class RLConfig:
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay: float = 0.99
    batch_size: int = 32
    memory_size: int = 10000
    target_update_freq: int = 10

@dataclass
class ClusteringConfig:
    n_clusters: int = 10
    max_iter: int = 50

@dataclass
class NoiseConfig:
    use_noise: bool = False
    noise_model: str = "depolarizing"

@dataclass
class TrainingConfig:
    n_epochs: int = 10

@dataclass
class QRLSTCConfig:
    vqdqn: VQDQNConfig = field(default_factory=VQDQNConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
```

---

*End of Technical Documentation*
