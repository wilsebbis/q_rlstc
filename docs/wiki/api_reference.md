# API Reference

[← Back to README](../../README_2.md) · [Debugging](debugging.md) · **API Reference**

---

## Configuration — [`config.py`](../../q_rlstc/config.py)

### `VQDQNConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `n_qubits` | `int` | 5 | Qubits in VQ-DQN (5 for A, 8 for B) |
| `n_layers` | `int` | 2 | Variational layers |
| `n_actions` | `int` | 2 | Action space size (EXTEND, CUT) |
| `readout_mode` | `str` | `"standard"` | `"standard"` or `"multi_observable"` |

### `NoiseConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `use_noise` | `bool` | `False` | Enable noise simulation |
| `noise_model` | `str` | `"depolarizing"` | `"depolarizing"`, `"thermal"`, `"ibm_fake"` |
| `use_mitigation` | `bool` | `True` | Enable readout error mitigation |
| `calibration_shots` | `int` | 8192 | Shots for calibration circuits |

### `SPSAConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `a` | `float` | 0.1 | Initial step size |
| `c` | `float` | 0.1 | Initial perturbation size |
| `A` | `float` | 10.0 | Step size offset |
| `alpha` | `float` | 0.602 | Step size decay exponent |
| `gamma` | `float` | 0.101 | Perturbation decay exponent |

### `RLConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `gamma` | `float` | 0.99 | Discount factor |
| `epsilon_start` | `float` | 1.0 | Initial exploration rate |
| `epsilon_min` | `float` | 0.1 | Minimum exploration rate |
| `epsilon_decay` | `float` | 0.99 | Per-episode decay |
| `batch_size` | `int` | 32 | Replay sampling batch size |
| `memory_size` | `int` | 10000 | Replay buffer capacity |
| `target_update_freq` | `int` | 10 | Episodes between target sync |

### `ClusteringConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `k` | `int` | 5 | Number of clusters |
| `max_iter` | `int` | 100 | K-means max iterations |
| `init_method` | `str` | `"kmeans++"` | Initialisation method |

### `QRLSTCConfig`

Top-level config that composes all sub-configs:

```python
@dataclass
class QRLSTCConfig:
    version: str = "A"                    # "A" or "B"
    vqdqn: VQDQNConfig = field(default_factory=VQDQNConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    spsa: SPSAConfig = field(default_factory=SPSAConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
```

---

## Quantum Circuit — [`quantum/vqdqn_circuit.py`](../../q_rlstc/quantum/vqdqn_circuit.py)

### `VQDQNCircuitBuilder`

| Method | Signature | Returns |
|---|---|---|
| `__init__` | `(n_qubits, n_layers, readout_mode)` | — |
| `build` | `(state: ndarray, params: ndarray) → QuantumCircuit` | Parameterised circuit |
| `compute_expectation_from_counts` | `(counts, shots, qubit_idx, n_qubits) → float` | `⟨Zᵢ⟩ ∈ [-1, 1]` |
| `compute_parity_from_counts` | `(counts, shots, q_a, q_b) → float` | `⟨ZₐZᵦ⟩ ∈ [-1, 1]` |

---

## Agent — [`rl/vqdqn_agent.py`](../../q_rlstc/rl/vqdqn_agent.py)

### `VQDQNAgent`

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(config: QRLSTCConfig, backend)` | Initialise circuit, params, target params |
| `get_q_values` | `(state, use_target=False) → ndarray[2]` | Run circuit, return Q-values |
| `select_action` | `(state, epsilon) → int` | ε-greedy action selection |
| `update` | `(batch: list[Experience])` | SPSA gradient step on Huber TD loss |
| `update_target` | `()` | Copy online params → target params |
| `decay_epsilon` | `()` | `epsilon *= epsilon_decay` |

---

## Training — [`rl/train.py`](../../q_rlstc/rl/train.py)

### `MDPEnvironment`

| Method | Signature | Description |
|---|---|---|
| `reset` | `(trajectory: Trajectory) → ndarray` | Reset environment for new trajectory |
| `step` | `(action: int) → (state, reward, done)` | Execute EXTEND or CUT |
| `get_segments` | `() → list[list[Point]]` | Current segmentation |

### `train`

```python
def train(config: QRLSTCConfig, dataset, n_epochs: int) → TrainingResult
```

Top-level training function. Returns `TrainingResult` with history, final metrics, and model parameters.

---

## SPSA Optimizer — [`rl/spsa.py`](../../q_rlstc/rl/spsa.py)

### `SPSAOptimizer`

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(config: SPSAConfig)` | Initialise decay schedules |
| `step` | `(params, loss_fn) → params` | One SPSA update step |
| `get_learning_rate` | `(k: int) → float` | `a / (k + A + 1)^α` |
| `get_perturbation` | `(k: int) → float` | `c / (k + 1)^γ` |

---

## Replay Buffer — [`rl/replay_buffer.py`](../../q_rlstc/rl/replay_buffer.py)

### `Experience`

```python
@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
```

### `ReplayBuffer`

| Method | Signature | Description |
|---|---|---|
| `push` | `(state, action, reward, next_state, done)` | Add experience |
| `sample` | `(batch_size: int) → list[Experience]` | Uniform random sample |
| `is_ready` | `(batch_size: int) → bool` | `len(buffer) >= batch_size` |
| `__len__` | `() → int` | Current buffer size |

---

## Features — [`data/features.py`](../../q_rlstc/data/features.py)

### `StateFeatureExtractor` (Version A)

| Method | Signature | Description |
|---|---|---|
| `compute` | `(trajectory, current_idx, segments, cluster_state) → ndarray[5]` | 5D state vector |

### `StateFeatureExtractorB` (Version B)

| Method | Signature | Description |
|---|---|---|
| `compute` | `(trajectory, current_idx, segments, cluster_state) → ndarray[8]` | 8D state vector |

---

## Clustering — [`clustering/`](../../q_rlstc/clustering/)

### `ClassicalKMeans` — [`classical_kmeans.py`](../../q_rlstc/clustering/classical_kmeans.py)

| Method | Signature | Description |
|---|---|---|
| `fit` | `(data: ndarray) → KMeansResult` | Run k-means++ |
| `predict` | `(data: ndarray) → ndarray` | Assign clusters |

### Metrics — [`metrics.py`](../../q_rlstc/clustering/metrics.py)

| Function | Signature | Description |
|---|---|---|
| `overall_distance` | `(segments, centroids) → float` | Sum of segment-to-centroid distances |
| `silhouette_score` | `(segments, labels) → float` | Cluster quality ∈ [-1, 1] |
| `segmentation_f1` | `(predicted, ground_truth, tolerance) → float` | Boundary detection F1 |

---

## Backends — [`quantum/backends.py`](../../q_rlstc/quantum/backends.py)

| Function | Signature | Description |
|---|---|---|
| `get_backend` | `(mode, noise_model_name) → AerSimulator` | Backend factory |
| `get_noise_model` | `(name: str) → NoiseModel` | Named noise profile |

## Mitigation — [`quantum/mitigation.py`](../../q_rlstc/quantum/mitigation.py)

| Method | Signature | Description |
|---|---|---|
| `calibrate` | `(backend, shots) → None` | Build calibration matrix |
| `apply` | `(counts: dict) → dict` | Correct readout errors |

---

## Data — [`data/synthetic.py`](../../q_rlstc/data/synthetic.py)

### `SyntheticDataset`

| Method | Signature | Description |
|---|---|---|
| `generate` | `(n_trajectories, n_points, n_boundaries) → list[Trajectory]` | Generate with ground truth |

### `TrajectoryGenerator`

| Method | Signature | Description |
|---|---|---|
| `generate_trajectory` | `(behavior_sequence) → Trajectory` | Single trajectory from behaviours |

### Data Classes

```python
@dataclass
class Point:
    x: float; y: float; t: float
    def distance(self, other: Point) -> float
    def to_array(self) -> np.ndarray

@dataclass
class Trajectory:
    points: list[Point]
    boundaries: list[int]      # Ground truth boundary indices
    labels: list[str]          # Behaviour type per segment
```
