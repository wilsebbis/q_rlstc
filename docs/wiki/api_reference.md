# API Reference

[← Back to README](../../README.md) · [Debugging](debugging.md) · **API Reference**

---

## Configuration — [`config.py`](../../q_rlstc/config.py)

### `VQDQNConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `n_qubits` | `int` | 5 | Qubits in VQ-DQN (5 for A/D, 8 for B, 6 for C) |
| `n_layers` | `int` | 2 | Variational layers (3 for Version D) |
| `n_actions` | `int` | 2 | Action space size (2 for A/B/D, 3 for C) |
| `readout_mode` | `str` | `"standard"` | `"standard"` (A/D), `"multi_observable"` (B), `"softmax"` (C) |

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
| `a` | `float` | 0.12 | Initial step size |
| `c` | `float` | 0.10 | Initial perturbation size |
| `A` | `int` | 20 | Step size offset |
| `alpha` | `float` | 0.602 | Step size decay exponent |
| `gamma` | `float` | 0.101 | Perturbation decay exponent |
| `momentum` | `float` | 0.0 | EMA momentum (0.9 for m-SPSA in Version C) |

### `RLConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `gamma` | `float` | 0.90 | Discount factor |
| `epsilon_start` | `float` | 1.0 | Initial exploration rate |
| `epsilon_min` | `float` | 0.1 | Minimum exploration rate |
| `epsilon_decay` | `float` | 0.99 | Per-episode decay |
| `batch_size` | `int` | 32 | Replay sampling batch size |
| `memory_size` | `int` | 5000 | Replay buffer capacity |
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
    version: str = "A"                    # "A", "B", "C", or "D"
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
| `step` | `(action: int) → (state, reward, done)` | Execute EXTEND, CUT, DROP, or SKIP |
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

### `StateFeatureExtractorClassicalExact` (Version D)

| Method | Signature | Description |
|---|---|---|
| `compute` | `(trajectory, current_idx, segments, cluster_state) → ndarray[5]` | 5D VLDB-exact: `[OD_s, OD_n, OD_b, L_b, L_f]` |

---

## Trajectory Distance — [`clustering/trajdistance.py`](../../q_rlstc/clustering/trajdistance.py)

### Core IED Functions

| Function | Signature | Description |
|---|---|---|
| `traj2traj_ied` | `(pts1: List[Point], pts2: List[Point]) → float` | Full IED between two trajectories |
| `incremental_ied` | `(traj1, traj2, k_dict, k, i, sp_i) → dict` | Incremental IED update (O(1) per step) |
| `incremental_mindist` | `(traj_pts, start, curr, k_dict, cluster_dict) → (dist, id)` | Nearest cluster via incremental IED |
| `line2line_ied` | `(p1s, p1e, p2s, p2e) → float` | Segment-pair distance |
| `get_static_ied` | `(points, x, y, t1, t2) → float` | Static point-to-trajectory IED |
| `timed_traj` | `(points, ts, te) → Optional[Trajectory]` | Time-windowed sub-trajectory extraction |

### MDL Cost

| Function | Signature | Description |
|---|---|---|
| `traj_mdl_comp` | `(points, start_index, curr_index, mode) → float` | MDL cost ("simp" or "orign" mode) |

### Distance Classes

| Class | Method | Description |
|---|---|---|
| `FrechetDistance` | `compute(traj_c, traj_q) → float` | Discrete Fréchet distance |
| `DtwDistance` | `compute(traj_c, traj_q) → float` | Dynamic Time Warping distance |

---

## Pickle Data Loader — [`clustering/pickle_loader.py`](../../q_rlstc/clustering/pickle_loader.py)

| Function | Signature | Description |
|---|---|---|
| `load_trajectories` | `(path, limit=None) → List[Trajectory]` | Load pre-processed trajectories |
| `load_raw_trajectories` | `(path, limit=None) → list` | Load as raw RLSTCcode Traj objects |
| `load_cluster_centers` | `(path) → (Dict, float)` | Load cluster centers (Q-RLSTC format) |
| `load_cluster_centers_raw` | `(path) → (Dict, float)` | Load in MDP.py's native dict format |
| `load_subtrajectories` | `(path) → List[Trajectory]` | Load TRACLUS sub-trajectories |
| `load_test_set` | `(path) → List[Trajectory]` | Load held-out test/validation sets |
| `list_available_datasets` | `() → Dict[str, List[str]]` | List available pickle files in data dir |

---

## MDL Preprocessing — [`data/preprocessing.py`](../../q_rlstc/data/preprocessing.py)

| Function | Signature | Description |
|---|---|---|
| `simplify_trajectory` | `(trajectory: Trajectory) → Trajectory` | Greedy MDL-based simplification |
| `simplify_all` | `(trajectories) → List[Trajectory]` | Simplify all trajectories |
| `preprocess_tdrive` | `(raw, max_len, min_len, simplify) → List[Trajectory]` | Full pipeline |
| `filter_by_coordinates` | `(trajs, lon_range, lat_range) → list` | Geographic bounding box filter |
| `normalize_locations` | `(trajs) → list` | Z-score normalize spatial coords |
| `normalize_time` | `(trajs) → list` | Z-score normalize timestamps |
| `arrays_to_trajectories` | `(data) → List[Trajectory]` | Convert [lon,lat,time] → Trajectory |

---

## Clustering — [`clustering/`](../../q_rlstc/clustering/)

### `ClassicalKMeans` — [`classical_kmeans.py`](../../q_rlstc/clustering/classical_kmeans.py)

| Method | Signature | Description |
|---|---|---|
| `fit` | `(data: ndarray) → KMeansResult` | Run k-means++ |
| `predict` | `(data: ndarray) → ndarray` | Assign clusters |

### Incremental Cluster Management — [`classical_kmeans.py`](../../q_rlstc/clustering/classical_kmeans.py)

| Function | Signature | Description |
|---|---|---|
| `add_to_cluster` | `(cluster_dict, id, sub_traj, dist)` | Add sub-trajectory to cluster |
| `compute_center` | `(cluster_dict, id) → List[Point]` | Recompute center from time-indexed points |
| `update_all_centers` | `(cluster_dict)` | Update all centers, reset accumulators |
| `compute_overdist` | `(cluster_dict) → float` | Compute overall distance |
| `initialize_cluster_dict` | `(n_clusters, centers) → Dict` | Create empty cluster dict |

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

---

## Experiments — [`experiments/`](../../experiments/)

### `run_cross_comparison.py`

```python
# Run both classical and quantum arms on same data
python experiments/run_cross_comparison.py \
    --traj-path ../RLSTCcode/data/Tdrive_norm_traj \
    --centers-path ../RLSTCcode/data/tdrive_clustercenter \
    --amount 500 --run both
```

### `data_bridge.py`

| Function | Description |
|---|---|
| `convert_rlstc_to_qrlstc(traj_path, centers_path, amount)` | Convert RLSTCcode pickle → Q-RLSTC Trajectory objects |
