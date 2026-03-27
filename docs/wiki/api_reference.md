# API Reference


---

## Agent Configuration — `rl/vqdqn_agent.py`

### `AgentConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `version` | `str` | `"A"` | `"A"` (5q standard), `"B"` (8q multi-observable), `"D"` (5q standard), `"E"` (5q + input scaling + anti-BP) |
| `n_qubits` | `int` | 5 | Qubits in VQ-DQN (auto-set to 8 for Version B) |
| `n_layers` | `int` | 2 | Variational layers |
| `gamma` | `float` | 0.90 | Discount factor |
| `epsilon_start` | `float` | 1.0 | Initial exploration rate |
| `epsilon_min` | `float` | 0.1 | Minimum exploration rate |
| `epsilon_decay` | `float` | 0.99 | Per-episode decay |
| `shots` | `int` | 512 | Measurement shots |
| `use_double_dqn` | `bool` | `True` | Enable Double DQN |
| `target_update_freq` | `int` | 10 | Episodes between target sync |
| `entanglement` | `str` | `"linear"` | `"linear"`, `"circular"`, `"full"`, `"none"` |
| `exploration_mode` | `str` | `"epsilon_greedy"` | `"epsilon_greedy"` or `"boltzmann"` |
| `q_clip_range` | `float` | 50.0 | Symmetric Q-value clipping bound |
| `optimistic_cut_bias` | `float` | 0.0 | Extra initial bias for CUT action |
| `use_input_scaling` | `bool` | `False` | Learnable per-feature scale+shift |
| `anti_barren_plateau` | `bool` | `False` | Near-zero circuit param init |
| `use_soft_targets` | `bool` | `False` | Entropy-regularized targets (soft-DQN) |
| `soft_alpha` | `float` | 0.1 | Entropy temperature for soft targets |

### `ClassicalAgentConfig` — `rl/spsa_classical_agent.py`

| Field | Type | Default | Description |
|---|---|---|---|
| `hidden_sizes` | `List[int]` | `[64]` | Hidden layer sizes (empty = linear) |
| `feature_transform` | `str` | `"none"` | `"none"` (standard MLP) or `"rbf"` (Random Fourier Features) |
| `rbf_dim` | `int` | 10 | Number of RBF random features |
| `gamma` | `float` | 0.90 | Discount factor |
| `epsilon_start` | `float` | 1.0 | Initial exploration rate |
| `epsilon_min` | `float` | 0.1 | Minimum exploration rate |
| `epsilon_decay` | `float` | 0.99 | Per-episode decay |
| `use_double_dqn` | `bool` | `True` | Enable Double DQN |
| `target_update_freq` | `int` | 10 | Episodes between target sync |

### `AdamAgentConfig` — `rl/adam_classical_agent.py`

| Field | Type | Default | Description |
|---|---|---|---|
| `hidden_sizes` | `List[int]` | `[64]` | Hidden layer sizes |
| `gamma` | `float` | 0.90 | Discount factor |
| `lr` | `float` | 1e-3 | Learning rate |
| `beta1` | `float` | 0.9 | Adam β₁ |
| `beta2` | `float` | 0.999 | Adam β₂ |
| `max_grad_norm` | `float` | 10.0 | Gradient clipping norm |

---

## VQ-DQN Agent — `rl/vqdqn_agent.py`

### `VQDQNAgent`

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(config: AgentConfig, backend: AerSimulator, seed: int)` | Initialise circuit, params, target params |
| `get_q_values` | `(state, use_target=False) → ndarray[2]` | Run circuit, return Q-values |
| `act` | `(state, greedy=False) → int` | ε-greedy or Boltzmann action selection |
| `update` | `(states, actions, rewards, next_states, dones) → float` | Batched SPSA gradient step on Huber TD loss |
| `compute_targets_batch` | `(rewards, next_states, dones) → ndarray` | Compute TD targets for batch |
| `update_target_network` | `()` | Copy online params → target params |
| `decay_epsilon` | `()` | Decay ε and Boltzmann temp; periodic target sync |
| `get_circuit_info` | `() → CircuitInfo` | Circuit structure summary |
| `save_checkpoint` | `(path: str)` | Save agent state to `.npz` file |
| `load_checkpoint` | `(path: str)` | Load agent state from `.npz` file |

---

## Quantum Circuit — `quantum/vqdqn_circuit.py`

### `VQDQNCircuitBuilder`

| Method | Signature | Returns |
|---|---|---|
| `__init__` | `(n_qubits, n_layers, use_data_reuploading, entanglement)` | — |
| `build_circuit` | `(state, params, add_measurements) → QuantumCircuit` | Parameterised circuit |
| `get_circuit_info` | `(params) → CircuitInfo` | Circuit metrics |

### Module-Level Functions

| Function | Signature | Description |
|---|---|---|
| `angle_encode` | `(features, scaling='arctan') → ndarray` | Encode features as rotation angles |
| `build_vqdqn_circuit` | `(state, params, n_qubits, n_layers, use_data_reuploading, add_measurements) → QuantumCircuit` | Convenience wrapper |
| `evaluate_q_values` | `(state, params, backend, shots, ...) → ndarray[2]` | Run circuit and return Q-values |
| `q_values_batch` | `(states, params, n_qubits, n_layers, ...) → ndarray[B,2]` | Fast batched Q-value computation (pure numpy) |
| `compute_expectation_from_counts` | `(counts, shots, qubit_idx, n_qubits) → float` | `⟨Zᵢ⟩ ∈ [-1, 1]` |
| `compute_parity_expectation` | `(counts, shots, qubit_a, qubit_b, n_qubits) → float` | `⟨ZₐZ_b⟩ ∈ [-1, 1]` |

---

## SPSA Optimizer — `rl/spsa.py`

### `SPSAConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `A` | `int` | 20 | Stability constant |
| `a` | `float` | 0.12 | Initial learning rate scale |
| `c` | `float` | 0.08 | Initial perturbation magnitude |
| `alpha` | `float` | 0.602 | Learning rate decay exponent |
| `gamma` | `float` | 0.101 | Perturbation decay exponent |
| `max_iter` | `int` | 100 | Maximum iterations |
| `seed` | `int` | 42 | Random seed |
| `use_momentum` | `bool` | `True` | Enable momentum-SPSA |
| `momentum` | `float` | 0.9 | Momentum coefficient (β) |

### `SPSAOptimizer`

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(A, a, c, alpha, gamma, max_grad_norm, seed, use_momentum, momentum, n_perturbations, use_crn, crn_base_seed, param_scales)` | Initialise decay schedules |
| `step` | `(loss_fn, params) → (params, grad_norm)` | One SPSA update step |
| `compute_gradient` | `(loss_fn, params) → ndarray` | Estimate gradient (optionally momentum-averaged) |
| `optimize` | `(loss_fn, initial_params, max_iter, tolerance, callback) → (params, loss)` | Full optimization loop |
| `reset` | `()` | Reset iteration counter and momentum buffer |

---

## Replay Buffer — `rl/replay_buffer.py`

### `Experience`

```python
class Experience(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
```

### `ReplayBuffer`

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(max_size: int = 5000, seed: int = 42)` | Initialise circular buffer |
| `add` | `(state, action, reward, next_state, done)` | Add experience |
| `sample` | `(batch_size: int) → list[Experience]` | Uniform random sample |
| `sample_batch` | `(batch_size: int) → tuple[ndarray×5]` | Sample as numpy arrays (states, actions, rewards, next_states, dones) |
| `sample_batch_stratified` | `(batch_size, min_cut_quota=0.3) → tuple[ndarray×5]` | Sample with minimum CUT action quota |
| `is_ready` | `(min_size: int) → bool` | `len(buffer) >= min_size` |
| `clear` | `()` | Clear all experiences |
| `__len__` | `() → int` | Current buffer size |

---

## Trajectory Distance — `clustering/trajdistance.py`

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

## Pickle Data Loader — `clustering/pickle_loader.py`

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

## Preprocessing — `data/preprocessing.py`

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

## Clustering — `clustering/`

### `ClassicalKMeans` — `classical_kmeans.py`

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(n_clusters, max_iter, convergence_threshold, seed)` | Initialize k-means |
| `fit` | `(data: ndarray) → KMeansResult` | Run k-means++ |
| `predict` | `(data: ndarray) → ndarray` | Assign clusters |

### Metrics — `metrics.py`

| Function | Signature | Description |
|---|---|---|
| `overall_distance` | `(data, centroids, labels) → float` | Root-mean-square distance |
| `silhouette_score` | `(data, labels) → float` | Cluster quality ∈ [-1, 1] |
| `segmentation_f1` | `(predicted, true, tolerance) → (precision, recall, f1)` | Boundary detection F1 (returns tuple) |
| `incremental_od_update` | `(current_od, n_segments, new_segment_cost) → float` | Efficient reward-time OD update |
| `od_improvement_reward` | `(od_before, od_after, scale) → float` | Reward from OD improvement |
| `weighted_valcr` | `(per_segment_ods, per_segment_lengths, basesim, epsilon) → float` | Length-weighted ValCR |
| `random_policy_advantage` | `(agent_valcr, random_valcr) → float` | Δ_rand advantage metric |

### Random Frontier — `clustering/random_frontier.py`

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(fold_basesim, epsilon)` | Initialize (call `finalize()` to build) |
| `add_point` | `(cut_pct, val_cr)` | Add raw observation |
| `finalize` | `(n_bins, smoothing)` | Bin raw points into frontier curve |
| `interpolate` | `(cut_pct) → float` | Get frontier ValCR at CUT budget |
| `advantage` | `(agent_valcr, agent_cut_pct) → float` | Budget-matched Δ_rand |

---

## Backends — `quantum/backends.py`

### `BackendFactory`

| Method | Signature | Description |
|---|---|---|
| `get_ideal_backend` | `() → AerSimulator` | Noiseless statevector backend |
| `get_simple_noise_model` | `(single_qubit_error, two_qubit_error, readout_error) → NoiseModel` | Depolarizing noise |
| `get_thermal_noise_model` | `(t1, t2, gate_time_1q, gate_time_2q, ...) → NoiseModel` | Thermal + depolarizing noise |
| `get_ibm_eagle_noise_model` | `() → NoiseModel` | IBM Eagle r3 approximation |
| `get_ibm_heron_noise_model` | `() → NoiseModel` | IBM Heron approximation |
| `get_noisy_backend` | `(noise_model) → AerSimulator` | Noisy simulator |
| `get_noise_model_by_name` | `(name) → Optional[NoiseModel]` | Named lookup: `"ideal"`, `"simple"`, `"thermal"`, `"eagle"`, `"heron"` |

### Module-Level Function

| Function | Signature | Description |
|---|---|---|
| `get_backend` | `(mode, noise_model_name, device_name) → AerSimulator` | Backend factory: `"ideal"`, `"noisy_sim"`, `"ibm_runtime"` |

---

## Supporting Modules

### Adaptive Shots — `rl/adaptive_shots.py`

| Method | Signature | Description |
|---|---|---|
| `get_shots` | `(q_margin: float) → int` | Determine shot count from Q-value margin |
| `get_stats` | `() → dict` | Shot allocation statistics |
| `reset` | `()` | Clear history for new episode |

### DROP Action — `rl/drop_action.py`

| Property/Method | Signature | Description |
|---|---|---|
| `n_actions` | `→ int` | 3 when enabled, 2 when disabled |
| `is_drop_allowed` | `(consecutive_drops) → bool` | Check if DROP is allowed |
| `get_drop_penalty` | `(consecutive_drops) → float` | Escalating penalty |
| `check_retention` | `(n_total, n_dropped) → bool` | Validate retention constraint |

### Soft Targets — `rl/soft_targets.py`

| Function | Signature | Description |
|---|---|---|
| `soft_value` | `(q_values, alpha=0.1) → ndarray` | Entropy-regularized soft value V(s) = α·log(Σ exp(Q/α)) |
| `soft_policy` | `(q_values, alpha=0.1) → ndarray` | Boltzmann policy π(a|s) = softmax(Q/α) |

### Statistics — `utils/stats.py`

| Function | Signature | Description |
|---|---|---|
| `bootstrap_ci` | `(data, n_bootstrap, ci, seed) → (mean, ci_low, ci_high)` | Bootstrap confidence interval |
| `paired_bootstrap_test` | `(a, b, n_bootstrap, seed) → float` | Two-sided paired significance p-value |

---

## Experiments — [`experiments/`](../../experiments/)

### `run_thesis_experiments.py`

```python
# Run all thesis experiments
python experiments/run_thesis_experiments.py

# Specific experiments
python experiments/run_thesis_experiments.py --experiments D1,E1 --amount 100 --epochs 3
```

### `run_cross_comparison.py`

```python
# Run 4-agent comparison on same data
python experiments/run_cross_comparison.py --amount 500 --run all
```
