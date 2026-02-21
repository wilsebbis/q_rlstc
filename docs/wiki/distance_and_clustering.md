# Distance Estimation & Clustering

[← Back to README](../../README.md) · [Training Pipeline](training_pipeline.md) · **Distance & Clustering** · [Justifications →](justifications.md)

---

## The Distance Problem

Trajectory clustering requires comparing trajectories to cluster centroids repeatedly. Each training episode performs hundreds of distance computations. Q-RLSTC addresses this via incremental classical distance with an optional quantum swap test for research validation.

## Integrated Euclidean Distance (IED)

IED is the primary metric, inherited from classical RLSTC and now fully ported to Q-RLSTC in [`trajdistance.py`](../../q_rlstc/clustering/trajdistance.py).

### Geometric Interpretation

IED computes the _area_ between two trajectories over their temporal overlap — the integral of spatial distance across a common time interval.

```
For aligned segments with time interval [tₛ, tₑ]:
    d₁ = distance between starting points
    d₂ = distance between ending points
    IED ≈ 0.5 × (d₁ + d₂) × (tₑ − tₛ)     (trapezoidal approximation)
```

### Why IED?

| Metric | Pros | Cons | Use Case | Q-RLSTC Module |
|---|---|---|---|---|
| **IED** | Fast incremental updates, handles different lengths | Requires temporal overlap | Real-time trajectory clustering | `traj2traj_ied()` |
| Fréchet | Shape-invariant, robust | O(n²) computation | Offline analysis | `FrechetDistance.compute()` |
| DTW | Handles speed variation | Expensive, not incremental | Time series alignment | `DtwDistance.compute()` |
| Euclidean | Simple, fast | Requires same length | Fixed-size vectors only | `np.linalg.norm` |

## IED Functions — `trajdistance.py`

The full IED computation is ported from RLSTCcode with the following key functions:

| Function | Purpose |
|---|---|
| `traj2traj_ied(pts1, pts2)` | Full IED between two trajectory point sequences |
| `incremental_ied(traj1, traj2, k_dict, k, i, sp_i)` | Incremental IED update (O(1) per step) |
| `incremental_mindist(traj_pts, start, curr, k_dict, cluster_dict)` | Find nearest cluster using incremental IED |
| `line2line_ied(p1s, p1e, p2s, p2e)` | Segment-pair distance building block |
| `get_static_ied(points, x, y, t1, t2)` | Static point-to-trajectory IED |
| `timed_traj(points, ts, te)` | Extract time-windowed sub-trajectory |
| `traj_mdl_comp(points, start, curr, mode)` | MDL cost computation (for preprocessing) |

### Incremental Distance (from RLSTC)

The key optimisation: when extending a segment by one point, only the _new_ trapezoidal area is added. This reduces distance computation from O(n²) to **O(1) per step**.

State maintained per cluster in `k_dict`:
- `mid_dist` — Distance for the overlapping portion
- `real_dist` — Total distance including non-overlapping portions
- `lastp` — Last processed point in the cluster centre
- `j` — Current index into the cluster centre

## OD Proxy (Q-RLSTC)

Q-RLSTC also supports a lightweight surrogate for per-step reward computation:

```python
# Segment centroid-based running average
od_proxy = (current_od * n_segments + segment_cost) / (n_segments + 1)
```

This is _intentionally_ less precise than full IED — the reward only needs a directional signal (is clustering improving?), not exact distance values.

## K-Means Clustering

**When used:** Episode-end evaluation only — _not_ per-step.

Defined in [`classical_kmeans.py`](../../q_rlstc/clustering/classical_kmeans.py). Standard Lloyd's algorithm with k-means++ initialisation:

```python
class ClassicalKMeans:
    def fit(self, data):
        centroids = self._initialize_centroids(data)     # k-means++
        for _ in range(max_iter):
            labels = self._assign_clusters(data, centroids)
            new_centroids = self._update_centroids(data, labels)
            if max_shift < threshold:
                break
            centroids = new_centroids
        return KMeansResult(centroids, labels, objective)
```

### Evaluation Metrics

Defined in [`metrics.py`](../../q_rlstc/clustering/metrics.py):

| Metric | Measures | Usage |
|---|---|---|
| **Overall Distance (OD)** | Sum of distances to cluster centroids | Primary quality metric |
| **Silhouette Score** | Cluster cohesion vs. separation | Independent of boundary accuracy |
| **Segmentation F1** | Boundary detection accuracy | Against ground-truth boundaries |

## Incremental Cluster Management

Ported from RLSTCcode's `cluster.py` to provide online cluster updates during RL training. Defined in the lower half of [`classical_kmeans.py`](../../q_rlstc/clustering/classical_kmeans.py):

| Function | Purpose |
|---|---|
| `add_to_cluster(cluster_dict, id, sub_traj, dist)` | Assign a sub-trajectory to a cluster incrementally |
| `compute_center(cluster_dict, id)` | Recompute cluster center from time-indexed points |
| `update_all_centers(cluster_dict)` | Update all centers + reset accumulators (end of round) |
| `compute_overdist(cluster_dict)` | Compute overall distance: `sum(dists) / count(segments)` |
| `initialize_cluster_dict(n_clusters, centers)` | Create empty cluster dict with optional pre-computed centers |

### Cluster Dictionary Format

```python
cluster_dict[k] = [
    distances_list,       # [0] per-segment IED distances
    segment_trajs,        # [1] list of sub-trajectory Trajectory objects
    center_points,        # [2] center trajectory points (List[Point])
    time_indexed_points,  # [3] defaultdict(list) — points indexed by time
]
```

## Data Loading — `pickle_loader.py`

Loads RLSTCcode-format pickle files directly. Defined in [`pickle_loader.py`](../../q_rlstc/clustering/pickle_loader.py):

| Function | Purpose |
|---|---|
| `load_trajectories(path)` | Load `Tdrive_norm_traj` → List[Trajectory] |
| `load_raw_trajectories(path)` | Load as raw RLSTCcode Traj objects (for classical arm) |
| `load_cluster_centers(path)` | Load `tdrive_clustercenter` → (cluster_dict, baseline_od) |
| `load_cluster_centers_raw(path)` | Load in MDP.py's native dict format |
| `load_subtrajectories(path)` | Load TRACLUS sub-trajectories |
| `load_test_set(path)` | Load held-out test/validation sets |

## MDL Preprocessing — `preprocessing.py`

The TRACLUS-style MDL simplification pipeline, ported from RLSTCcode. Defined in [`preprocessing.py`](../../q_rlstc/data/preprocessing.py):

| Function | Purpose |
|---|---|
| `simplify_trajectory(traj)` | Greedy MDL-based trajectory simplification |
| `simplify_all(trajectories)` | Apply simplification to all trajectories |
| `preprocess_tdrive(raw, ...)` | Full pipeline: filter → normalize → simplify |
| `filter_by_coordinates(trajs)` | Geographic bounding box filter (Beijing) |
| `normalize_locations(trajs)` | Z-score normalize spatial coordinates |
| `normalize_time(trajs)` | Z-score normalize timestamps |

## Quantum Distance via Swap Test (Optional)

The swap test estimates inner products between amplitude-encoded states. Used only as a **research validation probe** — not in the training loop.

### Algorithm

1. Prepare ancilla qubit in |0⟩
2. Hadamard on ancilla → |+⟩
3. Controlled-SWAP between data registers
4. Hadamard on ancilla
5. Measure: P(0) = 0.5 × (1 + |⟨ψ|φ⟩|²)

### Cost Analysis

| Requirement | Cost |
|---|---|
| Amplitude encoding per segment | O(n) gates |
| Re-encoding when segment grows | Full circuit rebuild |
| Qubits | 2 × log₂(d) + 1 ancilla |
| Shots for ±0.01 precision | ≥ 4,096 |
| Circuit evals per episode | ~100 per trajectory |

**Verdict:** At ~100ms per circuit evaluation, quantum distance adds ~10 seconds per episode vs. <1ms classical. Acceptable for end-of-episode verification, impractical for per-step reward.

---

**Next:** [Classical vs. Quantum Justifications →](justifications.md)
