# Distance Estimation & Clustering

[← Back to README](../../README.md) · [Training Pipeline](training_pipeline.md) · **Distance & Clustering** · [Justifications →](justifications.md)

---

## The Distance Problem

Trajectory clustering requires comparing trajectories to cluster centroids repeatedly. Each training episode performs hundreds of distance computations. Q-RLSTC addresses this via incremental classical distance with an optional quantum swap test for research validation.

## Integrated Euclidean Distance (IED)

IED is the primary metric, inherited from classical RLSTC. Defined in [`cluster.py`](../../q_rlstc/../../../RLSTCcode/subtrajcluster/cluster.py) (classical) and approximated via OD proxy in [`features.py`](../../q_rlstc/data/features.py) (Q-RLSTC).

### Geometric Interpretation

IED computes the _area_ between two trajectories over their temporal overlap — the integral of spatial distance across a common time interval.

```
For aligned segments with time interval [tₛ, tₑ]:
    d₁ = distance between starting points
    d₂ = distance between ending points
    IED ≈ 0.5 × (d₁ + d₂) × (tₑ − tₛ)     (trapezoidal approximation)
```

### Why IED?

| Metric | Pros | Cons | Use Case |
|---|---|---|---|
| **IED** | Fast incremental updates, handles different lengths | Requires temporal overlap | Real-time trajectory clustering |
| Fréchet | Shape-invariant, robust | O(n²) computation | Offline analysis |
| DTW | Handles speed variation | Expensive, not incremental | Time series alignment |
| Euclidean | Simple, fast | Requires same length | Fixed-size vectors only |

## Incremental Distance (RLSTC)

The key optimisation: when extending a segment by one point, only the _new_ trapezoidal area is added. This reduces distance computation from O(n²) to **O(1) per step**.

State maintained per cluster:
- `mid_dist` — Distance for the overlapping portion
- `real_dist` — Total distance including non-overlapping portions
- `lastp` — Last processed point in the cluster centre
- `j` — Current index into the cluster centre

## OD Proxy (Q-RLSTC)

Q-RLSTC uses a lightweight surrogate instead of full IED for per-step reward computation:

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
