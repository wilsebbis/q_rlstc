# MDP & Reward Engineering

[← Back to README](../../README.md) · [Architecture](architecture.md) · **MDP & Rewards** · [Circuit Design →](quantum_circuit.md)

---

## MDP Formulation

The segmentation problem is modelled as a Markov Decision Process. At each point along a trajectory, the agent observes a state and decides to **extend** the current segment or **cut** to start a new one.

## State Space

### Version A — 5 Dimensions

Defined in [`features.py → StateFeatureExtractor`](../../q_rlstc/data/features.py):

| # | Feature | Description | Normalisation |
|---|---------|-------------|---------------|
| 0 | `od_segment` | Projected OD if we split here | `arctan` |
| 1 | `od_continue` | Projected OD if we extend | `arctan` |
| 2 | `baseline_cost` | TRACLUS-like MDL compression score | `arctan` |
| 3 | `len_backward` | Current segment length / total length | [0, 1] |
| 4 | `len_forward` | Remaining trajectory / total length | [0, 1] |

### Version B — 8 Dimensions

Inherits all Version A features plus three quantum-native additions in [`features.py → StateFeatureExtractorB`](../../q_rlstc/data/features.py):

| # | Feature | Description | Why added |
|---|---------|-------------|-----------|
| 5 | `angle_spread` | Variance of arctan-encoded features | Captures Bloch sphere spread |
| 6 | `curvature_gradient` | Rate of change of segment curvature | 2nd-order geometric signal; tanh-compressed |
| 7 | `segment_density` | Points per unit spatial distance | Congestion vs. free-flow without explicit speed |

### Feature Computation Details

**OD Proxies (Features 0-1):**

```python
def _compute_od_proxy(segment_points, current_od, n_segments):
    """Lightweight proxy for OD — no full k-means."""
    coords = np.array([[p.x, p.y] for p in segment_points])
    centroid = coords.mean(axis=0)
    segment_cost = np.linalg.norm(coords - centroid, axis=1).mean()
    return (current_od * n_segments + segment_cost) / (n_segments + 1)
```

**TRACLUS Baseline (Feature 2):**

Sum of perpendicular distances from interior points to the start-end line, plus a compression cost term (`log₂(line_length)`). Higher values indicate the segment deviates from a straight line — a signal that a segmentation boundary may be appropriate.

## Action Space

| Action | Value | Effect |
|---|---|---|
| **EXTEND** | 0 | Add next point to current segment |
| **CUT** | 1 | End current segment, start new one |

## Anti-Gaming Constraints

Defined as constants in `MDPEnvironment`:

```python
MIN_SEGMENT_LEN = 3    # CUT disallowed if segment < 3 points
MAX_SEGMENTS    = 50   # Episode terminates if exceeded
SEGMENT_PENALTY = 0.1  # λ: per-segment penalty in reward
```

**Enforcement:** If `action == CUT` but `current_segment_len < MIN_SEGMENT_LEN`, the action is forced to EXTEND. This prevents degenerate policies that cut at every step.

## Reward Function

### Why Not Raw OD?

The naive reward `reward = old_od - new_od` has three problems:

1. **Delayed signal** — OD only changes meaningfully on CUT
2. **Sparse rewards** — EXTEND actions receive near-zero reward
3. **Not Markov-safe** — Depends on global clustering state

### Current Design

The reward combines three components:

```
reward = α · od_improvement + β · boundary_sharpness − segment_penalty
```

| Component | Range | When Applied | Purpose |
|---|---|---|---|
| `od_improvement` | [0, ∞) | Every step | Local clustering quality signal |
| `boundary_sharpness × β` | [0, 0.5] | CUT only | Reward cuts at genuine behaviour transitions |
| `−SEGMENT_PENALTY` | -0.1 | CUT only | Penalise over-segmentation |

### Boundary Sharpness

Measures direction change at the proposed cut point — sharper turns indicate better boundaries:

```python
def _compute_boundary_sharpness(boundary_idx):
    v1 = points[boundary_idx] - points[boundary_idx - 1]     # Before
    v2 = points[boundary_idx + 1] - points[boundary_idx]     # After
    cos_angle = dot(v1, v2) / (norm(v1) * norm(v2))
    angle = arccos(clip(cos_angle, -1, 1))                   # [0, π]
    return angle / π                                          # [0, 1]
```

### Variance Delta

Provides per-step feedback for EXTEND actions:

```python
variance_delta = old_variance - new_variance  # Positive if variance decreased
reward += variance_delta * 0.1
```

## Termination

1. **End of trajectory** — All points consumed
2. **Max segments exceeded** — `n_segments ≥ MAX_SEGMENTS`

---

**Next:** [Quantum Circuit Design →](quantum_circuit.md)
