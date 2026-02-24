# MDP & Reward Engineering

[← Back to README](../../README.md) · [Architecture](architecture.md) · **MDP & Rewards** · [Circuit Design →](quantum_circuit.md)

---

## MDP Formulation

The segmentation problem is modelled as a Markov Decision Process. At each point along a trajectory, the agent observes a state and decides to **extend** the current segment or **cut** to start a new one.

**Implementation:** [`rlstc_mdp.py → TrajRLclus`](../../q_rlstc/data/rlstc_mdp.py)

## State Space (5D)

Computed directly in `TrajRLclus.reset()` and `TrajRLclus.step()`:

| # | Feature | Description | Range |
|---|---------|-------------|-------|
| 0 | `overall_sim` | Global overdistance (trajectory-to-nearest-center IED) | [0, ∞) |
| 1 | `split_overdist` | Running average OD incorporating current segment | [0, ∞) |
| 2 | `overall_sim × 10` | Scaled OD (omitted when `ablate_odb=True` → 4D state) | [0, ∞) |
| 3 | `len_backward` | Current segment length / total trajectory length | [0, 1] |
| 4 | `len_forward` | Remaining trajectory / total trajectory length | [0, 1] |

> **Note:** When `ablate_odb=True`, feature 2 is dropped and the state is 4D. This is tested in the ablation experiments.

## Action Space

| Action | Value | Effect |
|---|---|---|
| **EXTEND** | 0 | Add next point to current segment |
| **CUT** | 1 | End current segment, start new one |

## Anti-Gaming Constraints

Defined as parameters in `TrajRLclus.__init__()`:

```python
min_seg_len = 3  # CUT disallowed if segment < L_MIN points
```

**Enforcement:** If `action == CUT` but the current segment has fewer than `min_seg_len` points, or the remaining trajectory is shorter than `min_seg_len`, the action is silently forced to EXTEND. This prevents degenerate micro-segmentation policies.

```python
# From rlstc_mdp.py step():
if action == 1:
    seg_len = index - self._seg_start_idx + 1
    remaining = self.length - index
    if seg_len < self.min_seg_len or remaining < self.min_seg_len:
        action = 0  # force EXTEND
```

## Reward Function

The reward is computed in the experiment runner (`run_thesis_experiments.py`), not inside the MDP itself. The MDP `step()` returns `reward=0`; the outer training loop applies reward shaping.

### Current Design (PROTOCOL constants)

```python
# From run_thesis_experiments.py PROTOCOL dict:
EXTEND_COST     = 0.01    # Small cost to discourage idle extending
CUT_PENALTY     = 0.12    # Per-cut penalty to discourage over-segmentation
COMPLEXITY_LAMBDA = 0.02  # Complexity regularizer weight
scale_reward    = 100.0   # Amplification factor for OD improvement signal
```

The reward at each step:

```
if action == CUT:
    raw_reward = scale_reward × (old_overdist − new_overdist) − CUT_PENALTY
else:  # EXTEND
    raw_reward = scale_reward × (old_overdist − new_overdist) − EXTEND_COST
```

### Why Externalised Rewards?

1. **Flexibility** — Different experiments can use different reward shaping without modifying the MDP
2. **Clean separation** — The MDP handles environment dynamics; the runner handles learning signals
3. **Transparency** — All reward constants are visible in one PROTOCOL dict

## Q-Value Stability

Both the VQ-DQN and classical agents apply **output clamping** and **TD target clamping** to prevent value explosion:

```python
# Output clamping (in agent._forward / agent.get_q_values):
q_values = np.clip(q_values, -10.0, 10.0)

# TD target clamping (in agent.compute_targets_batch):
targets = np.clip(targets, -10.0, 10.0)
```

This was introduced after observing Q-value explosion to ~78M in early experiments. The ±10 bounds are empirically sufficient given the reward scale.

## Termination

1. **End of trajectory** — All points consumed (`index + 1 == self.length`)
2. The final segment is automatically closed and assigned to a cluster

---

**Next:** [Quantum Circuit Design →](quantum_circuit.md)
