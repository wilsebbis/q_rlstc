# System Architecture


---

## Three-Layer Hybrid Design

Q-RLSTC operates as a three-layer hybrid system. Each layer is assigned to classical or quantum execution based on algorithmic fit, hardware feasibility, and training-loop frequency.

```
┌──────────────────────────────────────────────────────────────────────┐
│ Layer 1: Environment & Distance Computation (CLASSICAL)              │
│   Trajectory → Incremental IED → 5D state observation                │
├──────────────────────────────────────────────────────────────────────┤
│ Layer 2: Policy Network (QUANTUM or CLASSICAL)                       │
│   State → Angle Encoding → 5q HEA (3L) → Z-Expectation → Q-values   │
│   OR: State → MLP (various sizes) → Q-values                        │
├──────────────────────────────────────────────────────────────────────┤
│ Layer 3: Clustering & Evaluation (CLASSICAL)                         │
│   Segments → Incremental center updates → ValCR evaluation           │
└──────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Raw Data (T-Drive / GeoLife pickle)
    │
    ▼
TrajRLclus.__init__()           ← Load trajectories + cluster centers
    │
    ▼
TrajRLclus.reset(episode)      ← Compute initial IED, 5D observation
    │
    ▼
Agent.act(observation)          ← VQ-DQN or MLP → Q-values → ε-greedy
    │
    ├── Q(EXTEND) ← action 0
    └── Q(CUT)    ← action 1
         │
         ▼
TrajRLclus.step(action)        ← Incremental IED update, segment assignment
    │
    ├── New observation (5D) → loop back to Agent.act()
    └── Segment → cluster_dict[k][0] (IED) + [1] (traj) + [4] (length)

Episode end:
    │
    ▼
compute_overdist(clusters_E)    ← Raw ValCR = mean(IED) / base_similarity
compute_overdist_per_point()    ← nValCR = mean(IED/len) / base_similarity
compute_overdist_length_weighted() ← wValCR = total_IED/total_pts / base
```

## Design Philosophy

### Hybrid First

Pure quantum solutions are not viable for NISQ. Q-RLSTC applies quantum computation _only_ where it provides value — the policy network — keeping everything else classical. This is architecturally correct, not a compromise. See [Justifications](justifications.md).

### NISQ Awareness

Every circuit design decision prioritises noise resilience:

- **Shallow depth**: 3 HEA layers (errors compound with depth)
- **Limited qubits**: 5 qubits (fewer error sources)
- **Statistical averaging**: configurable shot counts (128–4096)
- **Linear entanglement**: fewer 2-qubit gates than ring or full connectivity

### Modularity

Components are designed for independent testing and replacement:

- The VQ-DQN can be swapped for a classical MLP (Controls A/B/C) with identical training pipeline
- Noise models and shot counts are configurable via `backends.py`
- All experiment hyperparameters are centralised in the PROTOCOL dict (`run_thesis_experiments.py`)
- Distance module (`rlstc_trajdistance.py`) works identically with both systems

### Agent Comparison

| Agent | Architecture | Params | Implementation |
|---|---|---|---|
| **VQ-DQN** | 5q × 3L HEA + affine head | 34 | `vqdqn_agent.py` |
| **Control A** | 5→2 linear | 12 | `spsa_classical_agent.py` |
| **Control B** | 5→64→2 MLP | 514 | `spsa_classical_agent.py` |
| **Control C** | 5→32→32→2 MLP | 1,314 | `spsa_classical_agent.py` |

All agents share: SPSA optimizer, Double DQN, experience replay (5000), Huber loss, Q-value clamping (±10), TD target clamping (±10).

## Quantum Scope Boundary

| Component | Implementation | Rationale |
|---|---|---|
| Q-value estimation | **Quantum** (VQ-DQN) | Empirically parameter-efficient; clean 5→2 mapping |
| State encoding | **Quantum** (Angle) | Bounded features → rotation angles |
| Distance computation | **Classical** (IED) | Incremental O(1) updates; quantum would require full re-encoding |
| Clustering | **Classical** | Incremental center updates; no quantum centroid algorithm exists |
| Reward computation | **Classical** | Single floating-point arithmetic in experiment runner |
| Data loading | **Classical** | Pickle I/O |

---

**Next:** [MDP & Reward Engineering →](mdp_and_rewards.md)
