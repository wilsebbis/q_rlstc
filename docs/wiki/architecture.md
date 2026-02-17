# System Architecture

[← Back to README](../../README.md)

---

## Three-Layer Design

Q-RLSTC operates as a three-layer hybrid system. Each layer is assigned to classical or quantum execution based on algorithmic fit, hardware feasibility, and training-loop frequency.

```
┌──────────────────────────────────────────────────────────────────────┐
│ Layer 1: Feature Extraction (CLASSICAL)                              │
│   Trajectory → 5D/8D state vector via sequential geometry            │
├──────────────────────────────────────────────────────────────────────┤
│ Layer 2: Policy Network (QUANTUM)                                    │
│   State → Angle Encoding → HEA Ansatz → Z-Expectation → Q-values    │
├──────────────────────────────────────────────────────────────────────┤
│ Layer 3: Clustering & Evaluation (CLASSICAL)                         │
│   Segments → K-means → OD / Silhouette / F1 metrics                 │
└──────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Trajectory
    │
    ▼
StateFeatureExtractor          ← Classical: sequential geometric computation
    │
    ▼
VQ-DQN Circuit (5/8 qubits)   ← Quantum: angle encode → variational layers → measure
    │
    ├── Q(EXTEND)
    └── Q(CUT)
         │
         ▼
    ε-greedy action selection
         │
         ▼
    MDPEnvironment.step()      ← Classical: OD proxy, boundary sharpness, penalty
         │
         ├── Reward → Replay Buffer → SPSA Update
         └── Next state (loop)
```

## Design Philosophy

### Hybrid First

Pure quantum solutions are not viable for NISQ. Q-RLSTC applies quantum computation _only_ where it provides value — the policy network — keeping everything else classical. This is not a compromise; it is the architecturally correct choice. See [Justifications](justifications.md) for the component-by-component analysis.

### NISQ Awareness

Every circuit design decision prioritises noise resilience:

- **Shallow depth** (≤11 layers): Errors compound with depth
- **Limited qubit count** (5 or 8): Fewer qubits = fewer error sources
- **Statistical averaging** (512-1024 shots): Reduces shot noise in expectations
- **Linear entanglement**: Fewer 2-qubit gates than ring or full connectivity

### Modularity

Components are designed for independent testing and replacement:

- The VQ-DQN can be swapped for a classical DQN (for controlled experiments)
- The clustering can use purely classical distance
- Noise models are configurable without code changes
- All configuration is centralised in [`config.py`](../../q_rlstc/config.py)

### Reproducibility

Random seeds are exposed at every level. Circuit construction is deterministic given parameters. Results should be reproducible across runs with the same seed.

## Quantum Scope Boundary

| Component | Implementation | Rationale |
|---|---|---|
| Q-value estimation | **Quantum** (VQ-DQN) | Hilbert space expressivity; clean 5→2 mapping |
| State encoding | **Quantum** (Angle) | Bounded features → rotation angles |
| Feature extraction | **Classical** | Sequential path-dependent geometry; no quantum speedup |
| Distance computation | **Classical** | Incremental O(1) updates; quantum would require full re-encoding |
| Clustering (K-Means) | **Classical** | No quantum centroid update algorithm exists |
| Reward computation | **Classical** | Single floating-point arithmetic |

---

**Next:** [MDP & Reward Engineering →](mdp_and_rewards.md)
