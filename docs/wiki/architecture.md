# System Architecture

[← Back to README](../../README.md)

---

## Three-Layer Design

Q-RLSTC operates as a three-layer hybrid system. Each layer is assigned to classical or quantum execution based on algorithmic fit, hardware feasibility, and training-loop frequency.

```
┌──────────────────────────────────────────────────────────────────────┐
│ Layer 1: Feature Extraction & Preprocessing (CLASSICAL)              │
│   Trajectory → MDL simplification → IED distance → 5–8D state       │
├──────────────────────────────────────────────────────────────────────┤
│ Layer 2: Policy Network (QUANTUM)                                    │
│   State → Angle Encoding → HEA/EQC Ansatz → Z-Expectation → Q-vals  │
├──────────────────────────────────────────────────────────────────────┤
│ Layer 3: Clustering & Evaluation (CLASSICAL)                         │
│   Segments → Incremental updates → K-means → OD / Silhouette / F1   │
└──────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Raw Data (T-Drive pickle)
    │
    ▼
pickle_loader.py               ← Load pre-processed trajectory data
    │
    ▼
preprocessing.py                ← MDL simplification (if raw data)
    │
    ▼
StateFeatureExtractor           ← Classical: sequential geometric computation
    │                              (5D for A/D, 8D for B, 5D+shadow for C)
    ▼
VQ-DQN Circuit (5–8 qubits)    ← Quantum: angle encode → variational layers → measure
    │
    ├── Q(EXTEND)
    ├── Q(CUT)
    └── Q(DROP/SKIP) [C/D only]
         │
         ▼
    ε-greedy / SAC action selection
         │
         ▼
    MDPEnvironment.step()       ← Classical: IED distance, OD proxy, boundary sharpness
         │
         ├── Reward → Replay Buffer → SPSA/m-SPSA Update
         └── Next state (loop)
         
    Episode end:
         │
         ▼
    Incremental cluster updates  ← add_to_cluster → compute_center → update_all_centers
         │
         ▼
    K-means evaluation           ← OD, Silhouette, F1 metrics
```

## Design Philosophy

### Hybrid First

Pure quantum solutions are not viable for NISQ. Q-RLSTC applies quantum computation _only_ where it provides value — the policy network — keeping everything else classical. This is not a compromise; it is the architecturally correct choice. See [Justifications](justifications.md) for the component-by-component analysis.

### NISQ Awareness

Every circuit design decision prioritises noise resilience:

- **Shallow depth** (≤11 layers): Errors compound with depth
- **Limited qubit count** (5–8): Fewer qubits = fewer error sources
- **Statistical averaging** (512–4096 shots): Reduces shot noise in expectations
- **Linear/circular entanglement**: Fewer 2-qubit gates than ring or full connectivity

### Modularity

Components are designed for independent testing and replacement:

- The VQ-DQN can be swapped for a classical DQN (for controlled experiments)
- The clustering can use purely classical distance
- Noise models are configurable without code changes
- All configuration is centralised in [`config.py`](../../q_rlstc/config.py)
- Distance module (`trajdistance.py`) works identically with both systems
- Pickle loader enables seamless data sharing with RLSTCcode

### Four Versions

| Version | Focus | See |
|---|---|---|
| **A** | Scientific control — minimal quantum, matches classical dimensions | [Technical Deep Dive](technical_deep_dive.md#version-a) |
| **B** | Quantum-native — exploits larger Hilbert space and parity readout | [Technical Deep Dive](technical_deep_dive.md#version-b) |
| **C** | Next-gen — shadow qubit memory, EQC ansatz, SAC agent, adaptive shots | [Technical Deep Dive](technical_deep_dive.md#version-c) |
| **D** | VLDB-aligned — strict 1:1 paper MDP reproduction with VQC substitution | [Technical Deep Dive](technical_deep_dive.md#version-d) |

### Reproducibility

Random seeds are exposed at every level. Circuit construction is deterministic given parameters. Results should be reproducible across runs with the same seed.

## Quantum Scope Boundary

| Component | Implementation | Rationale |
|---|---|---|
| Q-value estimation | **Quantum** (VQ-DQN) | Hilbert space expressivity; clean 5→2 mapping |
| State encoding | **Quantum** (Angle) | Bounded features → rotation angles |
| Feature extraction | **Classical** | Sequential path-dependent geometry; no quantum speedup |
| Distance computation | **Classical** (IED) | Incremental O(1) updates; quantum would require full re-encoding |
| MDL simplification | **Classical** | Greedy compression; inherently sequential |
| Clustering (K-Means) | **Classical** | Incremental center updates; no quantum centroid algorithm exists |
| Reward computation | **Classical** | Single floating-point arithmetic |
| Data loading | **Classical** | Pickle I/O; file system operations |

---

**Next:** [MDP & Reward Engineering →](mdp_and_rewards.md)
