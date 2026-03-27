# Project Structure

---

## Directory Layout

```
q_rlstc/
├── q_rlstc/                   # Main Python package
│   ├── __init__.py
│   ├── data/                  # Trajectory data layer
│   │   ├── __init__.py
│   │   ├── dataset_loader.py      # Abstract dataset loader interface
│   │   ├── rlstc_point.py         # Point(x, y, t)
│   │   ├── rlstc_point_xy.py      # Point_xy(x, y) for segment geometry
│   │   ├── rlstc_segment.py       # Directed line segment
│   │   ├── rlstc_traj.py          # Trajectory container
│   │   ├── rlstc_trajdistance.py  # IED, Fréchet, DTW, segment distance
│   │   ├── rlstc_cluster.py       # Incremental IED clustering
│   │   ├── rlstc_mdp.py           # MDP environment (TrajRLclus)
│   │   ├── preprocessing.py       # GPS filtering & normalization
│   │   └── trajectory_scheduler.py # Epoch-based sampling
│   ├── rl/                    # Reinforcement learning agents
│   │   ├── __init__.py
│   │   ├── vqdqn_agent.py         # Quantum DQN agent
│   │   ├── spsa_classical_agent.py # Classical DQN with SPSA
│   │   ├── adam_classical_agent.py # Classical DQN with Adam
│   │   ├── original_classical_agent.py # Faithful RLSTCcode DQN
│   │   ├── replay_buffer.py       # Experience replay
│   │   ├── spsa.py                # SPSA optimizer
│   │   ├── adaptive_shots.py      # Q-margin-based shot allocation
│   │   ├── drop_action.py         # DROP action extension
│   │   └── soft_targets.py        # Entropy-regularized soft-DQN targets
│   ├── quantum/               # Quantum circuit infrastructure
│   │   ├── __init__.py
│   │   ├── vqdqn_circuit.py       # Circuit builder & evaluation
│   │   └── backends.py            # Backend factory (ideal, noisy, Runtime)
│   ├── clustering/            # Clustering algorithms & metrics
│   │   ├── __init__.py
│   │   ├── classical_kmeans.py    # Pure-NumPy k-means
│   │   ├── metrics.py             # OD, Silhouette, Segmentation F1, wValCR
│   │   ├── pickle_loader.py       # RLSTCcode pickle loader
│   │   ├── trajdistance.py        # q_rlstc-native IED
│   │   ├── initcenters.py         # K-means++ initialization
│   │   ├── splitmethod.py         # Post-hoc clustering (DBSCAN, etc.)
│   │   └── random_frontier.py     # Random-policy ValCR frontier
│   ├── utils/                 # Statistical utilities
│   │   ├── __init__.py
│   │   └── stats.py               # Bootstrap CI, paired significance tests
│   └── visualization/         # Plotting utilities
│       ├── __init__.py
│       └── plot_utils.py          # Learning curves, comparisons
├── experiments/               # Experiment runners
│   ├── run_thesis_experiments.py  # Multi-seed thesis experiments
│   ├── run_cross_comparison.py    # 4-agent cross-comparison
│   ├── smoke_matrix.py            # Quick smoke tests across configs
│   ├── compare_results.py         # Result comparison utilities
│   ├── generate_classical_report.py # Classical baseline report
│   └── run_significance_test.py   # Statistical significance tests
├── tests/                     # Unit & integration tests
├── docs/                      # Documentation
│   ├── wiki/                      # Wiki pages (this directory)
│   └── index.md                   # Docs index
├── results/                   # Experiment outputs (gitignored)
├── README.md                  # Project README
├── pyproject.toml             # Package configuration
└── run_all.sh                 # Script to run all experiments
```

## Module Dependencies

```
                    ┌──────────────┐
                    │  experiments/ │
                    └──────┬───────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌──────────┐    ┌──────────┐    ┌──────────────┐
   │  rl/     │    │ quantum/ │    │ clustering/  │
   │ agents   │───▶│ circuits │    │ metrics      │
   │ optimizers│   │ backends │    │ post-hoc     │
   └──────┬───┘    └──────────┘    └──────┬───────┘
          │                               │
          └───────────┬───────────────────┘
                      ▼
              ┌──────────────┐
              │   data/      │
              │ Point, Traj  │
              │ IED, MDP     │
              │ preprocessing│
              └──────────────┘
```

## Key Design Principles

1. **No cross-layer imports.** `data/` never imports from `rl/` or `quantum/`.
2. **Shared agent interface.** All 4 agents expose `act()`, `update()`, `get_q_values()`, etc.
3. **Classical-first data pipeline.** Distance computation, clustering, and MDP are all pure NumPy.
4. **Quantum only in the policy.** The quantum circuit is isolated to `quantum/` and `rl/vqdqn_agent.py`.
5. **RLSTCcode compatibility.** The `_RLSTCUnpickler` class redirects legacy pickle module paths.

## Package Exports

```python
# rl/ exports
from q_rlstc.rl import (
    VQDQNAgent, SPSAClassicalDQN, AdamClassicalDQN, OriginalClassicalDQN,
    ReplayBuffer, Experience, SPSAOptimizer,
)
# quantum/ exports
from q_rlstc.quantum import (
    VQDQNCircuitBuilder, BackendFactory,
    build_vqdqn_circuit, evaluate_q_values, angle_encode, get_backend,
)
# clustering/ exports
from q_rlstc.clustering import (
    ClassicalKMeans, KMeansResult, kmeans_fit,
    overall_distance, silhouette_score, segmentation_f1,
    initialize_centers, compute_distance_matrix,
)
# utils/ exports
from q_rlstc.utils import bootstrap_ci, paired_bootstrap_test
```

---

[→ Data Layer](data_layer.md) | [→ RL Agents](rl_agents.md) | [→ Architecture](architecture.md)
