# Q-RLSTC — Quantum-Enhanced RL for Sub-Trajectory Clustering

A hybrid quantum-classical reinforcement learning framework for trajectory segmentation. Uses a 5-qubit Variational Quantum Deep Q-Network (VQ-DQN) as a parameter-efficient policy approximator (34 trainable parameters), with SPSA optimization, within the RLSTC sub-trajectory clustering pipeline. Evaluated against three classical DQN baselines under matched segmentation budget constraints.

## Quick Start

```bash
# Install
uv sync

# Run all thesis experiments (diagnostics + benchmarks + plots)
python experiments/run_thesis_experiments.py

# Quick diagnostic subset
python experiments/run_thesis_experiments.py --experiments D1,D4,E1 --amount 100 --epochs 3

# Multi-seed run (mean ± 95% CI)
python experiments/run_thesis_experiments.py --experiments E1 --amount 50 --epochs 3 \
    --seeds 42,123,7,99,2025

# Entanglement ablation (no-CNOT vs linear)
python experiments/run_thesis_experiments.py --experiments AB1 --amount 100 --epochs 3 \
    --seeds 42,123,7,99,2025

# Replicate classical RLSTCcode results (faithful MLP comparison)
python experiments/run_cross_comparison.py --amount 500 --run all

# Regenerate plots from saved JSON
python experiments/run_thesis_experiments.py --plots-only results/thesis/thesis_results_*.json

# Run tests
python -m pytest tests/ -v
```

## Project Structure

```
q_rlstc/
├── q_rlstc/                    # Core package
│   ├── data/                   # MDP environment, trajectory types, clustering
│   │   ├── rlstc_mdp.py        #   TrajRLclus — RL environment (state, step, reward)
│   │   ├── rlstc_cluster.py    #   Incremental clustering + ValCR (raw, nValCR, wValCR)
│   │   ├── rlstc_point.py      #   Point data type
│   │   ├── rlstc_point_xy.py   #   Point_xy + point-to-line distance
│   │   ├── rlstc_segment.py    #   Segment data type
│   │   ├── rlstc_traj.py       #   Trajectory data type
│   │   ├── rlstc_trajdistance.py  # IED distance computation
│   │   ├── preprocessing.py    #   GPS filtering & normalization
│   │   ├── dataset_loader.py   #   Abstract dataset loader interface
│   │   └── trajectory_scheduler.py # Train/val split + drift/low-data modes
│   ├── rl/                     # RL agents and optimization
│   │   ├── vqdqn_agent.py      #   VQ-DQN agent (quantum policy)
│   │   ├── spsa_classical_agent.py # Classical DQN (MLP + SPSA)
│   │   ├── adam_classical_agent.py #  Classical DQN (MLP + Adam backprop)
│   │   ├── original_classical_agent.py # RLSTCcode-faithful DQN (5→64→2)
│   │   ├── spsa.py             #   SPSA optimizer
│   │   ├── replay_buffer.py    #   Experience replay
│   │   ├── adaptive_shots.py   #   Q-margin-based shot allocation
│   │   ├── drop_action.py      #   DROP action extension
│   │   └── soft_targets.py     #   Entropy-regularized soft-DQN targets
│   ├── quantum/                # Quantum circuit infrastructure
│   │   ├── vqdqn_circuit.py    #   VQC builder, angle encoding, fast simulation
│   │   └── backends.py         #   Backend factory (ideal, Eagle, Heron noise)
│   ├── clustering/             # Classical clustering utilities
│   │   ├── classical_kmeans.py #   K-means implementation
│   │   ├── metrics.py          #   OD, SSE, F1, wValCR metrics
│   │   ├── pickle_loader.py    #   Pickle file I/O
│   │   ├── trajdistance.py     #   Trajectory distance functions
│   │   └── random_frontier.py  #   Random-policy ValCR frontier
│   ├── utils/                  # Statistical utilities
│   │   └── stats.py            #   Bootstrap CI, paired significance tests
│   └── visualization/          # Plotting
│       └── plot_utils.py       #   All plot functions
├── experiments/                # Experiment scripts
│   ├── run_thesis_experiments.py   # ★ Unified thesis runner (D1-D5, E1-E6, S1, AB1)
│   ├── run_cross_comparison.py     # Classical MLP vs VQ-DQN comparison
│   ├── smoke_matrix.py             # Quick smoke tests across configs
│   ├── compare_results.py          # Result comparison utilities
│   ├── generate_classical_report.py # Classical baseline report
│   └── run_significance_test.py    # Statistical significance tests
├── docs/wiki/                  # Documentation wiki
│   ├── ThesisOutline.md        # ★ Full thesis chapter outline
│   ├── architecture.md         # System architecture
│   ├── mdp_and_rewards.md      # MDP formulation, state space, rewards
│   ├── experimental_design.md  # Experiment matrix and analysis plan
│   ├── quantum_circuit.md      # VQC design and angle encoding
│   ├── training_pipeline.md    # Training loop and SPSA details
│   ├── distance_and_clustering.md  # IED and clustering internals
│   ├── comparison.md           # Classical vs quantum comparison
│   ├── justifications.md       # Design decision rationale
│   ├── noise_and_hardware.md   # NISQ noise models
│   ├── benchmarking.md         # Performance benchmarks
│   ├── api_reference.md        # API documentation
│   ├── technical_deep_dive.md  # Deep technical details
│   ├── debugging.md            # Debugging guide
│   ├── visualization_and_plotting.md  # Plot generation
│   ├── compute_backends.md     # Backend configuration
│   └── roadmap.md              # Future work & research directions
├── tests/                      # Unit tests
│   ├── test_angle_encoding.py
│   ├── test_hea_depth.py
│   ├── test_kmeans_update.py
│   ├── test_metrics.py
│   ├── test_replay_buffer.py
│   └── test_spsa.py
└── results/                    # Experiment outputs (JSON + MD + plots)
```

## Architecture

```
Observation (5D)  →  Angle Encoding  →  5-qubit VQC (n HEA layers)  →  Q(extend), Q(cut)
       ↑                                        ↓
  TrajRLclus MDP  ←────── action ←──── ε-greedy policy
       ↓
  Reward: IED delta × scale − CUT_PENALTY / EXTEND_COST
       ↓
  Replay Buffer  →  SPSA batch update  →  parameter perturbation
```

**Key design choices:**
- **SPSA optimizer** (not backprop) — gradient-free, compatible with quantum hardware
- **Statevector simulation** for noiseless runs; Aer noise models for NISQ experiments
- **Q-value clamping** (±10) and **TD target clamping** — prevents value explosion
- **L_MIN = 3** hard constraint prevents degenerate micro-segments

## Evaluation Metrics

| Metric | Definition | Notes |
|---|---|---|
| **ValCR** (raw) | mean(segment IED) / base_sim | ⚠️ Structurally degenerate — favours over-cutting |
| **nValCR** | mean(IED / segment_len) / base_sim | Per-point normalized; removes length bias |
| **wValCR** | total_IED / total_points / base_sim | Length-weighted; robust to segment count inflation |

**Metric pathology:** Raw ValCR decreases monotonically with CUT% because IED grows with segment length. D1 confirms this. Results are reported via **Pareto-constrained** comparison (best ValCR at matched CUT budgets).

## Experiments

| ID | Name | Description |
|----|------|-------------|
| D1 | ValCR vs CUT% | Metric degeneracy diagnostic (random policy); reports raw, nValCR, wValCR |
| D2 | Q-margin | Q(extend) − Q(cut) evolution per epoch |
| D3 | Training action dist | CUT ratio in training actions |
| D4 | Policy basin test | Forced policies under drift mode |
| D5 | Buffer histogram | Replay buffer action distribution |
| E1 | Core Quantum Utility | VQ-DQN vs classical controls (noiseless) |
| E2 | NISQ Viability | Eagle/Heron noise models |
| E3 | Shot Sensitivity | 128/512/2048 shots |
| E4 | Drift Resilience | Performance under concept drift |
| E5 | Low-Data | 10%/25%/50% data fractions |
| E6 | Version Progression | Circuit architecture comparison |
| S1 | Scalability | Inference timing at 250-1000 trajectories |
| AB1 | Entanglement Ablation | No-CNOT vs linear CNOT circuit comparison |

## Data

The framework uses T-Drive and GeoLife trajectory datasets. Data files are gitignored due to size. Place them in `q_rlstc/data/`:
- `Tdrive_norm_traj` — normalized T-Drive trajectories
- `tdrive_clustercenter` — T-Drive cluster centers
- `tdrive_testset{0-4}` — T-Drive cross-validation folds
- `geolife_norm_traj`, `geolife_clustercenter`, `geolife_testset{0-4}` — GeoLife equivalents

## Scope and Limitations

- **Simulation-only.** All quantum experiments use Qiskit Aer statevector/shot simulation and noise models. No real quantum hardware has been used. Results represent feasibility under ideal and modeled noise conditions.
- **Parameter efficiency, not speedup.** VQ-DQN training is ~100× slower than classical controls due to circuit simulation overhead. The value proposition is compact policy representation (34 vs 514–1,314 parameters), not wall-clock performance.
- **Single dataset family.** Validated on T-Drive and GeoLife (taxi/transportation GPS). Generalization to other trajectory domains has not been demonstrated.
- **ValCR metric limitations.** Raw ValCR is structurally degenerate (Chapter 4 of thesis). All comparisons use budget-constrained evaluation at matched CUT rates.

## Documentation

Full documentation is served via [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/):

```bash
# Local preview
pip install mkdocs-material
mkdocs serve
```

Key pages:
- **[Thesis Outline](docs/wiki/ThesisOutline.md)** — Complete chapter structure
- **[Architecture](docs/wiki/architecture.md)** — System design and data flow
- **[MDP & Rewards](docs/wiki/mdp_and_rewards.md)** — State space, actions, reward engineering
- **[Quantum Circuit](docs/wiki/quantum_circuit.md)** — VQC design and angle encoding
- **[Experimental Design](docs/wiki/experimental_design.md)** — Full experiment matrix
- **[Design Justifications](docs/wiki/justifications.md)** — Why quantum here and not elsewhere
