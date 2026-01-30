# Q-RLSTC: Quantum-Enhanced RL for Sub-Trajectory Clustering

Hybrid quantum-classical framework for sub-trajectory clustering using Variational Quantum Deep Q-Networks (VQ-DQN).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Q-RLSTC System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐     ┌─────────────────┐     ┌──────────────┐  │
│  │  Trajectory │────▶│   MDP Env       │────▶│   VQ-DQN     │  │
│  │    Data     │     │  (Segmentation) │     │  (5 qubits)  │  │
│  └─────────────┘     └────────┬────────┘     └──────┬───────┘  │
│                               │                      │          │
│                               ▼                      ▼          │
│  ┌─────────────┐     ┌─────────────────┐     ┌──────────────┐  │
│  │   Hybrid    │◀────│  Reward =       │◀────│   Action:    │  │
│  │   K-Means   │     │  OD Improvement │     │  Extend/Cut  │  │
│  └──────┬──────┘     └─────────────────┘     └──────────────┘  │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐                                               │
│  │  Swap Test  │  ◀── Quantum Distance Estimation              │
│  │  (Amplitude)│                                               │
│  └─────────────┘                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
cd q_rlstc
pip install -e ".[dev]"
```

## Quick Start

```bash
# Run demo
python experiments/run_synth_demo.py

# Run tests
pytest tests/ -v
```

## Encoding Dichotomy

| Component | Encoding | Reason |
|-----------|----------|--------|
| **VQ-DQN Agent** | Angle (arctan) | Efficient for bounded 5-dim state |
| **Swap Test** | Amplitude | Enables inner product via interference |

### Angle Encoding (Policy Network)

Maps unbounded features to rotation angles: θ = 2·arctan(x)

```
State [5 features] ──▶ RY(θ₀)─RY(θ₁)─...─RY(θ₄) ──▶ Variational Layers
```

### Amplitude Encoding (Distance Estimation)

Normalizes vectors and encodes as quantum amplitudes:

```
Vector [N dims] ──▶ |ψ⟩ = Σᵢ αᵢ|i⟩  where αᵢ = xᵢ/‖x‖
```

## NISQ Constraints

- **Shallow circuits**: Depth=2 HEA to minimize decoherence
- **Data re-uploading**: Increases expressivity without depth
- **Shots tradeoff**: 512 training / 1024+ inference
- **~30 parameters**: Avoids barren plateaus
- **Linear entanglement**: CNOT chain (0→1→2→3→4→0)

## Classical vs Quantum Components

| Classical | Quantum |
|-----------|---------|
| Centroid update (mean) | Distance estimation (swap test) |
| Replay buffer | Policy network (VQ-DQN) |
| Feature extraction | Q-value computation |
| SPSA finite differences | Circuit parameter optimization |

## MCP Qiskit Tool Inventory

The MCP qiskit server provides:

| Tool | Purpose |
|------|---------|
| `analyze_circuit_tool` | Circuit metrics (depth, gates) |
| `transpile_circuit_tool` | Transpilation with optimization |
| `compare_optimization_levels_tool` | Compare O0-O3 transpilation |
| `convert_qasm3_to_qpy_tool` | QASM3 → QPY conversion |
| `convert_qpy_to_qasm3_tool` | QPY → QASM3 conversion |

> **Note**: Circuit execution uses local `qiskit-aer` simulator. MCP provides analysis/transpilation.

## Project Structure

```
q_rlstc/
├── pyproject.toml
├── README.md
├── q_rlstc/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── synthetic.py      # Trajectory generation
│   │   └── features.py       # State vector extraction
│   ├── quantum/
│   │   ├── __init__.py
│   │   ├── vqdqn_circuit.py  # VQ-DQN circuit builder
│   │   ├── swaptest_distance.py
│   │   ├── backends.py
│   │   └── mitigation.py
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── replay_buffer.py
│   │   ├── vqdqn_agent.py
│   │   ├── spsa.py
│   │   └── train.py
│   └── clustering/
│       ├── __init__.py
│       ├── hybrid_kmeans.py
│       └── metrics.py
├── experiments/
│   └── run_synth_demo.py
└── tests/
    ├── test_angle_encoding.py
    ├── test_hea_depth.py
    ├── test_swaptest_distance_basic.py
    ├── test_kmeans_update.py
    └── test_training_smoke.py
```

## References

1. Liang et al. — "Sub-trajectory clustering with deep reinforcement learning"
2. Chen et al. — "Variational Quantum Circuits for Deep Reinforcement Learning"
3. Schuld et al. — "Evaluating analytic gradients on quantum hardware"

## License

MIT License - Research code for academic use.
