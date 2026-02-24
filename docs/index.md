# Q-RLSTC

**Quantum-Enhanced Reinforcement Learning for Sub-Trajectory Clustering**

---

## What is Q-RLSTC?

Q-RLSTC uses a **Variational Quantum Deep Q-Network (VQ-DQN)** to learn optimal trajectory segmentation policies. A 5-qubit quantum circuit serves as a parameter-efficient policy network (34 trainable parameters), deciding where to cut GPS trajectories into meaningful sub-trajectories that cluster well together.

> [!NOTE]
> **Scope:** All quantum experiments use Qiskit Aer simulation (statevector + noise models). The value proposition is parameter efficiency (34 vs 514–1,314 classical parameters), not training speedup. Results are reported under budget-constrained evaluation to account for ValCR metric degeneracy.

## Quick Links

| Section | Description |
|---------|-------------|
| [Architecture](wiki/architecture.md) | Three-layer hybrid system design |
| [Data Layer](wiki/data_layer.md) | Point, Traj, IED distance, MDP environment |
| [RL Agents](wiki/rl_agents.md) | 4 DQN agents (1 quantum, 3 classical baselines) |
| [Training Pipeline](wiki/training_pipeline.md) | End-to-end training workflow |
| [Quantum Circuit](wiki/quantum_circuit.md) | VQ-DQN circuit architecture |
| [API Reference](wiki/api_reference.md) | Module-level API documentation |
| [Project Structure](wiki/project_structure.md) | Directory layout and module dependencies |

## Architecture at a Glance

```
┌──────────────────────────────────────────────────────────────────┐
│ Layer 1: Environment & Distance Computation (CLASSICAL)          │
│   Trajectory → Incremental IED → 5D state observation            │
├──────────────────────────────────────────────────────────────────┤
│ Layer 2: Policy Network (QUANTUM or CLASSICAL)                   │
│   State → Angle Encoding → 5q HEA → Z-Expectation → Q-values    │
├──────────────────────────────────────────────────────────────────┤
│ Layer 3: Clustering & Evaluation (CLASSICAL)                     │
│   Segments → Incremental center updates → ValCR evaluation       │
└──────────────────────────────────────────────────────────────────┘
```

## Getting Started

```bash
# Install
pip install -e ".[dev]"

# Run experiments
python experiments/run_thesis_experiments.py --help

# Build docs
pip install mkdocs-material
mkdocs serve
```
