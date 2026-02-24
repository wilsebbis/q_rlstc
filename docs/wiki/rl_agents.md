# RL Agents

---

## Overview

Q-RLSTC includes **four DQN agents** — one quantum and three classical — sharing a common interface so the experiment runner can treat them identically. This enables controlled comparison of quantum policy networks against classical baselines.

## Agent Comparison

| Agent | Module | Optimizer | Architecture | Key Hyperparameters |
|-------|--------|-----------|--------------|---------------------|
| **VQDQNAgent** | `rl/vqdqn_agent.py` | SPSA | 5q HEA quantum circuit | shots=512, 3 layers |
| **SPSAClassicalDQN** | `rl/spsa_classical_agent.py` | SPSA | MLP (configurable) | Same SPSA config as VQDQN |
| **AdamClassicalDQN** | `rl/adam_classical_agent.py` | Adam (backprop) | MLP [64] | lr=1e-3, β₁=0.9, β₂=0.999 |
| **OriginalClassicalDQN** | `rl/original_classical_agent.py` | SGD | MLP [64] (faithful RLSTCcode) | lr=0.001, γ=0.99, τ=0.05 |

## Common Interface

All four agents implement the same public API:

```python
agent.get_q_values(state, use_target=False) → np.ndarray  # (2,) Q-values
agent.act(state, greedy=False) → int                       # ε-greedy action
agent.update(states, actions, rewards, next_states, dones)  # batch update
agent.compute_targets_batch(rewards, next_states, dones)    # TD targets
agent.update_target_network()                               # copy online → target
agent.decay_epsilon()                                       # ε decay + target sync
agent.save_checkpoint(path)                                 # serialize
agent.load_checkpoint(path)                                 # deserialize
```

## Design Rationale

### Why Four Agents?

Each agent isolates a specific experimental variable:

1. **VQDQNAgent** — The quantum policy network under test
2. **SPSAClassicalDQN** — Controls for optimizer: same SPSA, classical network.
   Any performance difference vs. VQDQN is attributable to the quantum circuit.
3. **AdamClassicalDQN** — Controls for architecture: shows how well a classical
   MLP performs with a strong optimizer (Adam + backprop).
4. **OriginalClassicalDQN** — 1:1 faithful reproduction of the original RLSTCcode
   DQN (SGD, γ=0.99, soft Polyak updates). Ensures backward compatibility.

### Architecture Details

#### VQDQNAgent (Quantum)
- **Encoding:** Angle encoding on 5 qubits (one per state feature)
- **Ansatz:** Hardware-Efficient Ansatz with `n_layers` repetitions of RY+RZ+CNOT
- **Readout:** Z-expectation values → scale + bias → Q-values
- **Double DQN:** Online params for action selection, target params for evaluation

#### Classical Agents (MLP)
- **Default architecture:** 5→64→2 (single hidden layer with ReLU)
- **Configurable:** `hidden_sizes` parameter allows arbitrary depth/width
- **Feature transforms:** SPSAClassicalDQN supports RBF features for nonlinear readout
- **Weight init:** Xavier uniform for weights, zero for biases

## Supporting Components

| Component | Module | Purpose |
|-----------|--------|---------|
| `ReplayBuffer` | `rl/replay_buffer.py` | Stores `(s, a, r, s', done)` transitions |
| `SPSAOptimizer` | `rl/spsa.py` | Gradient-free optimization with learning rate scheduling |
| `VQDQNCircuitBuilder` | `quantum/vqdqn_circuit.py` | Builds and evaluates quantum circuits |
| `BackendFactory` | `quantum/backends.py` | Creates Qiskit backends (ideal, noisy, IBM Runtime) |

## Configuration

Each agent has a corresponding `@dataclass` config:

```python
AgentConfig         # VQDQNAgent
ClassicalAgentConfig # SPSAClassicalDQN
AdamAgentConfig     # AdamClassicalDQN
OriginalAgentConfig # OriginalClassicalDQN
```

Shared defaults: `gamma=0.90` (except Original: 0.99), `epsilon: 1.0→0.1 (decay=0.99)`, `use_double_dqn=True`, `target_update_freq=10`.

---

[→ Data Layer](data_layer.md) | [→ Training Pipeline](training_pipeline.md)
