# Training Pipeline

[← Back to README](../../README.md) · [Circuit Design](quantum_circuit.md) · **Training** · [Distance & Clustering →](distance_and_clustering.md)

---

## Training Loop

Defined in [`train.py`](../../q_rlstc/rl/train.py):

```python
for epoch in range(n_epochs):
    for trajectory in dataset.trajectories:
        state = env.reset(trajectory)

        while not done:
            q_values = agent.get_q_values(state)         # 512 shots
            action = epsilon_greedy(q_values, epsilon)
            next_state, reward, done = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)

            if replay_buffer.is_ready(batch_size):
                batch = replay_buffer.sample(batch_size)
                agent.update(batch)                       # SPSA step

            state = next_state

        agent.decay_epsilon()

    # Episode-end: k-means evaluation (metrics only, not training signal)
    if epoch % eval_interval == 0:
        od_score = classical_kmeans_objective(segments)
        log_metrics(od_score)
```

## SPSA Optimizer

Defined in [`spsa.py`](../../q_rlstc/rl/spsa.py). SPSA estimates gradients with only **2** function evaluations, regardless of parameter count.

### Algorithm

```
θ_{k+1} = θ_k − aₖ · ĝₖ

where:
    Δ ∼ Rademacher(±1)              # Random perturbation direction
    ĝₖ = [L(θ + cₖΔ) − L(θ − cₖΔ)] / (2cₖΔ)   # Gradient estimate
    aₖ = a / (k + A + 1)^α         # Decaying step size
    cₖ = c / (k + 1)^γ             # Decaying perturbation size
```

### Why SPSA (Not Backprop or Parameter-Shift)

| Method | Evals per Step | Works with Shot Noise? | Notes |
|---|---|---|---|
| Backpropagation | 1 (forward+backward) | N/A — requires differentiable model | Cannot differentiate through quantum measurement |
| Parameter-shift | 2 × n_params (40 for 20 params) | Yes | Exact quantum gradients, but expensive |
| **SPSA** | **2** | **Yes** | Approximate but unbiased; O(1) cost |

### Hyperparameters

```python
class SPSAOptimizer:
    a: float = 0.12         # Initial step size
    c: float = 0.10         # Initial perturbation size (larger to overcome shot noise)
    A: int   = 20           # Step size offset (stabilises early training)
    alpha: float = 0.602    # Step size decay rate (standard SPSA theory)
    gamma: float = 0.101    # Perturbation decay rate
```

### Shot Noise Robustness

SPSA is naturally robust to shot noise because:
1. It only needs function _values_, not exact gradients
2. Random perturbations average out noise over iterations
3. Decaying perturbation size reduces noise impact over time

## TD Loss (Huber)

```python
def compute_td_loss(state, action, target):
    q_value = get_q_values(state)[action]
    td_error = target - q_value
    delta = 1.0
    if abs(td_error) <= delta:
        return 0.5 * td_error ** 2         # Smooth near zero
    else:
        return delta * (abs(td_error) - 0.5 * delta)  # Linear for outliers
```

## Double DQN

Standard DQN overestimates Q-values because the same network selects _and_ evaluates actions. Double DQN decouples these:

```python
def compute_target(reward, next_state, done):
    if done:
        return reward

    # Online network selects best action
    best_action = argmax(get_q_values(next_state, use_target=False))

    # Target network evaluates that action
    target_q = get_q_values(next_state, use_target=True)

    return reward + gamma * target_q[best_action]
```

## Target Network Updates

| Strategy | Used By | Mechanism | Default |
|---|---|---|---|
| **Soft update** | RLSTC (original paper) | `θ_target ← τ·θ_online + (1−τ)·θ_target` after each step | τ = 0.05 |
| **Hard copy** | **Q-RLSTC (this implementation)** | `θ_target ← θ_online` every N episodes | N = 10 |

## Experience Replay

Defined in [`replay_buffer.py`](../../q_rlstc/rl/replay_buffer.py):

```python
@dataclass
class Experience:
    state: np.ndarray        # 5D or 8D
    action: int              # 0 (EXTEND) or 1 (CUT)
    reward: float
    next_state: np.ndarray
    done: bool

class ReplayBuffer:
    buffer: deque(maxlen=5_000)
    # Uniform random sampling (no prioritised replay)
```

Training only starts when the buffer has `≥ batch_size` samples.

## Exploration

| Parameter | Value |
|---|---|
| `epsilon_start` | 1.0 (pure exploration) |
| `epsilon_min` | 0.1 (always 10% exploration) |
| `epsilon_decay` | 0.99 per episode |

## Hyperparameter Summary

| Parameter | Default | Description |
|---|---|---|
| `gamma` | 0.90 | Discount factor (shorter horizon; NISQ noise makes long-term credit unreliable) |
| `batch_size` | 32 | Replay batch size |
| `memory_size` | 5,000 | Replay buffer capacity |
| `target_update_freq` | 10 | Episodes between target sync (hard) |
| `shots_train` | 512 | Measurement shots during training |
| `shots_eval` | 4,096 | Measurement shots during evaluation |

---

**Next:** [Distance & Clustering →](distance_and_clustering.md)
