"""Faithful reimplementation of RLSTCcode/subtrajcluster/rl_nn.py DeepQNetwork.

This is the EXACT classical control: same architecture (5→64→2), same optimizer
(SGD lr=0.001), same hyperparameters (γ=0.99, ε=1→0.1 decay=0.99, mem=5000),
same soft target update (τ=0.05), same Huber loss, same batch_size=32.

Purpose: isolate the effect of ALL q_rlstc changes (quantum circuit, SPSA,
reward shaping, clamping, etc.) from the original RLSTCcode baseline.

Pure NumPy — no TensorFlow dependency.
"""

import numpy as np
import random
from collections import deque
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class OriginalAgentConfig:
    """Configuration matching RLSTCcode defaults exactly."""
    # Architecture: 5→64→2 (single hidden layer, ReLU, linear output)
    hidden_size: int = 64
    # Hyperparameters from rl_nn.py
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay: float = 0.99
    learning_rate: float = 0.001
    memory_size: int = 5000
    batch_size: int = 32
    # Soft target update (from rl_train.py: soft_update(0.05))
    tau: float = 0.05


class OriginalClassicalDQN:
    """1:1 faithful reproduction of RLSTCcode's DeepQNetwork.

    Architecture: Sequential([Dense(64, relu), Dense(2, linear)])
    Optimizer: SGD (lr=0.001)
    Loss: Huber (δ=1.0)
    Target update: soft (τ=0.05)

    Same interface as SPSAClassicalDQN / AdamClassicalDQN / VQDQNAgent.
    """

    STATE_DIM = 5
    ACTION_DIM = 2

    def __init__(
        self,
        config: Optional[OriginalAgentConfig] = None,
        seed: int = 42,
    ):
        self.config = config or OriginalAgentConfig()
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        h = self.config.hidden_size

        # Xavier initialization (matches Keras default glorot_uniform ≈ Xavier)
        std1 = np.sqrt(2.0 / (self.STATE_DIM + h))
        self.W1 = self.rng.normal(0, std1, (self.STATE_DIM, h))
        self.b1 = np.zeros(h)

        std2 = np.sqrt(2.0 / (h + self.ACTION_DIM))
        self.W2 = self.rng.normal(0, std2, (h, self.ACTION_DIM))
        self.b2 = np.zeros(self.ACTION_DIM)

        # Target network (deep copy)
        self.W1_target = self.W1.copy()
        self.b1_target = self.b1.copy()
        self.W2_target = self.W2.copy()
        self.b2_target = self.b2.copy()

        # Replay memory (matches deque(maxlen=5000))
        self.memory = deque(maxlen=self.config.memory_size)

        # Exploration
        self.epsilon = self.config.epsilon_start

        # Parameter count: 5*64 + 64 + 64*2 + 2 = 514
        self.n_params = (self.STATE_DIM * h + h) + (h * self.ACTION_DIM + self.ACTION_DIM)

        # Statistics
        self.episode_count = 0
        self.training_step = 0

    # ── Forward pass ────────────────────────────────────────────────

    def _forward(self, states: np.ndarray, use_target: bool = False) -> np.ndarray:
        """Forward pass: 5→64(ReLU)→2(linear).

        Matches Keras Sequential([Dense(64, relu), Dense(2)]).
        """
        x = np.atleast_2d(states)
        if use_target:
            z1 = x @ self.W1_target + self.b1_target
            a1 = np.maximum(0, z1)
            return a1 @ self.W2_target + self.b2_target
        else:
            z1 = x @ self.W1 + self.b1
            a1 = np.maximum(0, z1)
            return a1 @ self.W2 + self.b2

    def _forward_with_cache(self, states: np.ndarray):
        """Forward pass returning intermediate values for backprop."""
        x = np.atleast_2d(states)
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = a1 @ self.W2 + self.b2
        return z2, x, z1, a1

    # ── Action selection ────────────────────────────────────────────

    def get_q_values(
        self, state: np.ndarray, use_target: bool = False
    ) -> np.ndarray:
        """Compute Q-values for a single state."""
        q = self._forward(state, use_target)
        return q.ravel()

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        """ε-greedy action selection (matches rl_nn.py act())."""
        if not greedy and np.random.rand() <= self.epsilon:
            return random.randrange(self.ACTION_DIM)
        q = self._forward(state)
        return int(np.argmax(q.ravel()))

    def fast_online_act(self, state: np.ndarray) -> int:
        """Greedy action using cached weights (matches rl_nn.py fast_online_act)."""
        return self.act(state, greedy=True)

    # ── Replay memory ──────────────────────────────────────────────

    def remember(self, state, action, reward, next_state, done):
        """Store transition (matches rl_nn.py remember())."""
        self.memory.append((state, action, reward, next_state, done))

    # ── Training (SGD + Huber) ─────────────────────────────────────

    def replay(self, batch_size: int = None):
        """Sample minibatch and perform one SGD step with Huber loss.

        Faithfully reproduces rl_nn.py replay() logic:
        1. Sample minibatch
        2. Compute full target Q-values (not just for taken action)
        3. Overwrite taken action's target with TD target
        4. Backprop Huber loss, SGD step
        5. Decay epsilon
        """
        batch_size = batch_size or self.config.batch_size
        if len(self.memory) < batch_size:
            return 0.0

        minibatch = random.sample(self.memory, batch_size)

        states = np.array([np.atleast_1d(s).ravel() for s, _, _, _, _ in minibatch])
        actions = np.array([a for _, a, _, _, _ in minibatch])
        rewards = np.array([r for _, _, r, _, _ in minibatch])
        next_states = np.array([np.atleast_1d(ns).ravel() for _, _, _, ns, _ in minibatch])
        dones = np.array([float(d) for _, _, _, _, d in minibatch])

        # Compute targets: model.predict(state), then overwrite action slot
        targets_full = self._forward(states)  # (B, 2)

        # TD target: r + (1-done) * γ * max(target_model.predict(next_state))
        next_q_target = self._forward(next_states, use_target=True)
        td_targets = rewards + (1 - dones) * self.config.gamma * np.max(next_q_target, axis=1)

        # Overwrite only the taken action's Q-value
        targets_full[np.arange(batch_size), actions] = td_targets

        # Backprop: Huber loss between model output and targets_full
        loss = self._sgd_step(states, targets_full)

        # Decay epsilon
        if self.epsilon > self.config.epsilon_min:
            self.epsilon *= self.config.epsilon_decay

        self.training_step += 1
        return loss

    def _sgd_step(self, states: np.ndarray, targets: np.ndarray) -> float:
        """One SGD step with Huber loss (δ=1.0).

        Matches Keras model.fit(states, targets, epochs=1).
        """
        B = len(states)
        q_pred, x_in, z1, a1 = self._forward_with_cache(states)

        # Huber loss gradient (all outputs, not just taken action)
        error = targets - q_pred  # (B, 2)
        delta = 1.0
        abs_err = np.abs(error)
        loss = np.where(abs_err <= delta, 0.5 * error ** 2, delta * (abs_err - 0.5 * delta))
        scalar_loss = float(loss.mean())

        # d(Huber)/d(pred) = -(target - pred) clamped at ±δ
        d_out = np.where(abs_err <= delta, -error, -delta * np.sign(error)) / B

        # Backprop through layer 2: z2 = a1 @ W2 + b2
        dW2 = a1.T @ d_out
        db2 = d_out.sum(axis=0)

        # Backprop through ReLU
        d_a1 = d_out @ self.W2.T
        d_z1 = d_a1 * (z1 > 0).astype(np.float64)

        # Backprop through layer 1: z1 = x @ W1 + b1
        dW1 = x_in.T @ d_z1
        db1 = d_z1.sum(axis=0)

        # SGD update (lr=0.001)
        lr = self.config.learning_rate
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

        return scalar_loss

    # ── Target network ─────────────────────────────────────────────

    def soft_update(self, tau: float = None):
        """Soft (Polyak) target update: θ_target ← τ·θ + (1-τ)·θ_target.

        Matches rl_nn.py soft_update(w=0.05).
        """
        tau = tau if tau is not None else self.config.tau
        self.W1_target = tau * self.W1 + (1 - tau) * self.W1_target
        self.b1_target = tau * self.b1 + (1 - tau) * self.b1_target
        self.W2_target = tau * self.W2 + (1 - tau) * self.W2_target
        self.b2_target = tau * self.b2 + (1 - tau) * self.b2_target

    def update_target_network(self):
        """Hard copy (for compatibility with q_rlstc interface)."""
        self.W1_target = self.W1.copy()
        self.b1_target = self.b1.copy()
        self.W2_target = self.W2.copy()
        self.b2_target = self.b2.copy()

    def decay_epsilon(self):
        """Decay ε (compatible with q_rlstc training loop interface)."""
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay,
        )
        self.episode_count += 1

    # ── Update (q_rlstc training loop interface) ───────────────────

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        """Batch update interface (compatible with q_rlstc training loop).

        This bypasses the replay buffer and directly trains on the provided
        batch, matching the interface of AdamClassicalDQN.update().
        """
        B = len(states)
        targets_full = self._forward(states)
        next_q = self._forward(next_states, use_target=True)
        td_targets = rewards + (1 - dones.astype(float)) * self.config.gamma * np.max(next_q, axis=1)
        targets_full[np.arange(B), actions.astype(int)] = td_targets
        return self._sgd_step(states, targets_full)

    def compute_targets_batch(
        self,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        """Compute TD targets (compatible with q_rlstc interface)."""
        targets = rewards.copy()
        alive = ~dones.astype(bool)
        if alive.any():
            next_q = self._forward(next_states[alive], use_target=True)
            targets[alive] += self.config.gamma * np.max(next_q, axis=1)
        return targets

    # ── Checkpoint ─────────────────────────────────────────────────

    def save_checkpoint(self, path: str):
        data = {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "W1_target": self.W1_target, "b1_target": self.b1_target,
            "W2_target": self.W2_target, "b2_target": self.b2_target,
            "epsilon": self.epsilon,
            "episode_count": self.episode_count,
            "training_step": self.training_step,
        }
        np.save(path, data, allow_pickle=True)

    def load_checkpoint(self, path: str):
        data = np.load(path, allow_pickle=True).item()
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]
        self.W1_target = data["W1_target"]
        self.b1_target = data["b1_target"]
        self.W2_target = data["W2_target"]
        self.b2_target = data["b2_target"]
        self.epsilon = data["epsilon"]
        self.episode_count = data.get("episode_count", 0)
        self.training_step = data.get("training_step", 0)
