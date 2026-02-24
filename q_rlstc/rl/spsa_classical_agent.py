"""Classical MLP-based DQN agent using SPSA optimizer.

Provides a fair classical baseline for quantum comparison by using
the SAME optimizer (SPSA) as the VQ-DQN agent. No SGD, no Adam,
no backpropagation — gradient-free optimization only.

Supports variable architectures:
  - Control A: 5→4→2 MLP (34 params)  — parameter-matched to VQ-DQN
  - Control B: 5→64→2 MLP (~514 params) — original paper architecture
  - Control C: 5→2 linear (12 params) — linearity test
  - Control D: 5→RBF(10)→2 (22 params) — structurally different (not an MLP)
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .spsa import SPSAOptimizer
from .replay_buffer import ReplayBuffer


@dataclass
class ClassicalAgentConfig:
    """Configuration for classical SPSA-DQN agent.

    Attributes:
        hidden_sizes: List of hidden layer sizes. Empty = linear.
        feature_transform: Feature transform before linear readout.
            "none": standard MLP.
            "rbf": Random Fourier features φ(x) = cos(Wx + b)
                with fixed random projections, then linear readout.
        rbf_dim: Number of RBF random features (only if feature_transform="rbf").
        gamma: Discount factor.
        epsilon_start: Initial exploration rate.
        epsilon_min: Minimum exploration rate.
        epsilon_decay: Decay rate per episode.
        use_double_dqn: Whether to use Double DQN.
        target_update_freq: Episodes between hard target copies.
    """
    hidden_sizes: List[int] = None  # set in __post_init__
    feature_transform: str = "none"  # "none" or "rbf"
    rbf_dim: int = 10
    gamma: float = 0.90
    epsilon_start: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay: float = 0.99
    use_double_dqn: bool = True
    target_update_freq: int = 10

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [64]


class SPSAClassicalDQN:
    """Classical DQN with MLP policy, optimized by SPSA.

    Same interface as VQDQNAgent so the benchmark runner can
    treat quantum and classical models identically.
    """

    STATE_DIM = 5
    ACTION_DIM = 2

    def __init__(
        self,
        config: Optional[ClassicalAgentConfig] = None,
        seed: int = 42,
    ):
        self.config = config or ClassicalAgentConfig()
        self.rng = np.random.default_rng(seed)
        self._use_rbf = self.config.feature_transform == "rbf"

        if self._use_rbf:
            # Random Fourier Features: φ(x) = cos(W_fixed @ x + b_fixed)
            # Only the linear readout (rbf_dim → 2) is trainable.
            rbf_dim = self.config.rbf_dim
            self._rbf_W = self.rng.normal(0, 1.0, (self.STATE_DIM, rbf_dim))
            self._rbf_b = self.rng.uniform(0, 2 * np.pi, rbf_dim)
            # Trainable: readout weights (rbf_dim × 2) + biases (2)
            self._layer_shapes = [((rbf_dim, self.ACTION_DIM), (self.ACTION_DIM,))]
            self.n_params = rbf_dim * self.ACTION_DIM + self.ACTION_DIM
        else:
            # Standard MLP
            dims = [self.STATE_DIM] + list(self.config.hidden_sizes) + [self.ACTION_DIM]
            self._layer_shapes = []
            n_params = 0
            for i in range(len(dims) - 1):
                w_shape = (dims[i], dims[i + 1])
                b_shape = (dims[i + 1],)
                self._layer_shapes.append((w_shape, b_shape))
                n_params += w_shape[0] * w_shape[1] + b_shape[0]
            self.n_params = n_params

        # Flat parameter vector (Xavier init)
        self.params = np.zeros(self.n_params)
        offset = 0
        for w_shape, b_shape in self._layer_shapes:
            fan_in, fan_out = w_shape
            std = np.sqrt(2.0 / (fan_in + fan_out))
            n_w = fan_in * fan_out
            n_b = b_shape[0]
            self.params[offset:offset + n_w] = self.rng.normal(0, std, n_w)
            offset += n_w
            # biases start at zero
            offset += n_b

        self.target_params = self.params.copy()

        # Exploration
        self.epsilon = self.config.epsilon_start

        # SPSA optimizer (same config as quantum)
        self.optimizer = SPSAOptimizer(
            a=0.12, c=0.10, A=20,
            alpha=0.602, gamma=0.101,
            seed=seed,
        )

        # Statistics
        self.episode_count = 0
        self.training_step = 0

    # ── MLP forward pass ──────────────────────────────────────────

    def _forward(self, states: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Forward pass through the model.

        For MLP: standard dense layers with ReLU.
        For RBF: fixed random projection → cos nonlinearity → trainable linear.

        Args:
            states: (B, 5) or (5,) input states.
            params: Flat parameter vector.

        Returns:
            (B, 2) Q-values.
        """
        x = np.atleast_2d(states)

        if self._use_rbf:
            # Fixed random Fourier features (not trainable)
            x = np.cos(x @ self._rbf_W + self._rbf_b)  # (B, rbf_dim)

        # Linear readout (trainable) — or full MLP layers
        offset = 0
        for i, (w_shape, b_shape) in enumerate(self._layer_shapes):
            n_w = w_shape[0] * w_shape[1]
            n_b = b_shape[0]
            W = params[offset:offset + n_w].reshape(w_shape)
            offset += n_w
            b = params[offset:offset + n_b]
            offset += n_b
            x = x @ W + b
            # ReLU on all layers except the last (MLP only; RBF has 1 layer)
            if not self._use_rbf and i < len(self._layer_shapes) - 1:
                x = np.maximum(0, x)
        # Clamp outputs to prevent value explosion (max |Q| ≈ γ/(1-γ) * max|r| ≈ 10)
        return np.clip(x, -10.0, 10.0)  # (B, 2)

    def _forward_batch(
        self, states: np.ndarray, params: np.ndarray
    ) -> np.ndarray:
        """Alias for _forward (already supports batches)."""
        return self._forward(states, params)

    # ── Action selection ──────────────────────────────────────────

    def get_q_values(
        self, state: np.ndarray, use_target: bool = False
    ) -> np.ndarray:
        """Compute Q-values for a single state."""
        p = self.target_params if use_target else self.params
        return self._forward(state, p).flatten()

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        """ε-greedy action selection."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(2))
        q = self.get_q_values(state)
        return int(np.argmax(q))

    # ── Target computation ────────────────────────────────────────

    def compute_targets_batch(
        self,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        """Compute TD targets for a batch. Same logic as VQDQNAgent."""
        targets = rewards.copy()
        alive = ~dones.astype(bool)
        if not alive.any():
            return targets

        alive_next = next_states[alive]

        if self.config.use_double_dqn:
            q_online = self._forward_batch(alive_next, self.params)
            q_target = self._forward_batch(alive_next, self.target_params)
            best_actions = np.argmax(q_online, axis=1)
            next_values = q_target[np.arange(len(alive_next)), best_actions]
        else:
            q_target = self._forward_batch(alive_next, self.target_params)
            next_values = np.max(q_target, axis=1)

        targets[alive] += self.config.gamma * next_values
        return np.clip(targets, -10.0, 10.0)

    # ── SPSA update ───────────────────────────────────────────────

    def _batch_loss(
        self,
        params: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """Batched Huber loss. Called twice by SPSA (θ±δ)."""
        q_all = self._forward_batch(states, params)
        B = len(states)
        preds = q_all[np.arange(B), actions.astype(int)]
        td_errors = targets - preds

        delta = 1.0
        abs_err = np.abs(td_errors)
        loss = np.where(
            abs_err <= delta,
            0.5 * td_errors ** 2,
            delta * (abs_err - 0.5 * delta),
        )
        return float(loss.mean())

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        """Perform one SPSA optimization step."""
        targets = self.compute_targets_batch(rewards, next_states, dones)

        def loss_fn(params):
            return self._batch_loss(params, states, actions, targets)

        self.params, _ = self.optimizer.step(loss_fn, self.params)
        self.training_step += 1
        return 0.0

    # ── Epsilon decay + hard target copy ──────────────────────────

    def update_target_network(self) -> None:
        """Hard copy online → target (not soft update)."""
        self.target_params = self.params.copy()

    def decay_epsilon(self) -> None:
        """Decay ε and periodically hard-copy target network."""
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay,
        )
        self.episode_count += 1
        if self.episode_count % self.config.target_update_freq == 0:
            self.update_target_network()

    # ── Persistence ───────────────────────────────────────────────

    def save_checkpoint(self, path: str) -> None:
        np.savez(
            path,
            params=self.params,
            target_params=self.target_params,
            epsilon=self.epsilon,
            episode_count=self.episode_count,
            training_step=self.training_step,
        )

    def load_checkpoint(self, path: str) -> None:
        data = np.load(path)
        self.params = data["params"]
        self.target_params = data["target_params"]
        self.epsilon = float(data["epsilon"])
        self.episode_count = int(data["episode_count"])
        self.training_step = int(data["training_step"])
