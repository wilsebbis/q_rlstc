"""Classical MLP-based DQN agent using Adam optimizer (backpropagation).

Provides an Adam-trained classical baseline to isolate the effect of
the SPSA optimizer vs. standard backprop. Same architecture options
as SPSAClassicalDQN but uses analytical gradients + Adam.

This is critical for NeurIPS/ICML credibility: reviewers must know
that classical models aren't handicapped by gradient-free optimization.
"""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field

from .replay_buffer import ReplayBuffer


@dataclass
class AdamAgentConfig:
    """Configuration for Adam-trained classical DQN agent."""
    hidden_sizes: List[int] = field(default_factory=lambda: [64])
    gamma: float = 0.90
    epsilon_start: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay: float = 0.99
    use_double_dqn: bool = True
    target_update_freq: int = 10
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-8
    max_grad_norm: float = 10.0
    exploration_mode: str = "epsilon_greedy"  # "epsilon_greedy" or "boltzmann"
    boltzmann_temp: float = 1.0
    boltzmann_temp_min: float = 0.1
    boltzmann_temp_decay: float = 0.99
    q_clip_range: float = 50.0
    optimistic_cut_bias: float = 0.0

    def __post_init__(self):
        self.hidden_sizes = list(self.hidden_sizes)


class AdamClassicalDQN:
    """Classical DQN with MLP policy, optimized by Adam (backprop).

    Same interface as SPSAClassicalDQN / VQDQNAgent so the benchmark
    runner can treat all models identically.
    """

    STATE_DIM = 5
    ACTION_DIM = 2

    def __init__(
        self,
        config: Optional[AdamAgentConfig] = None,
        seed: int = 42,
    ):
        self.config = config or AdamAgentConfig()
        self.rng = np.random.default_rng(seed)

        # Build layer shapes: [(W_shape, b_shape), ...]
        dims = [self.STATE_DIM] + list(self.config.hidden_sizes) + [self.ACTION_DIM]
        self._layer_shapes = []
        self.n_params = 0
        for i in range(len(dims) - 1):
            w_shape = (dims[i], dims[i + 1])
            b_shape = (dims[i + 1],)
            self._layer_shapes.append((w_shape, b_shape))
            self.n_params += w_shape[0] * w_shape[1] + b_shape[0]
        self.n_layers = len(self._layer_shapes)

        # Structured parameters: list of (W, b) tuples
        self.weights = []
        for w_shape, b_shape in self._layer_shapes:
            fan_in, fan_out = w_shape
            std = np.sqrt(2.0 / (fan_in + fan_out))  # Xavier
            W = self.rng.normal(0, std, w_shape)
            b = np.zeros(b_shape)
            self.weights.append((W.copy(), b.copy()))

        # Target network: deep copy
        self.target_weights = [(W.copy(), b.copy()) for W, b in self.weights]

        # Adam state: one (m, v) pair per parameter array
        self.adam_m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in self.weights]
        self.adam_v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in self.weights]
        self.adam_t = 0  # timestep counter

        # Optimistic CUT bias (fair comparison with quantum)
        if self.config.optimistic_cut_bias != 0.0:
            _, b_out = self.weights[-1]
            b_out[1] = self.config.optimistic_cut_bias
            _, b_out_t = self.target_weights[-1]
            b_out_t[1] = self.config.optimistic_cut_bias

        # Exploration
        self.epsilon = self.config.epsilon_start
        self.boltzmann_temp = self.config.boltzmann_temp

        # Statistics
        self.episode_count = 0
        self.training_step = 0

    # ── Forward pass ────────────────────────────────────────────────

    def _forward(self, states: np.ndarray, weights=None) -> np.ndarray:
        """Forward pass, returning Q-values (B, 2).

        Also returns layer activations if `return_cache` is used internally.
        """
        if weights is None:
            weights = self.weights
        x = np.atleast_2d(states)
        for i, (W, b) in enumerate(weights):
            x = x @ W + b
            if i < self.n_layers - 1:
                x = np.maximum(0, x)  # ReLU
        return np.clip(x, -self.config.q_clip_range, self.config.q_clip_range)

    def _forward_with_cache(self, states: np.ndarray, weights=None):
        """Forward pass returning intermediate activations for backprop."""
        if weights is None:
            weights = self.weights
        x = np.atleast_2d(states)
        pre_activations = []  # before ReLU
        activations = [x]     # layer inputs (after ReLU)

        for i, (W, b) in enumerate(weights):
            z = x @ W + b
            pre_activations.append(z)
            if i < self.n_layers - 1:
                x = np.maximum(0, z)  # ReLU
            else:
                x = z  # no activation on output
            activations.append(x)

        # Clamp output
        clamped = np.clip(x, -self.config.q_clip_range, self.config.q_clip_range)
        # Track clamp mask for gradient
        clamp_mask = (x >= -self.config.q_clip_range) & (x <= self.config.q_clip_range)
        activations[-1] = clamped

        return clamped, activations, pre_activations, clamp_mask

    def _forward_batch(self, states: np.ndarray, weights=None) -> np.ndarray:
        """Alias for _forward."""
        return self._forward(states, weights)

    # ── Action selection ────────────────────────────────────────────

    def get_q_values(
        self, state: np.ndarray, use_target: bool = False
    ) -> np.ndarray:
        """Compute Q-values for a single state."""
        w = self.target_weights if use_target else self.weights
        q = self._forward(state, w)
        return q.ravel()

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        """Action selection (ε-greedy or Boltzmann)."""
        q = self.get_q_values(state)
        if greedy:
            return int(np.argmax(q))
        if self.config.exploration_mode == "boltzmann":
            tau = max(self.boltzmann_temp, 1e-8)
            logits = q / tau
            logits -= np.max(logits)
            probs = np.exp(logits)
            probs /= probs.sum()
            return int(self.rng.choice(2, p=probs))
        else:
            if self.rng.random() < self.epsilon:
                return int(self.rng.integers(0, self.ACTION_DIM))
            return int(np.argmax(q))

    # ── TD target computation ──────────────────────────────────────

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
            return np.clip(targets, -self.config.q_clip_range, self.config.q_clip_range)

        alive_next = next_states[alive]

        if self.config.use_double_dqn:
            q_online = self._forward(alive_next, self.weights)
            q_target = self._forward(alive_next, self.target_weights)
            best_actions = np.argmax(q_online, axis=1)
            next_values = q_target[np.arange(len(alive_next)), best_actions]
        else:
            q_target = self._forward(alive_next, self.target_weights)
            next_values = np.max(q_target, axis=1)

        targets[alive] += self.config.gamma * next_values
        return np.clip(targets, -10.0, 10.0)

    # ── Backpropagation ────────────────────────────────────────────

    def _compute_gradients(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
    ):
        """Compute gradients of Huber loss w.r.t. all weights via backprop.

        Returns list of (dW, db) tuples and the scalar loss.
        """
        B = len(states)
        actions = actions.astype(int)

        # Forward pass with cache
        q_all, activations, pre_activations, clamp_mask = \
            self._forward_with_cache(states)

        # Predicted Q-values for taken actions
        preds = q_all[np.arange(B), actions]
        td_errors = targets - preds

        # Huber loss (δ=1.0)
        delta = 1.0
        abs_err = np.abs(td_errors)
        loss = np.where(
            abs_err <= delta,
            0.5 * td_errors ** 2,
            delta * (abs_err - 0.5 * delta),
        )
        scalar_loss = float(loss.mean())

        # Gradient of Huber loss w.r.t. predictions
        # d(loss)/d(pred) = -(target - pred) for |err| <= δ, else -δ·sign(err)
        d_pred = np.where(
            abs_err <= delta,
            -td_errors,
            -delta * np.sign(td_errors),
        ) / B  # mean reduction

        # d(loss)/d(q_all): only the taken actions have gradient
        d_q = np.zeros_like(q_all)
        d_q[np.arange(B), actions] = d_pred

        # Apply clamp mask gradient (zero gradient where clamped)
        d_q = d_q * clamp_mask

        # Backprop through layers (reverse order)
        grads = [None] * self.n_layers
        d_out = d_q  # (B, ACTION_DIM) — gradient at output

        for i in range(self.n_layers - 1, -1, -1):
            W, b = self.weights[i]
            a_in = activations[i]  # input to this layer

            # Gradients for this layer's parameters
            dW = a_in.T @ d_out       # (fan_in, fan_out)
            db = d_out.sum(axis=0)    # (fan_out,)
            grads[i] = (dW, db)

            if i > 0:
                # Backprop through this layer to get gradient for previous layer
                d_out = d_out @ W.T   # (B, fan_in)
                # ReLU derivative (for hidden layers)
                relu_mask = (pre_activations[i - 1] > 0).astype(np.float64)
                d_out = d_out * relu_mask

        return grads, scalar_loss

    # ── Adam update ────────────────────────────────────────────────

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        """Perform one Adam optimization step. Returns loss."""
        # Compute TD targets
        targets = self.compute_targets_batch(rewards, next_states, dones)

        # Compute gradients
        grads, loss = self._compute_gradients(states, actions, targets)

        # Gradient clipping (global norm)
        grad_norm = 0.0
        for dW, db in grads:
            grad_norm += np.sum(dW ** 2) + np.sum(db ** 2)
        grad_norm = np.sqrt(grad_norm)
        if grad_norm > self.config.max_grad_norm:
            clip_coef = self.config.max_grad_norm / (grad_norm + 1e-8)
            grads = [(dW * clip_coef, db * clip_coef) for dW, db in grads]

        # Adam update
        self.adam_t += 1
        lr = self.config.lr
        b1 = self.config.beta1
        b2 = self.config.beta2
        eps = self.config.adam_eps
        t = self.adam_t

        # Bias correction factors
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t

        for i, (dW, db) in enumerate(grads):
            mW, mb = self.adam_m[i]
            vW, vb = self.adam_v[i]
            W, b = self.weights[i]

            # Update moments (weights)
            mW[:] = b1 * mW + (1 - b1) * dW
            vW[:] = b2 * vW + (1 - b2) * dW ** 2
            W -= lr * (mW / bc1) / (np.sqrt(vW / bc2) + eps)

            # Update moments (biases)
            mb[:] = b1 * mb + (1 - b1) * db
            vb[:] = b2 * vb + (1 - b2) * db ** 2
            b -= lr * (mb / bc1) / (np.sqrt(vb / bc2) + eps)

        self.training_step += 1
        return loss

    # ── Target network ─────────────────────────────────────────────

    def update_target_network(self):
        """Hard copy online → target."""
        self.target_weights = [(W.copy(), b.copy()) for W, b in self.weights]

    def decay_epsilon(self):
        """Decay exploration (ε and Boltzmann temperature) + target copy."""
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay,
        )
        self.boltzmann_temp = max(
            self.config.boltzmann_temp_min,
            self.boltzmann_temp * self.config.boltzmann_temp_decay,
        )
        self.episode_count += 1
        if self.episode_count % self.config.target_update_freq == 0:
            self.update_target_network()

    # ── Checkpoint ─────────────────────────────────────────────────

    def save_checkpoint(self, path: str):
        data = {
            "weights": self.weights,
            "target_weights": self.target_weights,
            "epsilon": self.epsilon,
            "episode_count": self.episode_count,
            "training_step": self.training_step,
            "adam_m": self.adam_m,
            "adam_v": self.adam_v,
            "adam_t": self.adam_t,
        }
        np.save(path, data, allow_pickle=True)

    def load_checkpoint(self, path: str):
        data = np.load(path, allow_pickle=True).item()
        self.weights = data["weights"]
        self.target_weights = data["target_weights"]
        self.epsilon = data["epsilon"]
        self.episode_count = data["episode_count"]
        self.training_step = data["training_step"]
        self.adam_m = data.get("adam_m", self.adam_m)
        self.adam_v = data.get("adam_v", self.adam_v)
        self.adam_t = data.get("adam_t", 0)
