"""Classical Critic network for Hybrid Quantum Soft Actor-Critic (Q-SAC).

The critic absorbs the heavy mathematical lifting of estimating the
continuous value function V(s), while the quantum actor (VQ-DQN)
handles policy output π(a|s).

This is a lightweight 2-layer MLP with a simple gradient-based optimizer,
maintaining Q-RLSTC's philosophy of "classical where possible, quantum
where meaningful."
"""

import numpy as np
from typing import Optional, Tuple


class ClassicalCritic:
    """Two-layer MLP critic for value function estimation.
    
    Architecture:
        state → Linear(state_dim, hidden) → ReLU → Linear(hidden, 1) → V(s)
    
    Uses simple SGD with momentum for updates (no PyTorch dependency).
    
    Attributes:
        state_dim: Input state dimension.
        hidden_dim: Hidden layer size.
        lr: Learning rate.
    """
    
    def __init__(
        self,
        state_dim: int = 5,
        hidden_dim: int = 64,
        lr: float = 0.001,
        seed: int = 42,
    ):
        """Initialize critic network.
        
        Args:
            state_dim: State vector dimension.
            hidden_dim: Hidden layer neurons.
            lr: Learning rate for SGD.
            seed: Random seed.
        """
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        
        rng = np.random.default_rng(seed)
        
        # Xavier initialization
        scale1 = np.sqrt(2.0 / (state_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + 1))
        
        # Layer 1: state_dim → hidden_dim
        self.W1 = rng.normal(0, scale1, (state_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        
        # Layer 2: hidden_dim → 1
        self.W2 = rng.normal(0, scale2, (hidden_dim, 1))
        self.b2 = np.zeros(1)
        
        # Momentum buffers
        self._mW1 = np.zeros_like(self.W1)
        self._mb1 = np.zeros_like(self.b1)
        self._mW2 = np.zeros_like(self.W2)
        self._mb2 = np.zeros_like(self.b2)
        self._momentum = 0.9
    
    def predict(self, state: np.ndarray) -> float:
        """Predict V(s) for a single state.
        
        Args:
            state: State vector (state_dim,).
        
        Returns:
            Scalar value estimate.
        """
        state = np.asarray(state).flatten()[:self.state_dim]
        
        # Forward pass
        h = state @ self.W1 + self.b1
        h = np.maximum(h, 0)  # ReLU
        v = h @ self.W2 + self.b2
        
        return float(v[0])
    
    def predict_batch(self, states: np.ndarray) -> np.ndarray:
        """Predict V(s) for a batch of states.
        
        Args:
            states: Batch of states (batch_size, state_dim).
        
        Returns:
            Value estimates (batch_size,).
        """
        states = np.asarray(states)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        
        h = states @ self.W1 + self.b1
        h = np.maximum(h, 0)
        v = (h @ self.W2 + self.b2).flatten()
        
        return v
    
    def update(
        self,
        states: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """Update critic towards target values.
        
        Uses MSE loss with SGD+momentum.
        
        Args:
            states: Batch of states (batch_size, state_dim).
            targets: Target V(s) values (batch_size,).
        
        Returns:
            MSE loss.
        """
        states = np.asarray(states)
        targets = np.asarray(targets).flatten()
        
        if states.ndim == 1:
            states = states.reshape(1, -1)
        
        batch_size = len(states)
        
        # Forward pass with saved activations
        z1 = states @ self.W1 + self.b1          # (B, H)
        h1 = np.maximum(z1, 0)                    # ReLU
        v = (h1 @ self.W2 + self.b2).flatten()   # (B,)
        
        # Loss
        error = v - targets                        # (B,)
        loss = float(np.mean(error ** 2))
        
        # Backward pass
        dv = (2.0 / batch_size) * error            # (B,)
        
        # Layer 2 gradients
        dW2 = h1.T @ dv.reshape(-1, 1)            # (H, 1)
        db2 = np.sum(dv)                           # scalar
        
        # Layer 1 gradients (through ReLU)
        dh1 = dv.reshape(-1, 1) @ self.W2.T       # (B, H)
        dz1 = dh1 * (z1 > 0)                      # ReLU derivative
        
        dW1 = states.T @ dz1                       # (D, H)
        db1 = np.sum(dz1, axis=0)                  # (H,)
        
        # SGD with momentum
        self._mW1 = self._momentum * self._mW1 + self.lr * dW1
        self._mb1 = self._momentum * self._mb1 + self.lr * db1
        self._mW2 = self._momentum * self._mW2 + self.lr * dW2
        self._mb2 = self._momentum * self._mb2 + self.lr * db2
        
        self.W1 -= self._mW1
        self.b1 -= self._mb1
        self.W2 -= self._mW2
        self.b2 -= self._mb2
        
        return loss
    
    @property
    def n_params(self) -> int:
        """Total number of classical parameters."""
        return (
            self.state_dim * self.hidden_dim +
            self.hidden_dim +
            self.hidden_dim * 1 +
            1
        )
