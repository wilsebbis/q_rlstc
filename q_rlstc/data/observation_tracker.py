"""Online observation statistics tracking for stable reinforcement learning.

Uses Welford's online algorithm to maintain running mean and variance 
of observation features. This stabilizes training, particularly when driving
sensitive angle-encoded quantum gates (e.g. R_Y(arctan(x))) which saturate
under high-variance unnormalized inputs.
"""

import numpy as np

class ObservationTracker:
    """Maintains running statistics of vectors using Welford's algorithm."""
    
    def __init__(
        self, 
        feature_dim: int, 
        clip: float = 3.0, 
        epsilon: float = 1e-8,
        warmup_steps: int = 0
    ):
        """
        Args:
            feature_dim: Dimensions of the state vector.
            clip: Maximum standard deviations to clamp (e.g., +/- 3.0).
            epsilon: Numerical stability parameter.
            warmup_steps: Number of initial steps to accumulate stats before freezing.
                          If 0, stats update forever.
        """
        self.feature_dim = feature_dim
        self.clip = clip
        self.epsilon = epsilon
        self.warmup_steps = warmup_steps
        
        self.count = 0
        self.mean = np.zeros(feature_dim, dtype=np.float64)
        self.M2 = np.zeros(feature_dim, dtype=np.float64)
        
    def update(self, obs: np.ndarray) -> None:
        """Update running statistics with a new observation (or batch).
        
        Args:
            obs: Vector or batch matrix of shape (D,) or (B, D).
        """
        if self.warmup_steps > 0 and self.count >= self.warmup_steps:
            return  # Frozen after warmup
            
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
            
        for vec in obs:
            self.count += 1
            delta = vec - self.mean
            self.mean += delta / self.count
            delta2 = vec - self.mean
            self.M2 += delta * delta2
            
    def _variance(self) -> np.ndarray:
        if self.count < 2:
            return np.ones(self.feature_dim, dtype=np.float64)
        # Ensure non-negative variance due to float precision drift
        return np.maximum(self.M2 / (self.count - 1), 0.0)
        
    def normalize(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        """Normalize observation to ~N(0, 1) and optionally update trackers.
        
        Args:
            obs: Raw observation vector.
            update: Whether to update statistics with this observation.
            
        Returns:
            Z-score clamped observation.
        """
        if update:
            self.update(obs)
            
        if self.count < 2:
            # Not enough data, just clip
            return np.clip(obs, -self.clip, self.clip)
            
        var = self._variance()
        std = np.sqrt(var + self.epsilon)
        
        # Keep precision to float32 for downstream NN inputs but compute in float64
        z_scored = (obs - self.mean) / std
        return np.clip(z_scored, -self.clip, self.clip).astype(np.float32)
