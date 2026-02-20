"""Parameter Server for federated Q-RLSTC (FedAvg).

Aggregates gradient updates from N edge clients using
Federated Averaging. Maintains the global model and
distributes updates.

The tiny model size (80-224 bytes) makes this extraordinarily
efficient — aggregation of 1000 clients requires processing
only ~80KB of data.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple

from .serialization import (
    serialize_params,
    deserialize_gradient,
)


class ParameterServer:
    """Federated parameter server for Q-RLSTC.
    
    Implements FedAvg: weighted average of client gradients
    based on their local data size.
    
    Attributes:
        global_params: Current global model parameters.
        n_clients: Number of registered clients.
        round_number: Current federation round.
    """
    
    def __init__(
        self,
        n_params: int = 20,
        learning_rate: float = 1.0,
        seed: int = 42,
    ):
        """Initialize parameter server.
        
        Args:
            n_params: Number of model parameters.
            learning_rate: Server-side learning rate for aggregated gradient.
            seed: Random seed.
        """
        self.n_params = n_params
        self.lr = learning_rate
        
        rng = np.random.default_rng(seed)
        self.global_params = rng.uniform(-np.pi, np.pi, n_params)
        
        self.round_number = 0
        self._client_gradients: List[Tuple[np.ndarray, float]] = []
        self._client_weights: List[float] = []
    
    def get_global_params(self) -> bytes:
        """Get current global parameters for distribution.
        
        Returns:
            Serialized parameter vector.
        """
        return serialize_params(self.global_params)
    
    def receive_gradient(
        self,
        gradient_bytes: bytes,
        client_weight: float = 1.0,
    ) -> None:
        """Receive gradient update from one client.
        
        Args:
            gradient_bytes: Serialized gradient from edge client.
            client_weight: Weight for this client (e.g., data size).
        """
        gradient, scale = deserialize_gradient(gradient_bytes)
        self._client_gradients.append((gradient * scale, client_weight))
    
    def aggregate(self) -> np.ndarray:
        """Aggregate gradients using FedAvg and update global model.
        
        Returns:
            Updated global parameters.
        """
        if not self._client_gradients:
            return self.global_params
        
        # Weighted average of gradients
        total_weight = sum(w for _, w in self._client_gradients)
        if total_weight < 1e-10:
            total_weight = 1.0
        
        avg_gradient = np.zeros(self.n_params)
        for gradient, weight in self._client_gradients:
            # Ensure gradient matches param dimension
            g = gradient[:self.n_params]
            if len(g) < self.n_params:
                g = np.pad(g, (0, self.n_params - len(g)))
            avg_gradient += (weight / total_weight) * g
        
        # Apply aggregated update
        self.global_params += self.lr * avg_gradient
        
        # Clear buffers and advance round
        self._client_gradients.clear()
        self.round_number += 1
        
        return self.global_params
    
    @property
    def n_pending(self) -> int:
        """Number of pending gradient updates."""
        return len(self._client_gradients)
    
    def stats(self) -> Dict[str, float]:
        """Get server statistics.
        
        Returns:
            Dict with round number, param norm, etc.
        """
        return {
            "round": self.round_number,
            "n_pending": self.n_pending,
            "param_norm": float(np.linalg.norm(self.global_params)),
            "param_mean": float(np.mean(self.global_params)),
            "model_bytes": 4 + self.n_params * 4,
        }
