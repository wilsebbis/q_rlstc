"""Edge Client for privacy-preserving federated Q-RLSTC.

Each edge client:
1. Holds raw GPS data locally (never transmitted)
2. Runs local feature extraction
3. Performs local VQ-DQN inference
4. Computes SPSA gradient on local data
5. Transmits only the gradient update (80-224 bytes)

Privacy guarantee: the parameter server never sees raw trajectories.
"""

import numpy as np
from typing import Optional, List, Tuple

from ..rl.spsa import SPSAOptimizer
from .serialization import serialize_gradient, deserialize_params


class EdgeClient:
    """Federated edge client for Q-RLSTC.
    
    Manages local model copy, local training data, and
    gradient computation for federated aggregation.
    
    Attributes:
        client_id: Unique identifier.
        params: Local copy of model parameters.
        n_local_steps: Local SGD steps per round.
    """
    
    def __init__(
        self,
        client_id: str,
        n_params: int = 20,
        n_local_steps: int = 5,
        seed: int = 42,
    ):
        """Initialize edge client.
        
        Args:
            client_id: Unique client identifier.
            n_params: Number of model parameters.
            n_local_steps: Local SPSA steps per federation round.
            seed: Random seed.
        """
        self.client_id = client_id
        self.n_params = n_params
        self.n_local_steps = n_local_steps
        
        rng = np.random.default_rng(seed)
        self.params = rng.uniform(-np.pi, np.pi, n_params)
        
        self.optimizer = SPSAOptimizer(
            A=10, a=0.08, c=0.06,
            use_momentum=True, momentum=0.9,
            seed=seed,
        )
        
        self._local_data: List[np.ndarray] = []
    
    def receive_global_params(self, param_bytes: bytes) -> None:
        """Receive updated global parameters from server.
        
        Args:
            param_bytes: Serialized parameter vector.
        """
        self.params = deserialize_params(param_bytes)
        self.optimizer.reset()
    
    def add_local_data(self, data: np.ndarray) -> None:
        """Add local training data (stays on device).
        
        Args:
            data: Local trajectory features or state-action pairs.
        """
        self._local_data.append(np.asarray(data))
    
    def compute_local_gradient(
        self,
        loss_fn,
    ) -> bytes:
        """Run local SPSA steps and return gradient update.
        
        The gradient is the only thing transmitted to the server.
        Raw GPS data stays on-device.
        
        Args:
            loss_fn: Loss function that takes params → scalar.
        
        Returns:
            Serialized gradient update (bytes).
        """
        initial_params = self.params.copy()
        
        # Run N local SPSA steps
        for _ in range(self.n_local_steps):
            self.params, _ = self.optimizer.step(loss_fn, self.params)
        
        # The effective gradient is the parameter delta
        gradient = self.params - initial_params
        
        return serialize_gradient(gradient, scale=1.0)
    
    @property
    def data_size(self) -> int:
        """Number of local data samples."""
        return len(self._local_data)
