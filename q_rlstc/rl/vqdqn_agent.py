"""VQ-DQN agent wrapper for RL training.

Combines:
- VQ-DQN circuit for Q-value computation
- Epsilon-greedy action selection
- SPSA parameter updates
- Target network for stable training
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field

from qiskit_aer import AerSimulator

from ..quantum.vqdqn_circuit import (
    VQDQNCircuitBuilder,
    evaluate_q_values,
    CircuitInfo,
)
from ..quantum.backends import get_backend
from .spsa import SPSAOptimizer
from .replay_buffer import ReplayBuffer


@dataclass
class AgentConfig:
    """Configuration for VQ-DQN agent.
    
    Attributes:
        n_qubits: Number of qubits.
        n_layers: Number of variational layers.
        gamma: Discount factor.
        epsilon_start: Initial exploration rate.
        epsilon_min: Minimum exploration rate.
        epsilon_decay: Decay rate per episode.
        shots: Measurement shots.
        use_double_dqn: Whether to use Double DQN.
        target_update_freq: Episodes between target updates.
    """
    n_qubits: int = 5
    n_layers: int = 2
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay: float = 0.99
    shots: int = 512
    use_double_dqn: bool = True
    target_update_freq: int = 10


class VQDQNAgent:
    """Variational Quantum Deep Q-Network agent.
    
    Uses a quantum circuit as the Q-function approximator,
    trained with SPSA optimization.
    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        backend: Optional[AerSimulator] = None,
        seed: int = 42,
    ):
        """Initialize VQ-DQN agent.
        
        Args:
            config: Agent configuration.
            backend: Qiskit backend for circuit execution.
            seed: Random seed.
        """
        self.config = config or AgentConfig()
        self.backend = backend or get_backend("ideal")
        self.rng = np.random.default_rng(seed)
        
        # Build circuit
        self.circuit_builder = VQDQNCircuitBuilder(
            n_qubits=self.config.n_qubits,
            n_layers=self.config.n_layers,
            use_data_reuploading=True,
        )
        
        # Initialize parameters
        self.n_params = self.circuit_builder.n_total_params
        self.params = self.rng.uniform(-np.pi, np.pi, self.n_params)
        self.target_params = self.params.copy()
        
        # Output scaling
        self.output_scale = np.ones(2)
        self.output_bias = np.zeros(2)
        
        # Exploration
        self.epsilon = self.config.epsilon_start
        
        # SPSA optimizer
        self.optimizer = SPSAOptimizer(seed=seed)
        
        # Statistics
        self.episode_count = 0
        self.training_step = 0
    
    def get_q_values(
        self,
        state: np.ndarray,
        use_target: bool = False,
    ) -> np.ndarray:
        """Compute Q-values for a state.
        
        Args:
            state: State vector.
            use_target: Whether to use target network.
        
        Returns:
            Q-values [Q(s, extend), Q(s, cut)].
        """
        params = self.target_params if use_target else self.params
        
        return evaluate_q_values(
            state=state,
            params=params,
            backend=self.backend,
            shots=self.config.shots,
            n_qubits=self.config.n_qubits,
            n_layers=self.config.n_layers,
            use_data_reuploading=True,
            output_scale=self.output_scale,
            output_bias=self.output_bias,
        )
    
    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current state.
            greedy: If True, ignore epsilon and act greedily.
        
        Returns:
            Action (0 = extend, 1 = cut).
        """
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(2))
        
        q_values = self.get_q_values(state)
        return int(np.argmax(q_values))
    
    def compute_target(
        self,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        """Compute TD target for update.
        
        Uses Double DQN if configured: uses online network to select
        best action, target network to evaluate it.
        
        Args:
            reward: Immediate reward.
            next_state: Next state.
            done: Whether episode ended.
        
        Returns:
            TD target value.
        """
        if done:
            return reward
        
        if self.config.use_double_dqn:
            # Online network selects best action
            online_q = self.get_q_values(next_state, use_target=False)
            best_action = int(np.argmax(online_q))
            
            # Target network evaluates
            target_q = self.get_q_values(next_state, use_target=True)
            next_value = target_q[best_action]
        else:
            # Standard DQN
            target_q = self.get_q_values(next_state, use_target=True)
            next_value = np.max(target_q)
        
        return reward + self.config.gamma * next_value
    
    def _compute_loss(
        self,
        params: np.ndarray,
        state: np.ndarray,
        action: int,
        target: float,
    ) -> float:
        """Compute TD loss for SPSA.
        
        Args:
            params: Parameters to evaluate.
            state: State.
            action: Action taken.
            target: TD target.
        
        Returns:
            Squared TD error.
        """
        q_values = evaluate_q_values(
            state=state,
            params=params,
            backend=self.backend,
            shots=self.config.shots,
            n_qubits=self.config.n_qubits,
            n_layers=self.config.n_layers,
            use_data_reuploading=True,
            output_scale=self.output_scale,
            output_bias=self.output_bias,
        )
        
        q_value = q_values[action]
        td_error = target - q_value
        
        return td_error ** 2
    
    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """Perform SPSA update on parameters.
        
        Args:
            states: Batch of states.
            actions: Batch of actions.
            targets: Batch of TD targets.
        
        Returns:
            Average loss.
        """
        def batch_loss(params):
            total_loss = 0.0
            for state, action, target in zip(states, actions, targets):
                total_loss += self._compute_loss(params, state, int(action), target)
            return total_loss / len(states)
        
        self.params, _ = self.optimizer.step(batch_loss, self.params)
        self.training_step += 1
        
        return batch_loss(self.params)
    
    def update_target_network(self) -> None:
        """Copy online parameters to target network."""
        self.target_params = self.params.copy()
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay
        )
        self.episode_count += 1
        
        # Update target network periodically
        if self.episode_count % self.config.target_update_freq == 0:
            self.update_target_network()
    
    def get_circuit_info(self) -> CircuitInfo:
        """Get information about the VQ-DQN circuit."""
        return self.circuit_builder.get_circuit_info(self.params)
    
    def save_checkpoint(self, path: str) -> None:
        """Save agent state to file."""
        np.savez(
            path,
            params=self.params,
            target_params=self.target_params,
            epsilon=self.epsilon,
            episode_count=self.episode_count,
            training_step=self.training_step,
            output_scale=self.output_scale,
            output_bias=self.output_bias,
        )
    
    def load_checkpoint(self, path: str) -> None:
        """Load agent state from file."""
        data = np.load(path)
        self.params = data['params']
        self.target_params = data['target_params']
        self.epsilon = float(data['epsilon'])
        self.episode_count = int(data['episode_count'])
        self.training_step = int(data['training_step'])
        self.output_scale = data['output_scale']
        self.output_bias = data['output_bias']
