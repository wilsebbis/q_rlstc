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
    q_values_batch,
    CircuitInfo,
)
from ..quantum.backends import get_backend
from .spsa import SPSAOptimizer
from .replay_buffer import ReplayBuffer


@dataclass
class AgentConfig:
    """Configuration for VQ-DQN agent.
    
    Attributes:
        version: "A" (5 qubits, standard) or "B" (8 qubits, multi-observable).
        n_qubits: Number of qubits (auto-set from version if not specified).
        n_layers: Number of variational layers.
        gamma: Discount factor.
        epsilon_start: Initial exploration rate.
        epsilon_min: Minimum exploration rate.
        epsilon_decay: Decay rate per episode.
        shots: Measurement shots.
        use_double_dqn: Whether to use Double DQN.
        target_update_freq: Episodes between target updates.
    """
    version: str = "A"
    n_qubits: int = 5
    n_layers: int = 2
    gamma: float = 0.90
    epsilon_start: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay: float = 0.99
    shots: int = 512
    use_double_dqn: bool = True
    target_update_freq: int = 10
    
    def __post_init__(self):
        """Auto-set n_qubits from version if still at default."""
        if self.version.upper() == "B" and self.n_qubits == 5:
            self.n_qubits = 8


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
        self.version = self.config.version.upper()
        
        # Readout mode based on version
        self.readout_mode = "multi_observable" if self.version == "B" else "standard"
        
        # Build circuit
        self.circuit_builder = VQDQNCircuitBuilder(
            n_qubits=self.config.n_qubits,
            n_layers=self.config.n_layers,
            use_data_reuploading=True,
        )
        
        # Initialize parameters
        self.n_circuit_params = self.circuit_builder.n_total_params
        circuit_params = self.rng.uniform(-np.pi, np.pi, self.n_circuit_params)
        
        # Learnable output affine head: Q(s,a) = ⟨Z_a⟩ * scale[a] + bias[a]
        # Scale=5.0 so Q-range is ~ [-5, 5], matching γ=0.99 TD targets.
        # Version B uses 4 scale weights (2 single + 2 parity).
        self._n_scale = 4 if self.version == "B" else 2
        init_scale = np.full(self._n_scale, 5.0)
        init_bias = np.zeros(2)
        
        # Concatenate [circuit_params | scale | bias] into one SPSA vector
        self.params = np.concatenate([circuit_params, init_scale, init_bias])
        self.target_params = self.params.copy()
        self.n_params = len(self.params)  # circuit + scale + bias
        
        # Exploration
        self.epsilon = self.config.epsilon_start
        
        # SPSA optimizer (optimizes all params: circuit + head)
        self.optimizer = SPSAOptimizer(seed=seed)
        
        # Statistics
        self.episode_count = 0
        self.training_step = 0
    
    # ── helpers to split the flat param vector ──────────────────
    
    def _split_params(self, params: np.ndarray):
        """Split [circuit | scale | bias] parameter vector."""
        circuit = params[:self.n_circuit_params]
        scale = params[self.n_circuit_params:self.n_circuit_params + self._n_scale]
        bias = params[self.n_circuit_params + self._n_scale:]
        return circuit, scale, bias
    
    @property
    def output_scale(self) -> np.ndarray:
        """Current learned scale (from online params)."""
        _, scale, _ = self._split_params(self.params)
        return scale
    
    @property
    def output_bias(self) -> np.ndarray:
        """Current learned bias (from online params)."""
        _, _, bias = self._split_params(self.params)
        return bias
    
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
        full_params = self.target_params if use_target else self.params
        circuit_p, scale, bias = self._split_params(full_params)
        
        q = evaluate_q_values(
            state=state,
            params=circuit_p,
            backend=self.backend,
            shots=self.config.shots,
            n_qubits=self.config.n_qubits,
            n_layers=self.config.n_layers,
            use_data_reuploading=True,
            output_scale=scale,
            output_bias=bias,
            readout_mode=self.readout_mode,
        )
        return np.clip(q, -10.0, 10.0)
    
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
    
    def compute_targets_batch(
        self,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        """Compute TD targets for an entire batch. 1-2 batched evals total.
        
        Args:
            rewards: (B,) rewards.
            next_states: (B, state_dim) next states.
            dones: (B,) done flags.
        
        Returns:
            (B,) TD targets.
        """
        B = len(rewards)
        targets = rewards.copy()
        
        # Find non-terminal transitions
        alive = ~dones.astype(bool)
        if not alive.any():
            return targets
        
        alive_next = next_states[alive]
        
        # Split online and target param vectors
        online_circ, online_scale, online_bias = self._split_params(self.params)
        target_circ, target_scale, target_bias = self._split_params(self.target_params)
        
        if self.config.use_double_dqn:
            # Double DQN: online selects, target evaluates — 2 batch evals
            q_online = q_values_batch(
                alive_next, online_circ,
                self.config.n_qubits, self.config.n_layers,
                output_scale=online_scale, output_bias=online_bias)
            q_target = q_values_batch(
                alive_next, target_circ,
                self.config.n_qubits, self.config.n_layers,
                output_scale=target_scale, output_bias=target_bias)
            best_actions = np.argmax(q_online, axis=1)
            next_values = q_target[np.arange(len(alive_next)), best_actions]
        else:
            # Standard DQN — 1 batch eval
            q_target = q_values_batch(
                alive_next, target_circ,
                self.config.n_qubits, self.config.n_layers,
                output_scale=target_scale, output_bias=target_bias)
            next_values = np.max(q_target, axis=1)
        
        targets[alive] += self.config.gamma * next_values
        return np.clip(targets, -10.0, 10.0)
    
    def _batch_loss(
        self,
        params: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """Batched Huber loss — ONE q_values_batch call per invocation.
        
        SPSA calls this twice (θ+δ, θ-δ) = 2 batch evals total.
        params is the full [circuit | scale | bias] vector.
        """
        circuit_p, scale, bias = self._split_params(params)
        q_all = q_values_batch(
            states, circuit_p,
            self.config.n_qubits, self.config.n_layers,
            output_scale=scale, output_bias=bias)
        
        B = len(states)
        preds = q_all[np.arange(B), actions.astype(int)]
        td_errors = targets - preds
        
        # Huber loss (vectorized)
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
        """Perform batched SPSA update.
        
        Total circuit evaluations per call:
        - compute_targets_batch: 1-2 q_values_batch calls (done ONCE)
        - SPSA step: 2 × _batch_loss = 2 q_values_batch calls
        Total: 3-4 batch evals instead of ~160 individual evals.
        """
        # Compute targets ONCE (outside SPSA loop)
        targets = self.compute_targets_batch(rewards, next_states, dones)
        
        # SPSA loss: only depends on params (targets are fixed)
        def loss_fn(params):
            return self._batch_loss(params, states, actions, targets)
        
        self.params, _ = self.optimizer.step(loss_fn, self.params)
        self.training_step += 1
        
        # No redundant 3rd loss eval — return 0 as placeholder
        return 0.0
    
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
        circuit_p, _, _ = self._split_params(self.params)
        return self.circuit_builder.get_circuit_info(circuit_p)
    
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
        # output_scale and output_bias are embedded in self.params
        # (read via @property accessors)
