"""VQ-DQN agent wrapper for RL training.

Combines:
- VQ-DQN circuit for Q-value computation
- Epsilon-greedy action selection
- SPSA parameter updates
- Target network for stable training
"""

import warnings

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
        version: "A" (5q, standard), "B" (8q, multi-observable),
                 "D" (5q, standard), "E" (5q, Quantum B: input scaling + anti-BP init).
        n_qubits: Number of qubits (auto-set from version if not specified).
        n_layers: Number of variational layers.
        gamma: Discount factor.
        epsilon_start: Initial exploration rate.
        epsilon_min: Minimum exploration rate.
        epsilon_decay: Decay rate per episode.
        shots: Measurement shots.
        use_double_dqn: Whether to use Double DQN.
        target_update_freq: Episodes between target updates.
        exploration_mode: "epsilon_greedy" or "boltzmann".
        boltzmann_temp: Initial Boltzmann temperature (higher → more random).
        boltzmann_temp_min: Minimum temperature.
        boltzmann_temp_decay: Multiplicative decay per episode.
        q_clip_range: Symmetric Q-value clipping bound.
        optimistic_cut_bias: Extra initial bias for CUT action (breaks symmetry).
        use_input_scaling: If True, learn per-feature scale+shift before encoding.
        anti_barren_plateau: If True, init circuit params near zero.
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
    entanglement: str = "linear"  # 'linear', 'circular', 'full', 'none'
    exploration_mode: str = "epsilon_greedy"  # "epsilon_greedy" or "boltzmann"
    boltzmann_temp: float = 1.0
    boltzmann_temp_min: float = 0.1
    boltzmann_temp_decay: float = 0.99
    q_clip_range: float = 50.0
    optimistic_cut_bias: float = 0.0  # extra initial bias for Q(cut)
    use_input_scaling: bool = False   # learnable per-feature scale+shift
    anti_barren_plateau: bool = False # near-zero circuit param init
    use_soft_targets: bool = False    # entropy-regularized targets (soft-DQN)
    soft_alpha: float = 0.1           # entropy temperature for soft targets
    adaptive_shots: bool = False      # dynamically scale shots via Hoeffding bounds
    confidence_delta: float = 0.05    # confidence bound for adaptive shots
    
    def __post_init__(self):
        """Auto-set fields from version."""
        v = self.version.upper()
        if v == "B" and self.n_qubits == 5:
            self.n_qubits = 8
        if v == "E":
            # Quantum B defaults: input scaling + anti-BP + circular entanglement
            self.use_input_scaling = True
            self.anti_barren_plateau = True
            if self.entanglement == "linear":
                self.entanglement = "circular"


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
            entanglement=self.config.entanglement,
        )
        
        # Initialize parameters
        self.n_circuit_params = self.circuit_builder.n_total_params
        
        # Anti-barren-plateau: init circuit params near zero
        if self.config.anti_barren_plateau:
            circuit_params = self.rng.uniform(-0.01, 0.01, self.n_circuit_params)
        else:
            circuit_params = self.rng.uniform(-np.pi, np.pi, self.n_circuit_params)
        
        # Learnable input scaling: x_scaled = input_scale * x + input_shift
        # Initialized: scale=1.0, shift=0.0 (identity transform)
        self._use_input_scaling = self.config.use_input_scaling
        self._n_input_params = 2 * self.config.n_qubits if self._use_input_scaling else 0
        if self._use_input_scaling:
            input_scale = np.ones(self.config.n_qubits)   # scale per feature
            input_shift = np.zeros(self.config.n_qubits)  # shift per feature
        
        # Learnable output affine head: Q(s,a) = ⟨Z_a⟩ * scale[a] + bias[a]
        # Scale=5.0 so Q-range is ~ [-5, 5], matching γ=0.99 TD targets.
        # Version B/E uses 4 scale weights (2 single + 2 parity).
        self._n_scale = 4 if self.version in ("B", "E") else 2
        init_scale = np.full(self._n_scale, 5.0)
        init_bias = np.zeros(2)
        # Fix 4: Optimistic CUT init — break never-cut attractor
        if self.config.optimistic_cut_bias != 0.0:
            init_bias[1] = self.config.optimistic_cut_bias
        
        # Concatenate into one SPSA vector:
        # [input_scale? | input_shift? | circuit_params | output_scale | output_bias]
        parts = []
        if self._use_input_scaling:
            parts.extend([input_scale, input_shift])
        parts.extend([circuit_params, init_scale, init_bias])
        self.params = np.concatenate(parts)
        self.target_params = self.params.copy()
        self.n_params = len(self.params)
        
        # Exploration
        self.epsilon = self.config.epsilon_start
        self.boltzmann_temp = self.config.boltzmann_temp
        
        # SPSA optimizer (optimizes all params: input_scaling + circuit + head)
        self.optimizer = SPSAOptimizer(seed=seed, use_momentum=True, momentum=0.9)
        
        # Statistics
        self.episode_count = 0
        self.training_step = 0
    
    # ── helpers to split the flat param vector ──────────────────
    
    def _split_params(self, params: np.ndarray):
        """Split param vector into components.
        
        Layout: [input_scale? | input_shift? | circuit | output_scale | output_bias]
        """
        offset = 0
        if self._use_input_scaling:
            nq = self.config.n_qubits
            in_scale = params[offset:offset + nq]
            offset += nq
            in_shift = params[offset:offset + nq]
            offset += nq
        else:
            in_scale = None
            in_shift = None
        
        circuit = params[offset:offset + self.n_circuit_params]
        offset += self.n_circuit_params
        scale = params[offset:offset + self._n_scale]
        offset += self._n_scale
        bias = params[offset:]
        return in_scale, in_shift, circuit, scale, bias
    
    def _scale_features(self, state: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Apply learnable input scaling: x_scaled = in_scale * x + in_shift.
        
        If input scaling is disabled, returns state unchanged.
        """
        if not self._use_input_scaling:
            return state
        in_scale, in_shift, _, _, _ = self._split_params(params)
        flat = np.asarray(state).flatten()
        return flat * in_scale + in_shift
    
    @property
    def output_scale(self) -> np.ndarray:
        """Current learned scale (from online params)."""
        _, _, _, scale, _ = self._split_params(self.params)
        return scale
    
    @property
    def output_bias(self) -> np.ndarray:
        """Current learned bias (from online params)."""
        _, _, _, _, bias = self._split_params(self.params)
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
        _, _, circuit_p, scale, bias = self._split_params(full_params)
        
        # Apply learnable input scaling before circuit evaluation
        scaled_state = self._scale_features(state, full_params)
        
        q = evaluate_q_values(
            state=scaled_state,
            params=circuit_p,
            backend=self.backend,
            shots=self.config.shots,
            n_qubits=self.config.n_qubits,
            n_layers=self.config.n_layers,
            use_data_reuploading=True,
            output_scale=scale,
            output_bias=bias,
            readout_mode=self.readout_mode,
            entanglement=self.config.entanglement,
            adaptive_shots=self.config.adaptive_shots,
            confidence_delta=self.config.confidence_delta,
        )
        # Fix 5: NaN guard + widened clip
        if not np.all(np.isfinite(q)):
            warnings.warn(f"Non-finite Q-values detected: {q}. Replacing with 0.")
            q = np.nan_to_num(q, nan=0.0, posinf=self.config.q_clip_range,
                             neginf=-self.config.q_clip_range)
        return np.clip(q, -self.config.q_clip_range, self.config.q_clip_range)
    
    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        """Select action using epsilon-greedy or Boltzmann policy.
        
        Args:
            state: Current state.
            greedy: If True, ignore exploration and act greedily.
        
        Returns:
            Action (0 = extend, 1 = cut).
        """
        q_values = self.get_q_values(state)
        
        if greedy:
            return int(np.argmax(q_values))
        
        if self.config.exploration_mode == "boltzmann":
            # Boltzmann / softmax exploration
            tau = max(self.boltzmann_temp, 1e-8)
            logits = q_values / tau
            logits -= np.max(logits)  # numerical stability
            probs = np.exp(logits)
            probs /= probs.sum()
            return int(self.rng.choice(2, p=probs))
        else:
            # ε-greedy exploration
            if self.rng.random() < self.epsilon:
                return int(self.rng.integers(2))
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
        _, _, online_circ, online_scale, online_bias = self._split_params(self.params)
        _, _, target_circ, target_scale, target_bias = self._split_params(self.target_params)
        
        # Apply input scaling to next_states for batch evaluation
        if self._use_input_scaling:
            in_s, in_sh, _, _, _ = self._split_params(self.params)
            alive_next_scaled = alive_next * in_s + in_sh
            t_in_s, t_in_sh, _, _, _ = self._split_params(self.target_params)
            alive_next_target_scaled = alive_next * t_in_s + t_in_sh
        else:
            alive_next_scaled = alive_next
            alive_next_target_scaled = alive_next
        
        if self.config.use_double_dqn:
            # Double DQN: online selects, target evaluates — 2 batch evals
            q_online = q_values_batch(
                alive_next_scaled, online_circ,
                self.config.n_qubits, self.config.n_layers,
                output_scale=online_scale, output_bias=online_bias,
                entanglement=self.config.entanglement)
            q_target = q_values_batch(
                alive_next_target_scaled, target_circ,
                self.config.n_qubits, self.config.n_layers,
                output_scale=target_scale, output_bias=target_bias,
                entanglement=self.config.entanglement)
            best_actions = np.argmax(q_online, axis=1)
            next_values = q_target[np.arange(len(alive_next)), best_actions]
        else:
            # Standard DQN — 1 batch eval
            q_target = q_values_batch(
                alive_next_target_scaled, target_circ,
                self.config.n_qubits, self.config.n_layers,
                output_scale=target_scale, output_bias=target_bias,
                entanglement=self.config.entanglement)
            next_values = np.max(q_target, axis=1)
        
        # Soft-DQN: optionally use entropy-regularized targets
        if self.config.use_soft_targets:
            from q_rlstc.rl.soft_targets import soft_value
            # Override with soft value (q_target already computed above)
            next_values = soft_value(q_target, alpha=self.config.soft_alpha)
        
        targets[alive] += self.config.gamma * next_values
        # Fix 5: widened target clip
        return np.clip(targets, -self.config.q_clip_range, self.config.q_clip_range)
    
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
        _, _, circuit_p, scale, bias = self._split_params(params)
        
        # Apply input scaling for batch evaluation
        if self._use_input_scaling:
            nq = self.config.n_qubits
            in_s = params[:nq]
            in_sh = params[nq:2*nq]
            states_scaled = states * in_s + in_sh
        else:
            states_scaled = states
        
        q_all = q_values_batch(
            states_scaled, circuit_p,
            self.config.n_qubits, self.config.n_layers,
            output_scale=scale, output_bias=bias,
            entanglement=self.config.entanglement)
        
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
        """Decay exploration rate (epsilon and/or Boltzmann temperature)."""
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay
        )
        # Also decay Boltzmann temperature
        self.boltzmann_temp = max(
            self.config.boltzmann_temp_min,
            self.boltzmann_temp * self.config.boltzmann_temp_decay
        )
        self.episode_count += 1
        
        # Update target network periodically
        if self.episode_count % self.config.target_update_freq == 0:
            self.update_target_network()
    
    def get_circuit_info(self) -> CircuitInfo:
        """Get information about the VQ-DQN circuit."""
        _, _, circuit_p, _, _ = self._split_params(self.params)
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
