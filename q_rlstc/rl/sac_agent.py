"""Hybrid Quantum Soft Actor-Critic (Q-SAC) agent.

Replaces the DQN ε-greedy paradigm with a proper actor-critic architecture:
- Quantum Actor: VQ-DQN circuit outputs π(a|s) as a probability distribution
- Classical Critic: Lightweight MLP estimates V(s)
- Entropy Bonus: Agent is rewarded for keeping its options open

This eliminates the clunky ε-greedy exploration heuristic and prevents
the quantum agent from collapsing into degenerate policies.

Used as the default agent for Version C (Next-Gen Q-RNN).
"""

import numpy as np
from typing import Optional, Tuple

from ..quantum.vqdqn_circuit import (
    VQDQNCircuitBuilder,
    evaluate_q_values,
)
from ..quantum.backends import get_backend
from .spsa import SPSAOptimizer
from .critic import ClassicalCritic


class QSACAgent:
    """Hybrid Quantum Soft Actor-Critic agent.
    
    The quantum circuit acts as the policy network (actor),
    outputting action probabilities. The classical MLP critic
    estimates the value function for TD learning.
    
    Key difference from VQDQNAgent:
    - No epsilon-greedy: actions sampled from soft policy π(a|s)
    - Entropy bonus: reward += α * H(π(·|s))
    - Classical critic for stable value estimation
    - Continuous confidence outputs instead of discrete ε-coin flip
    
    Attributes:
        n_actions: Number of actions (2 or 3).
        entropy_alpha: Entropy bonus coefficient.
        gamma: Discount factor.
    """
    
    def __init__(
        self,
        state_dim: int = 5,
        n_actions: int = 2,
        n_qubits: int = 5,
        n_layers: int = 2,
        gamma: float = 0.90,
        entropy_alpha: float = 0.2,
        critic_lr: float = 0.001,
        critic_hidden: int = 64,
        shots: int = 512,
        backend=None,
        seed: int = 42,
    ):
        """Initialize Q-SAC agent.
        
        Args:
            state_dim: State vector dimension.
            n_actions: Action space size.
            n_qubits: Qubits for quantum actor.
            n_layers: Variational layers.
            gamma: Discount factor.
            entropy_alpha: Entropy regularization coefficient.
            critic_lr: Learning rate for classical critic.
            critic_hidden: Hidden dimension for critic MLP.
            shots: Measurement shots for actor circuit.
            backend: Qiskit backend.
            seed: Random seed.
        """
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.entropy_alpha = entropy_alpha
        self.shots = shots
        self.seed = seed
        
        self.rng = np.random.default_rng(seed)
        
        # Quantum Actor (VQ-DQN circuit → action probabilities)
        self.circuit_builder = VQDQNCircuitBuilder(
            n_qubits=n_qubits,
            n_layers=n_layers,
            use_data_reuploading=True,
        )
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend = backend or get_backend(mode="ideal")
        
        # Actor parameters
        self.actor_params = self.rng.uniform(
            -np.pi, np.pi, self.circuit_builder.n_params
        )
        self.target_params = self.actor_params.copy()
        
        # Actor optimizer (SPSA with momentum)
        self.actor_optimizer = SPSAOptimizer(
            A=20, a=0.12, c=0.10,
            use_momentum=True, momentum=0.9,
            seed=seed,
        )
        
        # Classical Critic
        self.critic = ClassicalCritic(
            state_dim=state_dim,
            hidden_dim=critic_hidden,
            lr=critic_lr,
            seed=seed,
        )
        
        # Statistics
        self.update_count = 0
        self.target_update_freq = 10
    
    def _get_action_probs(
        self,
        state: np.ndarray,
        params: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Get action probability distribution from quantum actor.
        
        Converts raw Q-values to probabilities via softmax.
        
        Args:
            state: State vector.
            params: Actor parameters (uses self.actor_params if None).
        
        Returns:
            Action probabilities (n_actions,).
        """
        if params is None:
            params = self.actor_params
        
        q_values = evaluate_q_values(
            state=state[:self.n_qubits],
            params=params,
            backend=self.backend,
            shots=self.shots,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            use_data_reuploading=True,
        )
        
        # For 3-action: use first 3 qubits' expectations
        q = q_values[:self.n_actions]
        
        # Softmax with temperature (entropy_alpha controls exploration)
        temperature = max(self.entropy_alpha, 0.01)
        q_scaled = q / temperature
        q_scaled -= np.max(q_scaled)  # numerical stability
        exp_q = np.exp(q_scaled)
        probs = exp_q / np.sum(exp_q)
        
        return probs
    
    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        """Select action by sampling from soft policy.
        
        Unlike DQN's ε-greedy, this always uses the learned distribution.
        
        Args:
            state: Current state.
            greedy: If True, take argmax instead of sampling.
        
        Returns:
            Action index.
        """
        probs = self._get_action_probs(state)
        
        if greedy:
            return int(np.argmax(probs))
        
        return int(self.rng.choice(self.n_actions, p=probs))
    
    def compute_target(
        self,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        """Compute TD target with entropy bonus.
        
        target = r + γ * (V(s') + α * H(π(·|s')))
        
        Args:
            reward: Immediate reward.
            next_state: Next state.
            done: Whether episode ended.
        
        Returns:
            TD target value.
        """
        if done:
            return reward
        
        # Value from critic
        v_next = self.critic.predict(next_state)
        
        # Entropy bonus from actor
        probs = self._get_action_probs(next_state, self.target_params)
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        
        return reward + self.gamma * (v_next + self.entropy_alpha * entropy)
    
    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """Update both actor and critic.
        
        Args:
            states: Batch of states.
            actions: Batch of actions.
            targets: Batch of TD targets.
        
        Returns:
            Combined loss.
        """
        # Update critic
        critic_loss = self.critic.update(states, targets)
        
        # Update actor via SPSA
        def actor_loss(params):
            total = 0.0
            for i in range(len(states)):
                probs = self._get_action_probs(states[i], params)
                action = int(actions[i])
                
                # Policy gradient: -log(π(a|s)) * advantage
                v = self.critic.predict(states[i])
                advantage = targets[i] - v
                log_prob = np.log(probs[action] + 1e-8)
                
                # Entropy bonus
                entropy = -np.sum(probs * np.log(probs + 1e-8))
                
                total += -(log_prob * advantage + self.entropy_alpha * entropy)
            
            return total / len(states)
        
        self.actor_params, _ = self.actor_optimizer.step(
            actor_loss, self.actor_params
        )
        
        # Target network update
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_params = self.actor_params.copy()
        
        return critic_loss
    
    def get_circuit_info(self):
        """Get quantum actor circuit info."""
        return self.circuit_builder.get_circuit_info(self.actor_params)
    
    @property
    def readout_mode(self) -> str:
        """Readout mode label."""
        return "sac_soft_policy"
    
    def save_checkpoint(self, path: str) -> None:
        """Save agent state."""
        np.savez(
            path,
            actor_params=self.actor_params,
            target_params=self.target_params,
            critic_W1=self.critic.W1,
            critic_b1=self.critic.b1,
            critic_W2=self.critic.W2,
            critic_b2=self.critic.b2,
        )
    
    def load_checkpoint(self, path: str) -> None:
        """Load agent state."""
        data = np.load(path)
        self.actor_params = data['actor_params']
        self.target_params = data['target_params']
        self.critic.W1 = data['critic_W1']
        self.critic.b1 = data['critic_b1']
        self.critic.W2 = data['critic_W2']
        self.critic.b2 = data['critic_b2']
