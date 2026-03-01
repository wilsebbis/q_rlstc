"""Experience replay buffer for DQN training.

Stores transitions (s, a, r, s', done) and samples minibatches
for training the VQ-DQN agent.
"""

import numpy as np
from typing import List, Tuple, Optional, NamedTuple
from collections import deque


class Experience(NamedTuple):
    """A single experience tuple.
    
    Attributes:
        state: Current state.
        action: Action taken.
        reward: Reward received.
        next_state: Next state.
        done: Whether episode ended.
    """
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Circular replay buffer for experience storage.
    
    Supports prioritized sampling (uniform by default).
    """
    
    def __init__(
        self,
        max_size: int = 5000,
        seed: int = 42,
    ):
        """Initialize buffer.
        
        Args:
            max_size: Maximum number of experiences to store.
            seed: Random seed for sampling.
        """
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
        self.rng = np.random.default_rng(seed)
    
    def __len__(self) -> int:
        """Current number of experiences."""
        return len(self.buffer)
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add an experience to the buffer.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Resulting state.
            done: Whether episode ended.
        """
        experience = Experience(
            state=np.asarray(state),
            action=action,
            reward=reward,
            next_state=np.asarray(next_state),
            done=done,
        )
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a random minibatch of experiences.
        
        Args:
            batch_size: Number of experiences to sample.
        
        Returns:
            List of Experience tuples.
        
        Raises:
            ValueError: If batch_size exceeds buffer size.
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Not enough experiences: {len(self.buffer)} < {batch_size}"
            )
        
        indices = self.rng.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def sample_batch(self, batch_size: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Sample and return batch as numpy arrays.
        
        Args:
            batch_size: Number of experiences.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones).
        """
        batch = self.sample(batch_size)
        
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def sample_batch_stratified(
        self,
        batch_size: int,
        min_cut_quota: float = 0.3,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample batch with minimum CUT action quota.
        
        Guarantees at least `min_cut_quota` fraction of CUT (action=1)
        transitions in the batch, if enough exist in the buffer.
        Falls back to uniform sampling if insuffient CUT data.
        
        Args:
            batch_size: Number of experiences.
            min_cut_quota: Minimum fraction of CUT samples (0.0–1.0).
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones).
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Not enough experiences: {len(self.buffer)} < {batch_size}"
            )
        
        # Separate indices by action
        cut_indices = [i for i, e in enumerate(self.buffer) if e.action == 1]
        ext_indices = [i for i, e in enumerate(self.buffer) if e.action == 0]
        
        n_cut_needed = max(1, int(np.ceil(batch_size * min_cut_quota)))
        
        # Fallback: not enough CUT transitions → uniform sampling
        if len(cut_indices) < n_cut_needed:
            return self.sample_batch(batch_size)
        
        n_ext_needed = batch_size - n_cut_needed
        
        # If not enough EXTEND either, adjust
        if len(ext_indices) < n_ext_needed:
            n_ext_needed = len(ext_indices)
            n_cut_needed = batch_size - n_ext_needed
        
        chosen_cut = self.rng.choice(cut_indices, n_cut_needed, replace=False)
        chosen_ext = self.rng.choice(ext_indices, n_ext_needed, replace=False)
        indices = np.concatenate([chosen_cut, chosen_ext])
        self.rng.shuffle(indices)
        
        batch = [self.buffer[i] for i in indices]
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])
        
        return states, actions, rewards, next_states, dones

    def clear(self) -> None:
        """Clear all experiences from buffer."""
        self.buffer.clear()
    
    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough experiences.
        
        Args:
            min_size: Minimum required experiences.
        
        Returns:
            True if buffer has at least min_size experiences.
        """
        return len(self.buffer) >= min_size
