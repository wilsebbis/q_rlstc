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
