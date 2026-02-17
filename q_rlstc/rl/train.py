"""Episodic training loop for Q-RLSTC.

Implements the RL training procedure where:
1. Agent traverses trajectories, making extend/cut decisions
2. Rewards based on clustering quality improvement
3. SPSA updates on minibatch from replay buffer
4. Periodic cluster center updates
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import time

from ..config import QRLSTCConfig
from ..data.synthetic import Trajectory, SyntheticDataset
from ..data.features import StateFeatureExtractor
from ..clustering.metrics import overall_distance, od_improvement_reward, segmentation_f1
from ..quantum.backends import get_backend

from .replay_buffer import ReplayBuffer
from .vqdqn_agent import VQDQNAgent, AgentConfig


@dataclass
class EpisodeResult:
    """Result from a single episode (trajectory).
    
    Attributes:
        trajectory_id: ID of the trajectory.
        n_segments: Number of segments created.
        boundaries: Predicted segment boundaries.
        total_reward: Sum of rewards.
        final_od: Final overall distance.
    """
    trajectory_id: int
    n_segments: int
    boundaries: List[int]
    total_reward: float
    final_od: float


@dataclass
class TrainingResult:
    """Result from full training run.
    
    Attributes:
        n_epochs: Number of epochs completed.
        n_episodes: Total episodes.
        final_od: Final overall distance.
        final_f1: Final segmentation F1.
        episode_rewards: Rewards per episode.
        od_history: OD after each epoch.
        runtime_seconds: Total training time.
    """
    n_epochs: int
    n_episodes: int
    final_od: float
    final_f1: float
    episode_rewards: List[float]
    od_history: List[float]
    runtime_seconds: float


class MDPEnvironment:
    """MDP environment for sub-trajectory segmentation.
    
    State: 5-dimensional feature vector
        [segment_length, local_variance, centroid_distance, 
         trajectory_progress, segment_count]
    
    Actions: 
        0 = EXTEND (add next point to current segment)
        1 = CUT (end current segment, start new one)
    
    Termination:
        - End of trajectory (all points consumed)
        - Max segments exceeded
    
    Anti-Gaming Constraints:
        - Minimum segment length: CUT disallowed if < MIN_SEGMENT_LEN
        - Max segments: Episode terminates early if exceeded
        - Segment penalty: Reward -= λ per segment to prevent over-segmentation
    """
    
    # Anti-gaming constants
    MIN_SEGMENT_LEN = 3   # Minimum points before CUT is allowed
    MAX_SEGMENTS = 50     # Force termination if exceeded
    SEGMENT_PENALTY = 0.1 # λ: penalty per new segment
    
    def __init__(
        self,
        trajectory: Trajectory,
        feature_extractor: StateFeatureExtractor,
        n_clusters: int = 10,
    ):
        """Initialize environment.
        
        Args:
            trajectory: Current trajectory to segment.
            feature_extractor: Feature extractor.
            n_clusters: Number of clusters for OD.
        """
        self.trajectory = trajectory
        self.feature_extractor = feature_extractor
        self.n_clusters = n_clusters
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment for new episode.
        
        Returns:
            Initial state.
        """
        self.current_idx = 0
        self.split_point = 0
        self.segments: List[Tuple[int, int]] = []
        self.current_od = 0.0
        self.local_variance = 0.0
        self.n_segments = 0
        self.boundaries: List[int] = []
        self.done = False
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state features."""
        return self.feature_extractor.extract_features(
            self.trajectory,
            self.current_idx,
            self.split_point,
            self.current_od,
            self.n_segments,
        )
    
    def _compute_segment_cost(self, start: int, end: int) -> float:
        """Compute cost (avg distance to centroid) for a segment."""
        points = self.trajectory.points[start:end + 1]
        if len(points) < 1:
            return 0.0
        
        coords = np.array([[p.x, p.y] for p in points])
        centroid = coords.mean(axis=0)
        distances = np.linalg.norm(coords - centroid, axis=1)
        return float(distances.mean())
    
    def _compute_local_variance(self, start: int, end: int) -> float:
        """Compute variance within current segment."""
        points = self.trajectory.points[start:end + 1]
        if len(points) < 2:
            return 0.0
        
        coords = np.array([[p.x, p.y] for p in points])
        return float(np.var(coords))
    
    def _compute_boundary_sharpness(self, boundary_idx: int) -> float:
        """Compute how 'sharp' a boundary is (direction change)."""
        if boundary_idx < 1 or boundary_idx >= len(self.trajectory.points) - 1:
            return 0.0
        
        points = self.trajectory.points
        # Vector before boundary
        v1 = np.array([
            points[boundary_idx].x - points[boundary_idx - 1].x,
            points[boundary_idx].y - points[boundary_idx - 1].y
        ])
        # Vector after boundary
        v2 = np.array([
            points[boundary_idx + 1].x - points[boundary_idx].x,
            points[boundary_idx + 1].y - points[boundary_idx].y
        ])
        
        # Angle between vectors (higher = sharper turn = better boundary)
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)  # [0, π]
        
        # Normalize to [0, 1], higher angle = better boundary
        return float(angle / np.pi)
    
    def _is_cut_allowed(self) -> bool:
        """Check if CUT action is allowed (anti-gaming)."""
        current_segment_len = self.current_idx - self.split_point + 1
        return current_segment_len >= self.MIN_SEGMENT_LEN
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Execute action and return (next_state, reward, done).
        
        Action 0: EXTEND - add next point to current segment
        Action 1: CUT - end current segment, start new one
        
        Args:
            action: 0 or 1.
        
        Returns:
            Tuple of (next_state, reward, done).
        """
        if self.done:
            return self._get_state(), 0.0, True
        
        old_variance = self.local_variance
        reward = 0.0
        
        # Anti-gaming: enforce minimum segment length
        if action == 1 and not self._is_cut_allowed():
            action = 0  # Force EXTEND if segment too short
        
        if action == 1:  # CUT
            # End current segment
            segment = (self.split_point, self.current_idx)
            self.segments.append(segment)
            self.boundaries.append(self.current_idx)
            
            # Update OD incrementally
            seg_cost = self._compute_segment_cost(self.split_point, self.current_idx)
            if self.n_segments == 0:
                self.current_od = seg_cost
            else:
                self.current_od = (
                    (self.current_od * self.n_segments + seg_cost) /
                    (self.n_segments + 1)
                )
            self.n_segments += 1
            
            # Reward components for CUT
            # 1. Boundary sharpness bonus
            boundary_score = self._compute_boundary_sharpness(self.current_idx)
            reward += boundary_score * 0.5
            
            # 2. Segment penalty (anti-gaming)
            reward -= self.SEGMENT_PENALTY
            
            # Start new segment
            self.split_point = self.current_idx + 1
            self.local_variance = 0.0
        else:
            # EXTEND: update local variance
            self.local_variance = self._compute_local_variance(
                self.split_point, self.current_idx
            )
        
        # Move to next point
        self.current_idx += 1
        
        # Local delta reward: variance improvement
        new_variance = self._compute_local_variance(
            self.split_point, min(self.current_idx, len(self.trajectory.points) - 1)
        )
        variance_delta = old_variance - new_variance
        reward += variance_delta * 0.1  # Small weight for local signal
        
        # Check termination conditions
        if self.current_idx >= len(self.trajectory.points) - 1:
            # Force final segment
            if self.split_point < len(self.trajectory.points):
                segment = (self.split_point, len(self.trajectory.points) - 1)
                self.segments.append(segment)
                seg_cost = self._compute_segment_cost(
                    self.split_point, len(self.trajectory.points) - 1
                )
                if self.n_segments > 0:
                    self.current_od = (
                        (self.current_od * self.n_segments + seg_cost) /
                        (self.n_segments + 1)
                    )
                else:
                    self.current_od = seg_cost
                self.n_segments += 1
            self.done = True
        
        # Anti-gaming: terminate if too many segments
        if self.n_segments >= self.MAX_SEGMENTS:
            self.done = True
        
        self.local_variance = new_variance
        
        return self._get_state(), reward, self.done


class Trainer:
    """Trainer for Q-RLSTC.
    
    Orchestrates training loop, replay buffer, and evaluation.
    """
    
    def __init__(
        self,
        dataset: SyntheticDataset,
        config: Optional[QRLSTCConfig] = None,
    ):
        """Initialize trainer.
        
        Args:
            dataset: Training dataset.
            config: Configuration.
        """
        self.dataset = dataset
        self.config = config or QRLSTCConfig()
        
        # Determine version
        self.version = self.config.version.upper()
        
        # Create agent with version-aware config
        agent_config = AgentConfig(
            version=self.version,
            n_qubits=self.config.vqdqn.n_qubits,
            n_layers=self.config.vqdqn.n_layers,
            gamma=self.config.rl.gamma,
            epsilon_start=self.config.rl.epsilon_start,
            epsilon_min=self.config.rl.epsilon_min,
            epsilon_decay=self.config.rl.epsilon_decay,
            shots=self.config.vqdqn.shots_train,
            target_update_freq=self.config.rl.target_update_freq,
        )
        
        # Get backend
        backend = get_backend(
            mode="noisy_sim" if self.config.noise.use_noise else "ideal",
            noise_model_name=self.config.noise.noise_model,
        )
        
        self.agent = VQDQNAgent(config=agent_config, backend=backend)
        
        # Replay buffer
        self.buffer = ReplayBuffer(
            max_size=self.config.rl.memory_size,
        )
        
        # Feature extractor — Version B uses 8D features
        if self.version == "B":
            from ..data.features import StateFeatureExtractorB
            self.feature_extractor = StateFeatureExtractorB(
                n_clusters=self.config.clustering.n_clusters,
            )
        else:
            self.feature_extractor = StateFeatureExtractor(
                n_clusters=self.config.clustering.n_clusters,
            )
        
        # Statistics
        self.episode_rewards: List[float] = []
        self.od_history: List[float] = []
    
    def _run_episode(self, trajectory: Trajectory) -> EpisodeResult:
        """Run one episode (trajectory).
        
        Args:
            trajectory: Trajectory to segment.
        
        Returns:
            EpisodeResult.
        """
        env = MDPEnvironment(
            trajectory,
            self.feature_extractor,
            self.config.clustering.n_clusters,
        )
        
        state = env.reset()
        total_reward = 0.0
        
        while not env.done:
            # Select action
            action = self.agent.act(state)
            
            # Take step
            next_state, reward, done = env.step(action)
            
            # Store experience
            self.buffer.add(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
        
        return EpisodeResult(
            trajectory_id=trajectory.traj_id or 0,
            n_segments=env.n_segments,
            boundaries=env.boundaries,
            total_reward=total_reward,
            final_od=env.current_od,
        )
    
    def _train_step(self) -> float:
        """Perform one training step on minibatch.
        
        Returns:
            Loss.
        """
        if not self.buffer.is_ready(self.config.rl.batch_size):
            return 0.0
        
        states, actions, rewards, next_states, dones = self.buffer.sample_batch(
            self.config.rl.batch_size
        )
        
        # Compute targets
        targets = np.zeros(self.config.rl.batch_size)
        for i in range(self.config.rl.batch_size):
            targets[i] = self.agent.compute_target(
                rewards[i], next_states[i], dones[i]
            )
        
        # Update
        loss = self.agent.update(states, actions, targets)
        
        return loss
    
    def train(
        self,
        n_epochs: Optional[int] = None,
        verbose: bool = True,
    ) -> TrainingResult:
        """Run full training loop.
        
        Args:
            n_epochs: Number of epochs (overrides config).
            verbose: Whether to print progress.
        
        Returns:
            TrainingResult.
        """
        n_epochs = n_epochs or self.config.training.n_epochs
        start_time = time.time()
        
        for epoch in range(n_epochs):
            epoch_rewards = []
            epoch_ods = []
            
            # Run episodes
            for trajectory in self.dataset.trajectories:
                result = self._run_episode(trajectory)
                self.episode_rewards.append(result.total_reward)
                epoch_rewards.append(result.total_reward)
                epoch_ods.append(result.final_od)
                
                # Train step
                if self.buffer.is_ready(self.config.rl.batch_size):
                    self._train_step()
                
                # Decay epsilon
                self.agent.decay_epsilon()
            
            # Record epoch OD
            avg_od = np.mean(epoch_ods)
            self.od_history.append(avg_od)
            
            if verbose:
                print(
                    f"Epoch {epoch + 1}/{n_epochs}: "
                    f"Avg Reward = {np.mean(epoch_rewards):.4f}, "
                    f"Avg OD = {avg_od:.4f}, "
                    f"Epsilon = {self.agent.epsilon:.3f}"
                )
        
        # Compute final metrics
        elapsed = time.time() - start_time
        
        # Final F1 on last epoch
        all_pred_boundaries = []
        all_true_boundaries = []
        for trajectory in self.dataset.trajectories:
            env = MDPEnvironment(trajectory, self.feature_extractor)
            state = env.reset()
            while not env.done:
                action = self.agent.act(state, greedy=True)
                state, _, _ = env.step(action)
            all_pred_boundaries.extend(env.boundaries)
            all_true_boundaries.extend(trajectory.boundaries)
        
        _, _, final_f1 = segmentation_f1(all_pred_boundaries, all_true_boundaries)
        
        return TrainingResult(
            n_epochs=n_epochs,
            n_episodes=len(self.episode_rewards),
            final_od=self.od_history[-1] if self.od_history else 0.0,
            final_f1=final_f1,
            episode_rewards=self.episode_rewards,
            od_history=self.od_history,
            runtime_seconds=elapsed,
        )


def train_qrlstc(
    dataset: SyntheticDataset,
    n_epochs: int = 2,
    config: Optional[QRLSTCConfig] = None,
    verbose: bool = True,
) -> TrainingResult:
    """Convenience function for training Q-RLSTC.
    
    Args:
        dataset: Training dataset.
        n_epochs: Number of epochs.
        config: Configuration.
        verbose: Print progress.
    
    Returns:
        TrainingResult.
    """
    config = config or QRLSTCConfig()
    config.training.n_epochs = n_epochs
    
    trainer = Trainer(dataset, config)
    return trainer.train(n_epochs, verbose)
