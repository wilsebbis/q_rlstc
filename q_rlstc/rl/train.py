"""Episodic training loop for Q-RLSTC.

Implements the RL training procedure where:
1. Agent traverses trajectories, making extend/cut/drop decisions
2. Rewards based on clustering quality improvement + separability + degeneracy
3. SPSA updates on minibatch from replay buffer

Q-RLSTC 2.0 additions:
- 3-action MDP: EXTEND, CUT, DROP (anomaly filtering)
- Reward alignment: separability bonus, degeneracy penalty, empty-cluster penalty
- Per-epoch F1 logging for OD↔F1 correlation tracking
- SAC agent support (agent_type="sac")
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from ..config import QRLSTCConfig
from ..data.synthetic import SyntheticDataset, Trajectory, Point
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
        n_dropped: Number of dropped points (anomaly filtering).
    """
    trajectory_id: int = 0
    n_segments: int = 0
    boundaries: List[int] = field(default_factory=list)
    total_reward: float = 0.0
    final_od: float = 0.0
    n_dropped: int = 0


@dataclass
class TrainingResult:
    """Result from full training run.
    
    Attributes:
        n_epochs: Number of epochs completed.
        n_episodes: Total episodes.
        final_od: Final overall distance.
        final_f1: Final segmentation F1.
        episode_rewards: Reward per episode.
        od_history: OD after each epoch.
        f1_history: F1 after each epoch.
        runtime_seconds: Total training time.
    """
    n_epochs: int = 0
    n_episodes: int = 0
    final_od: float = 0.0
    final_f1: float = 0.0
    episode_rewards: List[float] = field(default_factory=list)
    od_history: List[float] = field(default_factory=list)
    f1_history: List[float] = field(default_factory=list)
    runtime_seconds: float = 0.0


class MDPEnvironment:
    """MDP environment for sub-trajectory segmentation.
    
    State: 5-dimensional feature vector (Version A) or 8-dimensional (Version B)
        [segment_length, local_variance, centroid_distance, 
         trajectory_progress, segment_count]
    
    Actions: 
        0 = EXTEND (add next point to current segment)
        1 = CUT (end current segment, start new one)
        2 = DROP or SKIP [if n_actions=3]:
            - DROP mode (Version C): discard anomalous point, bridge gap
            - SKIP mode (Version D): fast-forward S points when state is linear
    
    Termination:
        - End of trajectory (all points consumed)
        - Max segments exceeded
    
    Anti-Gaming Constraints:
        - Minimum segment length: CUT disallowed if < MIN_SEGMENT_LEN
        - Max segments: Episode terminates early if exceeded
        - Segment penalty: Reward -= λ per segment to prevent over-segmentation
    
    Reward Alignment (Q-RLSTC 2.0):
        - Cluster separability bonus on CUT
        - Degeneracy penalty: harsh penalty for single-cluster collapse
        - Empty cluster penalty: penalty for too few effective segments
        - DROP micro-penalty to prevent over-filtering
    """
    
    # Anti-gaming constants
    MIN_SEGMENT_LEN = 3
    MAX_SEGMENTS = 50
    SEGMENT_PENALTY = 0.1
    
    # Anomaly detection thresholds (for DROP action)
    MAX_SPEED_FACTOR = 10.0   # Factor above median speed to flag anomaly
    MAX_JUMP_METERS = 500.0   # Absolute jump distance threshold
    
    def __init__(
        self,
        trajectory: Trajectory,
        feature_extractor: StateFeatureExtractor,
        n_clusters: int = 10,
        n_actions: int = 2,
        drop_penalty: float = 0.05,
        separability_weight: float = 0.3,
        degeneracy_penalty: float = 2.0,
        empty_cluster_penalty: float = 1.0,
        skip_distance: int = 0,
    ):
        """Initialize environment.
        
        Args:
            trajectory: Current trajectory to segment.
            feature_extractor: Feature extractor.
            n_clusters: Number of clusters for OD.
            n_actions: 2 = [EXTEND, CUT], 3 = [EXTEND, CUT, DROP/SKIP].
            drop_penalty: Micro-penalty for DROP action.
            separability_weight: Weight for inter-cluster distance bonus.
            degeneracy_penalty: Penalty for single-cluster collapse.
            empty_cluster_penalty: Penalty for too few effective clusters.
            skip_distance: If >0, action 2 = Q-SKIP (fast-forward S points).
                           If 0, action 2 = DROP (discard single anomalous point).
        """
        self.trajectory = trajectory
        self.feature_extractor = feature_extractor
        self.n_clusters = n_clusters
        self.n_actions = n_actions
        self.drop_penalty = drop_penalty
        self.separability_weight = separability_weight
        self.degeneracy_penalty_val = degeneracy_penalty
        self.empty_cluster_penalty_val = empty_cluster_penalty
        self.skip_distance = skip_distance  # 0 = DROP mode, >0 = SKIP mode
        
        # Precompute median inter-point speed for anomaly detection
        self._precompute_speeds()
        
        self.reset()
    
    def _precompute_speeds(self):
        """Precompute median speed for anomaly detection."""
        points = self.trajectory.points
        if len(points) < 2:
            self.median_speed = 1.0
            return
        
        speeds = []
        for i in range(1, len(points)):
            dx = points[i].x - points[i - 1].x
            dy = points[i].y - points[i - 1].y
            dist = np.sqrt(dx**2 + dy**2)
            # Use a uniform time step assumption
            speeds.append(dist)
        
        self.median_speed = float(np.median(speeds)) if speeds else 1.0
    
    def reset(self) -> np.ndarray:
        """Reset environment for new episode.
        
        Returns:
            Initial state.
        """
        self.current_idx = 0
        self.split_point = 0
        self.segments: List[Tuple[int, int]] = []
        self.segment_centroids: List[np.ndarray] = []  # For separability
        self.current_od = 0.0
        self.local_variance = 0.0
        self.n_segments = 0
        self.boundaries: List[int] = []
        self.dropped: List[int] = []  # Dropped point indices
        self.skipped: int = 0  # Total points skipped (Q-SKIP)
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
    
    def _compute_segment_centroid(self, start: int, end: int) -> np.ndarray:
        """Compute centroid of a segment."""
        points = self.trajectory.points[start:end + 1]
        if len(points) < 1:
            return np.zeros(2)
        coords = np.array([[p.x, p.y] for p in points])
        return coords.mean(axis=0)
    
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
        v1 = np.array([
            points[boundary_idx].x - points[boundary_idx - 1].x,
            points[boundary_idx].y - points[boundary_idx - 1].y
        ])
        v2 = np.array([
            points[boundary_idx + 1].x - points[boundary_idx].x,
            points[boundary_idx + 1].y - points[boundary_idx].y
        ])
        
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        
        return float(angle / np.pi)
    
    def _compute_inter_cluster_distance(self) -> float:
        """Compute average distance between segment centroids.
        
        Higher = better cluster separability.
        Returns 0 if fewer than 2 segments exist.
        """
        if len(self.segment_centroids) < 2:
            return 0.0
        
        centroids = np.array(self.segment_centroids)
        total_dist = 0.0
        n_pairs = 0
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                total_dist += np.linalg.norm(centroids[i] - centroids[j])
                n_pairs += 1
        
        return total_dist / n_pairs if n_pairs > 0 else 0.0
    
    def _is_anomaly(self, idx: int) -> bool:
        """Check if a point is an anomaly (impossible speed/jump).
        
        Uses median inter-point speed as baseline.
        """
        if idx < 1 or idx >= len(self.trajectory.points):
            return False
        
        prev = self.trajectory.points[idx - 1]
        curr = self.trajectory.points[idx]
        dx = curr.x - prev.x
        dy = curr.y - prev.y
        dist = np.sqrt(dx**2 + dy**2)
        
        # Check both absolute threshold and relative to median
        if dist > self.MAX_JUMP_METERS:
            return True
        if self.median_speed > 1e-8 and dist > self.median_speed * self.MAX_SPEED_FACTOR:
            return True
        
        return False
    
    def _is_cut_allowed(self) -> bool:
        """Check if CUT action is allowed (anti-gaming)."""
        current_segment_len = self.current_idx - self.split_point + 1
        return current_segment_len >= self.MIN_SEGMENT_LEN
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Execute action and return (next_state, reward, done).
        
        Action 0: EXTEND - add next point to current segment
        Action 1: CUT - end current segment, start new one
        Action 2: DROP - discard point, bridge gap (if n_actions=3)
        
        Args:
            action: 0, 1, or 2.
        
        Returns:
            Tuple of (next_state, reward, done).
        """
        if self.done:
            return self._get_state(), 0.0, True
        
        old_variance = self.local_variance
        reward = 0.0
        
        # Clamp action to valid range
        if action >= self.n_actions:
            action = 0
        
        # Anti-gaming: enforce minimum segment length for CUT
        if action == 1 and not self._is_cut_allowed():
            action = 0
        
        # ── Action 2: DROP or SKIP ────────────────────────────────
        if action == 2 and self.n_actions >= 3:
            if self.skip_distance > 0:
                # ── Q-SKIP mode (Version D): fast-forward S points ────
                # Solves VLDB Section 5.10: avoids QPU queries on linear segments
                S = self.skip_distance
                end = min(
                    self.current_idx + S,
                    len(self.trajectory.points) - 1,
                )
                actual_skip = end - self.current_idx
                self.skipped += actual_skip
                self.current_idx = end
                
                # Small reward: saved S circuit evaluations at no quality cost
                # Only beneficial when the segment was linear/low-variance
                if self.local_variance < 0.1:
                    reward += 0.05 * actual_skip  # Good skip
                else:
                    reward -= 0.05  # Skipped over interesting data
            else:
                # ── DROP mode (Version C): discard anomalous point ────
                self.dropped.append(self.current_idx)
                
                # Micro-penalty to prevent over-dropping
                reward -= self.drop_penalty
                
                # Bonus if the point was actually anomalous
                if self._is_anomaly(self.current_idx):
                    reward += 0.2  # Good catch
                else:
                    reward -= 0.1  # Unnecessary drop, extra penalty
                
                # Skip this point — bridge to next
                self.current_idx += 1
        
        # ── CUT action ───────────────────────────────────────────
        elif action == 1:
            segment = (self.split_point, self.current_idx)
            self.segments.append(segment)
            self.boundaries.append(self.current_idx)
            
            # Compute and store centroid for separability
            centroid = self._compute_segment_centroid(
                self.split_point, self.current_idx
            )
            self.segment_centroids.append(centroid)
            
            # Update OD incrementally
            seg_cost = self._compute_segment_cost(
                self.split_point, self.current_idx
            )
            if self.n_segments == 0:
                self.current_od = seg_cost
            else:
                self.current_od = (
                    (self.current_od * self.n_segments + seg_cost) /
                    (self.n_segments + 1)
                )
            self.n_segments += 1
            
            # Reward component 1: Boundary sharpness bonus
            boundary_score = self._compute_boundary_sharpness(self.current_idx)
            reward += boundary_score * 0.5
            
            # Reward component 2: Segment penalty (anti-gaming)
            reward -= self.SEGMENT_PENALTY
            
            # Reward component 3: Cluster separability bonus
            if len(self.segment_centroids) >= 2:
                separability = self._compute_inter_cluster_distance()
                reward += separability * self.separability_weight
            
            # Start new segment
            self.split_point = self.current_idx + 1
            self.local_variance = 0.0
            
            # Move to next point
            self.current_idx += 1
        
        # ── EXTEND action ────────────────────────────────────────
        else:
            self.local_variance = self._compute_local_variance(
                self.split_point, self.current_idx
            )
            self.current_idx += 1
        
        # Local delta reward: variance improvement
        if self.current_idx < len(self.trajectory.points):
            new_variance = self._compute_local_variance(
                self.split_point,
                min(self.current_idx, len(self.trajectory.points) - 1),
            )
            variance_delta = old_variance - new_variance
            reward += variance_delta * 0.1
        else:
            new_variance = self.local_variance
        
        # ── Termination ──────────────────────────────────────────
        if self.current_idx >= len(self.trajectory.points) - 1:
            # Force final segment
            if self.split_point < len(self.trajectory.points):
                segment = (self.split_point, len(self.trajectory.points) - 1)
                self.segments.append(segment)
                centroid = self._compute_segment_centroid(
                    self.split_point, len(self.trajectory.points) - 1
                )
                self.segment_centroids.append(centroid)
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
            
            # ── End-of-episode reward shaping ────────────────────
            # Degeneracy penalty: single-cluster collapse
            if self.n_segments <= 1:
                reward -= self.degeneracy_penalty_val
            
            # Empty cluster penalty: too few effective segments
            n_effective = len(
                [s for s in self.segments if (s[1] - s[0]) >= 2]
            )
            if n_effective < 2 and self.n_segments > 1:
                reward -= self.empty_cluster_penalty_val
        
        # Anti-gaming: terminate if too many segments
        if self.n_segments >= self.MAX_SEGMENTS:
            self.done = True
        
        self.local_variance = new_variance
        
        return self._get_state(), reward, self.done


class Trainer:
    """Trainer for Q-RLSTC.
    
    Orchestrates training loop, replay buffer, and evaluation.
    Supports both DQN and SAC agent types.
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
        
        # Create agent
        if self.config.rl.agent_type == "sac":
            self._init_sac_agent()
        else:
            self._init_dqn_agent()
        
        # Replay buffer
        self.buffer = ReplayBuffer(
            max_size=self.config.rl.memory_size,
        )
        
        # Feature extractor — version-specific
        if self.version == "B":
            from ..data.features import StateFeatureExtractorB
            self.feature_extractor = StateFeatureExtractorB(
                n_clusters=self.config.clustering.n_clusters,
            )
        elif self.version == "D":
            from ..data.features import StateFeatureExtractorD
            self.feature_extractor = StateFeatureExtractorD(
                n_clusters=self.config.clustering.n_clusters,
            )
        else:
            self.feature_extractor = StateFeatureExtractor(
                n_clusters=self.config.clustering.n_clusters,
            )
        
        # Statistics
        self.episode_rewards: List[float] = []
        self.od_history: List[float] = []
        self.f1_history: List[float] = []
    
    def _init_dqn_agent(self):
        """Initialize DQN agent."""
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
        
        backend = get_backend(
            mode="noisy_sim" if self.config.noise.use_noise else "ideal",
            noise_model_name=self.config.noise.noise_model,
        )
        
        self.agent = VQDQNAgent(config=agent_config, backend=backend)
    
    def _init_sac_agent(self):
        """Initialize SAC agent (quantum actor + classical critic)."""
        from .sac_agent import QSACAgent
        
        state_dim = 8 if self.version == "B" else 5
        
        backend = get_backend(
            mode="noisy_sim" if self.config.noise.use_noise else "ideal",
            noise_model_name=self.config.noise.noise_model,
        )
        
        self.agent = QSACAgent(
            state_dim=state_dim,
            n_actions=self.config.rl.n_actions,
            n_qubits=self.config.vqdqn.n_qubits,
            n_layers=self.config.vqdqn.n_layers,
            gamma=self.config.rl.gamma,
            entropy_alpha=self.config.rl.entropy_alpha,
            critic_lr=self.config.rl.critic_lr,
            critic_hidden=self.config.rl.critic_hidden,
            shots=self.config.vqdqn.shots_train,
            backend=backend,
        )
    
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
            n_actions=self.config.rl.n_actions,
            drop_penalty=self.config.rl.drop_penalty,
            separability_weight=self.config.rl.separability_weight,
            degeneracy_penalty=self.config.rl.degeneracy_penalty,
            empty_cluster_penalty=self.config.rl.empty_cluster_penalty,
            skip_distance=getattr(self.config.rl, 'skip_distance', 0),
        )
        
        state = env.reset()
        total_reward = 0.0
        
        while not env.done:
            action = self.agent.act(state)
            next_state, reward, done = env.step(action)
            
            self.buffer.add(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
        
        return EpisodeResult(
            trajectory_id=trajectory.traj_id or 0,
            n_segments=env.n_segments,
            boundaries=env.boundaries,
            total_reward=total_reward,
            final_od=env.current_od,
            n_dropped=len(env.dropped),
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
        
        loss = self.agent.update(states, actions, targets)
        
        return loss
    
    def _compute_epoch_f1(self) -> float:
        """Compute F1 on the full dataset with greedy policy.
        
        Returns:
            Segmentation F1 score.
        """
        all_pred = []
        all_true = []
        for trajectory in self.dataset.trajectories:
            env = MDPEnvironment(
                trajectory,
                self.feature_extractor,
                n_actions=self.config.rl.n_actions,
            )
            state = env.reset()
            while not env.done:
                action = self.agent.act(state, greedy=True)
                state, _, _ = env.step(action)
            all_pred.extend(env.boundaries)
            all_true.extend(trajectory.boundaries)
        
        _, _, f1 = segmentation_f1(all_pred, all_true)
        return f1
    
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
            
            for trajectory in self.dataset.trajectories:
                result = self._run_episode(trajectory)
                self.episode_rewards.append(result.total_reward)
                epoch_rewards.append(result.total_reward)
                epoch_ods.append(result.final_od)
                
                if self.buffer.is_ready(self.config.rl.batch_size):
                    self._train_step()
                
                # Decay epsilon (DQN only — SAC uses entropy)
                if hasattr(self.agent, 'decay_epsilon'):
                    self.agent.decay_epsilon()
            
            # Record epoch metrics
            avg_od = np.mean(epoch_ods)
            self.od_history.append(avg_od)
            
            # Per-epoch F1 for OD↔F1 correlation tracking
            epoch_f1 = self._compute_epoch_f1()
            self.f1_history.append(epoch_f1)
            
            if verbose:
                eps_str = ""
                if hasattr(self.agent, 'epsilon'):
                    eps_str = f"  ε={self.agent.epsilon:.3f}"
                print(
                    f"Epoch {epoch + 1}/{n_epochs}: "
                    f"Reward={np.mean(epoch_rewards):.4f}  "
                    f"OD={avg_od:.4f}  "
                    f"F1={epoch_f1:.4f}"
                    f"{eps_str}"
                )
        
        elapsed = time.time() - start_time
        
        final_f1 = self.f1_history[-1] if self.f1_history else 0.0
        
        return TrainingResult(
            n_epochs=n_epochs,
            n_episodes=len(self.episode_rewards),
            final_od=self.od_history[-1] if self.od_history else 0.0,
            final_f1=final_f1,
            episode_rewards=self.episode_rewards,
            od_history=self.od_history,
            f1_history=self.f1_history,
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
    trainer = Trainer(dataset, config)
    return trainer.train(n_epochs=n_epochs, verbose=verbose)
