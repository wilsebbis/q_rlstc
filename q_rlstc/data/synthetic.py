"""Synthetic trajectory generation with piecewise behaviors.

Generates trajectories with three distinct behavior modes:
1. Straight high-speed segments
2. Stop-and-go congested segments
3. Turning/parking-like segments

Provides ground truth segmentation boundaries for evaluation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Literal
from enum import Enum


class BehaviorType(Enum):
    """Types of trajectory behaviors."""
    STRAIGHT_FAST = "straight_fast"
    STOP_AND_GO = "stop_and_go"
    TURNING = "turning"


@dataclass
class Point:
    """A spatial-temporal point.
    
    Attributes:
        x: X-coordinate (longitude-like).
        y: Y-coordinate (latitude-like).
        t: Timestamp (normalized).
    """
    x: float
    y: float
    t: float
    
    def distance(self, other: "Point") -> float:
        """Euclidean distance to another point."""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y, t]."""
        return np.array([self.x, self.y, self.t])


@dataclass
class Trajectory:
    """A trajectory as a sequence of points.
    
    Attributes:
        points: List of Point objects.
        traj_id: Optional identifier.
        boundaries: Ground truth segment boundaries (indices).
        labels: Behavior type for each segment.
    """
    points: List[Point]
    traj_id: Optional[int] = None
    boundaries: List[int] = field(default_factory=list)
    labels: List[BehaviorType] = field(default_factory=list)
    
    @property
    def size(self) -> int:
        """Number of points."""
        return len(self.points)
    
    @property
    def ts(self) -> float:
        """Start time."""
        return self.points[0].t if self.points else 0.0
    
    @property
    def te(self) -> float:
        """End time."""
        return self.points[-1].t if self.points else 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array of shape (n_points, 3)."""
        return np.array([[p.x, p.y, p.t] for p in self.points])
    
    @classmethod
    def from_array(cls, arr: np.ndarray, traj_id: Optional[int] = None) -> "Trajectory":
        """Create from numpy array of shape (n_points, 3)."""
        points = [Point(x=row[0], y=row[1], t=row[2]) for row in arr]
        return cls(points=points, traj_id=traj_id)


@dataclass
class SyntheticDataset:
    """A dataset of synthetic trajectories with ground truth.
    
    Attributes:
        trajectories: List of Trajectory objects.
        n_trajectories: Number of trajectories.
    """
    trajectories: List[Trajectory]
    
    @property
    def n_trajectories(self) -> int:
        return len(self.trajectories)
    
    def get_all_boundaries(self) -> List[List[int]]:
        """Get ground truth boundaries for all trajectories."""
        return [t.boundaries for t in self.trajectories]


class TrajectoryGenerator:
    """Generator for synthetic piecewise trajectories.
    
    Creates trajectories with varying behavior modes to test segmentation.
    """
    
    def __init__(
        self,
        seed: int = 42,
        dt: float = 1.0,
        noise_std: float = 0.1,
    ):
        """Initialize generator.
        
        Args:
            seed: Random seed for reproducibility.
            dt: Time step between points.
            noise_std: Standard deviation of position noise.
        """
        self.rng = np.random.default_rng(seed)
        self.dt = dt
        self.noise_std = noise_std
    
    def _generate_straight_fast(
        self, 
        start: Point, 
        n_points: int,
        speed: float = 10.0,
        heading: float = 0.0,
    ) -> Tuple[List[Point], BehaviorType]:
        """Generate straight high-speed segment.
        
        Args:
            start: Starting point.
            n_points: Number of points to generate.
            speed: Speed in units per time step.
            heading: Direction in radians.
        
        Returns:
            List of points and behavior type.
        """
        points = []
        x, y, t = start.x, start.y, start.t
        
        vx = speed * np.cos(heading)
        vy = speed * np.sin(heading)
        
        for i in range(n_points):
            noise_x = self.rng.normal(0, self.noise_std)
            noise_y = self.rng.normal(0, self.noise_std)
            
            points.append(Point(
                x=x + noise_x,
                y=y + noise_y,
                t=t
            ))
            
            x += vx * self.dt
            y += vy * self.dt
            t += self.dt
        
        return points, BehaviorType.STRAIGHT_FAST
    
    def _generate_stop_and_go(
        self,
        start: Point,
        n_points: int,
        speed: float = 2.0,
        heading: float = 0.0,
        stop_prob: float = 0.3,
    ) -> Tuple[List[Point], BehaviorType]:
        """Generate stop-and-go congested segment.
        
        Args:
            start: Starting point.
            n_points: Number of points.
            speed: Speed when moving.
            heading: General direction.
            stop_prob: Probability of stopping at each step.
        
        Returns:
            List of points and behavior type.
        """
        points = []
        x, y, t = start.x, start.y, start.t
        
        for i in range(n_points):
            is_stopped = self.rng.random() < stop_prob
            
            if not is_stopped:
                vx = speed * np.cos(heading) + self.rng.normal(0, speed * 0.3)
                vy = speed * np.sin(heading) + self.rng.normal(0, speed * 0.3)
                x += vx * self.dt
                y += vy * self.dt
            
            noise_x = self.rng.normal(0, self.noise_std * 0.5)
            noise_y = self.rng.normal(0, self.noise_std * 0.5)
            
            points.append(Point(
                x=x + noise_x,
                y=y + noise_y,
                t=t
            ))
            
            t += self.dt
        
        return points, BehaviorType.STOP_AND_GO
    
    def _generate_turning(
        self,
        start: Point,
        n_points: int,
        speed: float = 3.0,
        initial_heading: float = 0.0,
        turn_rate: float = 0.2,
    ) -> Tuple[List[Point], BehaviorType]:
        """Generate turning/parking-like segment.
        
        Args:
            start: Starting point.
            n_points: Number of points.
            speed: Speed while turning.
            initial_heading: Starting direction.
            turn_rate: Radians per time step.
        
        Returns:
            List of points and behavior type.
        """
        points = []
        x, y, t = start.x, start.y, start.t
        heading = initial_heading
        
        for i in range(n_points):
            heading += turn_rate + self.rng.normal(0, 0.05)
            
            vx = speed * np.cos(heading)
            vy = speed * np.sin(heading)
            
            x += vx * self.dt
            y += vy * self.dt
            
            noise_x = self.rng.normal(0, self.noise_std)
            noise_y = self.rng.normal(0, self.noise_std)
            
            points.append(Point(
                x=x + noise_x,
                y=y + noise_y,
                t=t
            ))
            
            t += self.dt
        
        return points, BehaviorType.TURNING
    
    def generate_trajectory(
        self,
        n_segments: int = 3,
        min_segment_len: int = 10,
        max_segment_len: int = 30,
        traj_id: Optional[int] = None,
    ) -> Trajectory:
        """Generate a single trajectory with multiple behavior segments.
        
        Args:
            n_segments: Number of distinct behavior segments.
            min_segment_len: Minimum points per segment.
            max_segment_len: Maximum points per segment.
            traj_id: Optional trajectory identifier.
        
        Returns:
            Trajectory with ground truth boundaries.
        """
        all_points: List[Point] = []
        boundaries: List[int] = []
        labels: List[BehaviorType] = []
        
        current_point = Point(x=0.0, y=0.0, t=0.0)
        heading = self.rng.uniform(0, 2 * np.pi)
        
        behavior_generators = [
            self._generate_straight_fast,
            self._generate_stop_and_go,
            self._generate_turning,
        ]
        
        for seg_idx in range(n_segments):
            seg_len = self.rng.integers(min_segment_len, max_segment_len + 1)
            gen_func = self.rng.choice(behavior_generators)
            
            # Generate segment
            if gen_func == self._generate_straight_fast:
                seg_points, behavior = self._generate_straight_fast(
                    current_point, seg_len, speed=10.0, heading=heading
                )
            elif gen_func == self._generate_stop_and_go:
                seg_points, behavior = self._generate_stop_and_go(
                    current_point, seg_len, speed=2.0, heading=heading
                )
            else:  # turning
                turn_rate = self.rng.uniform(-0.3, 0.3)
                seg_points, behavior = self._generate_turning(
                    current_point, seg_len, speed=3.0, 
                    initial_heading=heading, turn_rate=turn_rate
                )
                heading += turn_rate * seg_len
            
            # Mark boundary (except for first segment)
            if seg_idx > 0:
                boundaries.append(len(all_points))
            
            all_points.extend(seg_points)
            labels.append(behavior)
            
            # Update current position for next segment
            if seg_points:
                last = seg_points[-1]
                current_point = Point(x=last.x, y=last.y, t=last.t)
            
            # Random heading change between segments
            heading += self.rng.uniform(-np.pi/4, np.pi/4)
        
        return Trajectory(
            points=all_points,
            traj_id=traj_id,
            boundaries=boundaries,
            labels=labels,
        )
    
    def generate_dataset(
        self,
        n_trajectories: int = 100,
        n_segments_range: Tuple[int, int] = (2, 4),
        min_segment_len: int = 10,
        max_segment_len: int = 30,
    ) -> SyntheticDataset:
        """Generate a dataset of synthetic trajectories.
        
        Args:
            n_trajectories: Number of trajectories to generate.
            n_segments_range: (min, max) segments per trajectory.
            min_segment_len: Minimum points per segment.
            max_segment_len: Maximum points per segment.
        
        Returns:
            SyntheticDataset with all trajectories.
        """
        trajectories = []
        
        for i in range(n_trajectories):
            n_segs = self.rng.integers(n_segments_range[0], n_segments_range[1] + 1)
            traj = self.generate_trajectory(
                n_segments=n_segs,
                min_segment_len=min_segment_len,
                max_segment_len=max_segment_len,
                traj_id=i,
            )
            trajectories.append(traj)
        
        return SyntheticDataset(trajectories=trajectories)


def generate_synthetic_trajectories(
    n_trajectories: int = 100,
    n_segments_range: Tuple[int, int] = (2, 4),
    seed: int = 42,
) -> SyntheticDataset:
    """Convenience function to generate synthetic trajectories.
    
    Args:
        n_trajectories: Number of trajectories.
        n_segments_range: (min, max) segments per trajectory.
        seed: Random seed.
    
    Returns:
        SyntheticDataset with ground truth boundaries.
    """
    generator = TrajectoryGenerator(seed=seed)
    return generator.generate_dataset(
        n_trajectories=n_trajectories,
        n_segments_range=n_segments_range,
    )
