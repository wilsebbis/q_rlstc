"""Random baseline policy for trajectory segmentation.

Cuts at each point with a fixed probability to hit a target CUT% budget.
Used as the absolute floor of performance to prove the RL agent learns
meaningful geometry rather than just exploiting the length bias of ValCR.
"""

import random
from dataclasses import dataclass
from typing import List

from q_rlstc.data.rlstc_traj import Traj

@dataclass
class BaselineResult:
    """Standardized output for non-learned baseline policies."""
    trajectory_index: int
    boundaries: List[int]
    total_points: int
    cut_fraction: float

def run_random_policy(
    trajectory: Traj, 
    trajectory_index: int, 
    target_cut_budget: float = 0.1,
    seed: int = None
) -> BaselineResult:
    """Cut trajectory points randomly based on target budget.

    Args:
        trajectory: The trajectory to segment.
        trajectory_index: Index identifying this trajectory.
        target_cut_budget: Target fraction of points to split at (0.0 to 1.0).
        seed: Optional RNG seed.

    Returns:
        BaselineResult containing the randomly determined boundaries.
    """
    if seed is not None:
        random.seed(seed)
        
    size = len(trajectory)
    boundaries = []
    
    # We can't cut at index 0 or size-1
    for i in range(1, size - 1):
        if random.random() < target_cut_budget:
            boundaries.append(i)
            
    cut_fraction = len(boundaries) / (size - 2) if size > 2 else 0.0
    
    return BaselineResult(
        trajectory_index=trajectory_index,
        boundaries=boundaries,
        total_points=size,
        cut_fraction=cut_fraction
    )
