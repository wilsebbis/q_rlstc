"""Fixed-window baseline policy for trajectory segmentation.

Cuts the trajectory blindly every N points. This tests whether 
the dataset just needs arbitrary segmentation to achieve high ValCR,
or whether smart geometric decision making actually matters.
"""

from dataclasses import dataclass
from typing import List

from q_rlstc.data.rlstc_traj import Traj
from q_rlstc.baselines.random_policy import BaselineResult

def run_fixed_window(
    trajectory: Traj, 
    trajectory_index: int, 
    window_size: int = 10
) -> BaselineResult:
    """Cut trajectory exactly every N points.

    Args:
        trajectory: The trajectory to segment.
        trajectory_index: Index identifying this trajectory.
        window_size: The number of points between cuts.

    Returns:
        BaselineResult containing the periodic boundaries.
    """
    size = len(trajectory)
    boundaries = []
    
    for i in range(window_size, size - 1, window_size):
        boundaries.append(i)
            
    cut_fraction = len(boundaries) / (size - 2) if size > 2 else 0.0
    
    return BaselineResult(
        trajectory_index=trajectory_index,
        boundaries=boundaries,
        total_points=size,
        cut_fraction=cut_fraction
    )
