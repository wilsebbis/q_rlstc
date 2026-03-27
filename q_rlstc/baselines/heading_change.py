"""Heading-change baseline policy for trajectory segmentation.

A classical geometric heuristic that cuts the trajectory whenever 
the angle between consecutive segments exceeds a given threshold.
"""

import math
from dataclasses import dataclass
from typing import List

from q_rlstc.data.rlstc_traj import Traj
from q_rlstc.baselines.random_policy import BaselineResult

def _heading(p1, p2) -> float:
    return math.atan2(p2.y - p1.y, p2.x - p1.x)

def run_heading_change(
    trajectory: Traj, 
    trajectory_index: int, 
    threshold_degrees: float = 45.0
) -> BaselineResult:
    """Cut trajectory when heading change exceeds the threshold.

    Args:
        trajectory: The trajectory to segment.
        trajectory_index: Index identifying this trajectory.
        threshold_degrees: Threshold angle in degrees.

    Returns:
        BaselineResult containing the geometric boundaries.
    """
    size = len(trajectory)
    boundaries = []
    threshold_rad = math.radians(threshold_degrees)
    
    if size < 3:
        return BaselineResult(trajectory_index, [], size, 0.0)
        
    for i in range(1, size - 1):
        h1 = _heading(trajectory.points[i-1], trajectory.points[i])
        h2 = _heading(trajectory.points[i], trajectory.points[i+1])
        
        diff = abs(h2 - h1)
        # Normalize to [0, pi]
        if diff > math.pi:
            diff = 2 * math.pi - diff
            
        if diff > threshold_rad:
            boundaries.append(i)
            
    cut_fraction = len(boundaries) / (size - 2) if size > 2 else 0.0
    
    return BaselineResult(
        trajectory_index=trajectory_index,
        boundaries=boundaries,
        total_points=size,
        cut_fraction=cut_fraction
    )
