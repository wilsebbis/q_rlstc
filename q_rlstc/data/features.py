"""5-dimensional state feature extraction for VQ-DQN.

State vector:
  x = [OD_segment, OD_continue, baseline_TRACLUS_like, len_backward_norm, len_forward_norm]

Where:
- OD_segment: Projected overall distance if we split here
- OD_continue: Projected overall distance if we continue
- baseline_TRACLUS_like: Heuristic cost (MDL proxy / curvature score)
- len_backward_norm: Current segment length normalized
- len_forward_norm: Remaining trajectory length normalized
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .synthetic import Point, Trajectory


@dataclass
class ClusterState:
    """State of clustering for reward/OD computation.
    
    Attributes:
        centroids: Current cluster centroids as numpy arrays.
        assignments: Current cluster assignments.
        overall_distance: Current overall clustering distance.
    """
    centroids: np.ndarray
    assignments: np.ndarray
    overall_distance: float


class StateFeatureExtractor:
    """Extract 5-dimensional state features for VQ-DQN.
    
    State features encode information about:
    1. Clustering quality under different actions
    2. Trajectory geometry (curvature, segment lengths)
    3. Position within the trajectory
    """
    
    def __init__(
        self,
        n_clusters: int = 10,
        window_size: int = 5,
    ):
        """Initialize extractor.
        
        Args:
            n_clusters: Number of clusters for OD estimation.
            window_size: Window size for local curvature estimation.
        """
        self.n_clusters = n_clusters
        self.window_size = window_size
        
        # Cache for lightweight OD proxies
        self._od_cache: Dict[str, float] = {}
    
    def _compute_segment_curvature(
        self,
        points: List[Point],
        start_idx: int,
        end_idx: int,
    ) -> float:
        """Compute curvature score for a segment.
        
        Uses sum of angle changes between consecutive line segments.
        
        Args:
            points: List of trajectory points.
            start_idx: Segment start index.
            end_idx: Segment end index.
        
        Returns:
            Curvature score (higher = more curved).
        """
        if end_idx - start_idx < 2:
            return 0.0
        
        total_angle_change = 0.0
        
        for i in range(start_idx + 1, end_idx):
            # Vector from i-1 to i
            v1_x = points[i].x - points[i-1].x
            v1_y = points[i].y - points[i-1].y
            
            # Vector from i to i+1
            if i + 1 <= end_idx and i + 1 < len(points):
                v2_x = points[i+1].x - points[i].x
                v2_y = points[i+1].y - points[i].y
                
                # Angle between vectors
                dot = v1_x * v2_x + v1_y * v2_y
                cross = v1_x * v2_y - v1_y * v2_x
                angle = np.abs(np.arctan2(cross, dot))
                total_angle_change += angle
        
        return total_angle_change
    
    def _compute_segment_length(self, points: List[Point], start_idx: int, end_idx: int) -> float:
        """Compute total length of a segment.
        
        Args:
            points: List of points.
            start_idx: Start index.
            end_idx: End index (inclusive).
        
        Returns:
            Total Euclidean length.
        """
        total_length = 0.0
        for i in range(start_idx, min(end_idx, len(points) - 1)):
            total_length += points[i].distance(points[i + 1])
        return total_length
    
    def _compute_od_proxy(
        self,
        segment_points: List[Point],
        current_od: float,
        n_segments: int,
    ) -> float:
        """Lightweight proxy for overall distance after adding segment.
        
        Uses simple distance-based heuristic rather than full k-means.
        
        Args:
            segment_points: Points in the segment.
            current_od: Current overall distance.
            n_segments: Current number of segments.
        
        Returns:
            Projected overall distance.
        """
        if len(segment_points) < 2:
            return current_od
        
        # Compute segment "diameter" as simple quality proxy
        segment_arr = np.array([[p.x, p.y] for p in segment_points])
        centroid = segment_arr.mean(axis=0)
        distances = np.linalg.norm(segment_arr - centroid, axis=1)
        segment_cost = distances.mean()
        
        # Update running average
        new_od = (current_od * n_segments + segment_cost) / (n_segments + 1)
        return new_od
    
    def _compute_traclus_baseline(
        self,
        points: List[Point],
        split_idx: int,
    ) -> float:
        """Compute TRACLUS-like baseline score.
        
        Uses MDL-inspired cost: balance between compression and reconstruction error.
        
        Args:
            points: Trajectory points.
            split_idx: Current index (potential split point).
        
        Returns:
            Baseline cost score.
        """
        if split_idx < 2:
            return 0.0
        
        # Simplified MDL: cost of representing trajectory with straight line
        # vs actual trajectory
        
        # Start and end of current segment
        segment_points = points[:split_idx + 1]
        if len(segment_points) < 2:
            return 0.0
        
        start = segment_points[0]
        end = segment_points[-1]
        
        # Perpendicular distance from each point to start-end line
        total_perp_dist = 0.0
        line_vec = np.array([end.x - start.x, end.y - start.y])
        line_len = np.linalg.norm(line_vec)
        
        if line_len < 1e-10:
            return 0.0
        
        line_unit = line_vec / line_len
        
        for p in segment_points[1:-1]:
            vec_to_p = np.array([p.x - start.x, p.y - start.y])
            projection = np.dot(vec_to_p, line_unit)
            projection = np.clip(projection, 0, line_len)
            proj_point = np.array([start.x, start.y]) + projection * line_unit
            perp_dist = np.linalg.norm(np.array([p.x, p.y]) - proj_point)
            total_perp_dist += perp_dist
        
        # MDL score: log2(line_len) + sum of perpendicular distances
        mdl_score = np.log2(max(line_len, 1.0)) + total_perp_dist
        
        return mdl_score
    
    def extract_features(
        self,
        trajectory: Trajectory,
        current_idx: int,
        split_point: int,
        current_od: float,
        n_segments: int,
    ) -> np.ndarray:
        """Extract 5-dimensional state features.
        
        Args:
            trajectory: Current trajectory.
            current_idx: Current point index.
            split_point: Index of last split point.
            current_od: Current overall clustering distance.
            n_segments: Number of segments created so far.
        
        Returns:
            5-dimensional state vector.
        """
        points = trajectory.points
        total_length = len(points)
        
        # Feature 0: OD if we split here
        segment_points = points[split_point:current_idx + 1]
        od_segment = self._compute_od_proxy(segment_points, current_od, n_segments)
        
        # Feature 1: OD if we continue (add one more point)
        if current_idx + 1 < total_length:
            extended_points = points[split_point:current_idx + 2]
            od_continue = self._compute_od_proxy(extended_points, current_od, n_segments)
        else:
            od_continue = od_segment
        
        # Feature 2: TRACLUS-like baseline cost
        baseline_cost = self._compute_traclus_baseline(
            points[split_point:current_idx + 1],
            current_idx - split_point,
        )
        # Normalize to reasonable range
        baseline_cost = baseline_cost / max(total_length, 1)
        
        # Feature 3: Current segment length normalized
        segment_len = current_idx - split_point + 1
        len_backward_norm = segment_len / total_length
        
        # Feature 4: Remaining trajectory length normalized
        remaining_len = total_length - current_idx - 1
        len_forward_norm = remaining_len / total_length
        
        return np.array([
            od_segment,
            od_continue,
            baseline_cost,
            len_backward_norm,
            len_forward_norm,
        ], dtype=np.float32)


def extract_state_features(
    trajectory: Trajectory,
    current_idx: int,
    split_point: int,
    current_od: float = 0.0,
    n_segments: int = 0,
    n_clusters: int = 10,
) -> np.ndarray:
    """Convenience function to extract state features.
    
    Args:
        trajectory: Current trajectory.
        current_idx: Current point index.
        split_point: Last split point index.
        current_od: Current overall distance.
        n_segments: Number of segments so far.
        n_clusters: Number of clusters for OD estimation.
    
    Returns:
        5-dimensional state vector.
    """
    extractor = StateFeatureExtractor(n_clusters=n_clusters)
    return extractor.extract_features(
        trajectory, current_idx, split_point, current_od, n_segments
    )
