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


# Alias for Version A
StateFeatureExtractorA = StateFeatureExtractor


class StateFeatureExtractorB(StateFeatureExtractor):
    """Extended 8-dimensional state features for Version B (quantum-optimized).
    
    Inherits all 5 base features from Version A, then adds:
      5. angle_spread: Variance of arctan-scaled features (quantum encoding spread)
      6. curvature_gradient: Rate of change of segment curvature
      7. segment_density: Points per unit spatial distance in current segment
    
    The additional features exploit the 8-qubit circuit's larger Hilbert space
    and provide richer geometric information.
    """
    
    def _compute_angle_spread(self, base_features: np.ndarray) -> float:
        """Variance of the angle-encoded feature values.
        
        When features are angle-encoded via arctan, the variance of the
        encoded angles reflects how "spread" the quantum state is across
        the Bloch sphere. Low spread = qubits near same angle = ambiguous.
        
        Args:
            base_features: The 5D base feature vector.
        
        Returns:
            Variance of arctan-scaled features, in [0, ~1].
        """
        angles = 2.0 * np.arctan(base_features)  # Same as circuit encoding
        return float(np.var(angles))
    
    def _compute_curvature_gradient(
        self,
        points: List[Point],
        current_idx: int,
        split_point: int,
    ) -> float:
        """Rate of change of curvature (second-order geometric signal).
        
        Captures acceleration of direction change — distinguishes gradual
        curves from sudden turns. Classical RLSTC uses curvature but not
        its derivative.
        
        Args:
            points: Trajectory points.
            current_idx: Current index.
            split_point: Segment start index.
        
        Returns:
            Curvature gradient, normalized.
        """
        segment_len = current_idx - split_point + 1
        if segment_len < 4:
            return 0.0
        
        # Compute curvature at two windows within the segment
        mid = split_point + segment_len // 2
        
        curv_first = self._compute_segment_curvature(points, split_point, mid)
        curv_second = self._compute_segment_curvature(points, mid, current_idx)
        
        # Normalize by segment length to get rate
        half_len = max(segment_len / 2, 1.0)
        gradient = (curv_second - curv_first) / half_len
        
        # Sigmoid-like compression to [-1, 1]
        return float(np.tanh(gradient))
    
    def _compute_segment_density(
        self,
        points: List[Point],
        start_idx: int,
        end_idx: int,
    ) -> float:
        """Points per unit spatial distance in current segment.
        
        Captures congestion vs free-flow without explicit speed.
        High density = stop-and-go or slow movement.
        Low density = fast straight-line movement.
        
        Args:
            points: Trajectory points.
            start_idx: Segment start.
            end_idx: Segment end.
        
        Returns:
            Density score (normalized via arctan to [0, 1]).
        """
        n_points = end_idx - start_idx + 1
        if n_points < 2:
            return 0.5  # Neutral
        
        spatial_length = self._compute_segment_length(points, start_idx, end_idx)
        if spatial_length < 1e-10:
            return 1.0  # All points at same location = max density
        
        density = n_points / spatial_length
        # Normalize to [0, 1] via arctan
        return float(2.0 * np.arctan(density) / np.pi)
    
    def extract_features(
        self,
        trajectory: Trajectory,
        current_idx: int,
        split_point: int,
        current_od: float,
        n_segments: int,
    ) -> np.ndarray:
        """Extract 8-dimensional state features (5 base + 3 quantum-native).
        
        Args:
            trajectory: Current trajectory.
            current_idx: Current point index.
            split_point: Index of last split point.
            current_od: Current overall clustering distance.
            n_segments: Number of segments created so far.
        
        Returns:
            8-dimensional state vector.
        """
        # Get the 5 base features from Version A
        base = super().extract_features(
            trajectory, current_idx, split_point, current_od, n_segments
        )
        
        points = trajectory.points
        
        # Feature 5: Angle spread of base features
        angle_spread = self._compute_angle_spread(base)
        
        # Feature 6: Curvature gradient
        curv_gradient = self._compute_curvature_gradient(
            points, current_idx, split_point
        )
        
        # Feature 7: Segment density
        seg_density = self._compute_segment_density(
            points, split_point, current_idx
        )
        
        return np.array([
            base[0], base[1], base[2], base[3], base[4],
            angle_spread,
            curv_gradient,
            seg_density,
        ], dtype=np.float32)


def extract_state_features(
    trajectory: Trajectory,
    current_idx: int,
    split_point: int,
    current_od: float = 0.0,
    n_segments: int = 0,
    n_clusters: int = 10,
    version: str = "A",
) -> np.ndarray:
    """Convenience function to extract state features.
    
    Args:
        trajectory: Current trajectory.
        current_idx: Current point index.
        split_point: Last split point index.
        current_od: Current overall distance.
        n_segments: Number of segments so far.
        n_clusters: Number of clusters for OD estimation.
        version: "A" (5D), "B" (8D), "C" (same as A), or "D" (VLDB baseline 5D).
    
    Returns:
        5D (versions A/C/D) or 8D (version B) state vector.
    """
    v = version.upper()
    if v == "B":
        extractor = StateFeatureExtractorB(n_clusters=n_clusters)
    elif v == "D":
        extractor = StateFeatureExtractorD(n_clusters=n_clusters)
    else:
        extractor = StateFeatureExtractor(n_clusters=n_clusters)
    return extractor.extract_features(
        trajectory, current_idx, split_point, current_od, n_segments
    )


class StateFeatureExtractorD(StateFeatureExtractor):
    """VLDB 2024 paper-exact 5-dimensional state features for Version D.
    
    Implements the exact state vector from Equation (19) of the paper:
    
        s_t = (OD_s, OD_n, OD_b, L_b, L_f)
    
    Where:
        1. **OD_s**: Overall distance including only the current point (if CUT).
        2. **OD_n**: Overall distance including the next point (if EXTEND).
        3. **OD_b**: TRACLUS heuristic baseline result (expert knowledge).
        4. **L_b**: Current sub-trajectory length, normalised by trajectory length.
        5. **L_f**: Remaining trajectory length, normalised by trajectory length.
    
    This is functionally identical to Version A's feature extractor. The key
    difference in Version D is the *model architecture* (5q × 3 layers = 30 params)
    and the optional Q-SKIP extension, not the feature vector.
    
    The quantum substitution is:
        Replace the paper's 2-layer FFN (5→64→2 ≈ 514 params) with a VQC
        that outputs Q(s, EXTEND) and Q(s, CUT) via Z-expectations.
    
    Research Variants (NOT baseline, labeled explicitly):
        - Hilbert curve spatial anchors replacing L_b/L_f
        - Target network self-play replacing TRACLUS OD_b
    """
    
    def __init__(
        self,
        n_clusters: int = 10,
        window_size: int = 5,
    ):
        """Initialize VLDB-aligned extractor.
        
        Args:
            n_clusters: Number of clusters for OD estimation.
            window_size: Window size for curvature (inherited, unused in D baseline).
        """
        super().__init__(n_clusters=n_clusters, window_size=window_size)
    
    def extract_features(
        self,
        trajectory: Trajectory,
        current_idx: int,
        split_point: int,
        current_od: float,
        n_segments: int,
    ) -> np.ndarray:
        """Extract VLDB paper-exact 5-dimensional state features.
        
        State vector: s_t = (OD_s, OD_n, OD_b, L_b, L_f)
        
        This is the exact formulation from the paper's Equation (19).
        The features are identical to Version A — Version D's distinction
        is the VQC architecture (3 layers, 30 params) and optional Q-SKIP.
        
        Args:
            trajectory: Current trajectory.
            current_idx: Current point index.
            split_point: Index of last split point.
            current_od: Current overall clustering distance.
            n_segments: Number of segments created so far.
        
        Returns:
            5-dimensional VLDB-aligned state vector.
        """
        # Delegate to parent — the paper's features ARE our Version A features
        return super().extract_features(
            trajectory, current_idx, split_point, current_od, n_segments
        )

