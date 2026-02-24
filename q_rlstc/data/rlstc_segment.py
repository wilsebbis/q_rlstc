"""Directed line segment between two 2D points.

A :class:`Segment` connects a start and end :class:`Point_xy` and provides
methods for computing the three distance components used in trajectory
comparison (TRACLUS-style):

- **Perpendicular distance** — how far apart the segments are "sideways"
- **Parallel distance** — how far the projection endpoints overshoot
- **Angle distance** — how much the segments diverge directionally

These three components sum to produce the total IED (Integrated Euclidean
Distance) used for trajectory-to-trajectory comparison.

See Also:
    :class:`Point_xy` — the 2D endpoint type.
    :func:`compare` — utility to order two segments by length.
"""

import math
from typing import Optional, Tuple

from .rlstc_point_xy import Point_xy, _point2line_distance


class Segment:
    """A directed line segment from ``start`` to ``end`` in 2D space.

    Attributes:
        start: Starting point of the segment.
        end: Ending point of the segment.
        traj_id: Optional trajectory ID this segment belongs to.
        eps: Small epsilon to avoid division by zero in distance formulas.
    """

    eps: float = 1e-12

    def __init__(
        self,
        start_point: Point_xy,
        end_point: Point_xy,
        traj_id: Optional[int] = None,
    ) -> None:
        """Initialize a segment between two 2D points.

        Args:
            start_point: Start of the directed segment.
            end_point: End of the directed segment.
            traj_id: Optional identifier of the parent trajectory.
        """
        self.start: Point_xy = start_point
        self.end: Point_xy = end_point
        self.traj_id: Optional[int] = traj_id

    @property
    def length(self) -> float:
        """Euclidean length of this segment.

        Returns:
            Distance from start to end: ||end − start||.
        """
        return self.end.distance(self.start)

    def perpendicular_distance(self, other: "Segment") -> float:
        """Perpendicular distance between this segment and *other*.

        Projects ``other``'s endpoints onto ``self``'s line, then computes
        a weighted sum of the two projection errors:

            d_perp = (l₁² + l₂²) / (l₁ + l₂)

        where l₁ and l₂ are the distances from ``other``'s start and end
        to their respective projections on ``self``.

        Args:
            other: The segment to measure perpendicular distance to.

        Returns:
            Perpendicular distance component (≥ 0).
        """
        projection_of_start = self._projection_point(other, typed="start")
        projection_of_end = self._projection_point(other, typed="end")

        dist_start = other.start.distance(projection_of_start)
        dist_end = other.end.distance(projection_of_end)

        if dist_start < self.eps and dist_end < self.eps:
            return 0.0

        numerator = dist_start ** 2 + dist_end ** 2
        denominator = dist_start + dist_end
        return numerator / denominator

    def parallel_distance(self, other: "Segment") -> float:
        """Parallel distance between this segment and *other*.

        Measures how much the projection of ``other`` onto ``self``
        overshoots or undershoots ``self``'s endpoints.

        Args:
            other: The segment to measure parallel distance to.

        Returns:
            Parallel distance component (≥ 0).
        """
        projection_of_start = self._projection_point(other, typed="start")
        projection_of_end = self._projection_point(other, typed="end")

        # Distance from self's endpoints to the projection points
        min_dist_start = min(
            self.start.distance(projection_of_start),
            self.end.distance(projection_of_start),
        )
        min_dist_end = min(
            self.end.distance(projection_of_end),
            self.start.distance(projection_of_end),
        )
        return min(min_dist_start, min_dist_end)

    def angle_distance(self, other: "Segment") -> float:
        """Angle distance between this segment and *other*.

        Measures directional divergence.  If the two segments are nearly
        parallel (cos θ > 0), the angle distance is:

            d_angle = ||other|| · sin(θ)

        If they point in opposite directions (cos θ ≤ 0), the full
        length of ``other`` is used as the distance.

        Args:
            other: The segment to measure angle distance to.

        Returns:
            Angle distance component (≥ 0).
        """
        self_vector = self.end - self.start
        other_vector = other.end - other.start

        self_length = self.end.distance(self.start)
        other_length = other.end.distance(other.start)

        # Degenerate cases: zero-length segments
        if self_length == 0:
            return _point2line_distance(
                self.start.as_array(),
                other.start.as_array(),
                other.end.as_array(),
            )
        if other_length == 0:
            return _point2line_distance(
                other.start.as_array(),
                self.start.as_array(),
                self.end.as_array(),
            )

        # Identical segments
        if self.start == other.start and self.end == other.end:
            return 0.0

        cos_theta = self_vector.dot(other_vector) / (self_length * other_length)

        if cos_theta > self.eps:
            # Clamp to avoid numerical issues with acos
            cos_theta_clamped = min(cos_theta, 1.0)
            sin_theta = math.sqrt(1.0 - cos_theta_clamped ** 2)
            return other.length * sin_theta
        else:
            # Segments point in opposite directions — use full length
            return other.length

    def _projection_point(
        self,
        other: "Segment",
        typed: str = "end",
    ) -> Point_xy:
        """Project a point from *other* onto this segment's line.

        Computes the scalar projection parameter ``u`` and returns the
        point on ``self``'s line closest to the specified endpoint of
        ``other``.

        Args:
            other: The segment whose endpoint is being projected.
            typed: Which endpoint to project — ``"start"`` or ``"end"``.
                   Accepts abbreviations ``"s"`` and ``"e"``.

        Returns:
            The projected Point_xy on this segment's line.
        """
        if typed in ("s", "start"):
            point_to_project = other.start - self.start
        else:
            point_to_project = other.end - self.start

        segment_vector = self.end - self.start
        segment_length_squared = self.end.distance(self.start) ** 2

        if segment_length_squared == 0:
            # Degenerate segment (start == end)
            return self.start

        projection_parameter = point_to_project.dot(segment_vector) / segment_length_squared
        return self.start + segment_vector * projection_parameter

    def get_all_distance(self, other_segment: "Segment") -> float:
        """Sum of all three distance components to another segment.

        Combines perpendicular, parallel, and angle distances into
        a single scalar.  This is the total IED component used in
        trajectory comparison.

        Args:
            other_segment: The segment to measure total distance to.

        Returns:
            Sum of perpendicular + parallel + angle distances.
        """
        return (
            self.angle_distance(other_segment)
            + self.parallel_distance(other_segment)
            + self.perpendicular_distance(other_segment)
        )

    def __repr__(self) -> str:
        return f"Segment(start={self.start}, end={self.end})"


def compare(
    segment_a: Segment,
    segment_b: Segment,
) -> Tuple[Segment, Segment]:
    """Order two segments so the longer one comes first.

    Args:
        segment_a: First segment.
        segment_b: Second segment.

    Returns:
        Tuple of (longer_segment, shorter_segment).
    """
    if segment_a.length > segment_b.length:
        return segment_a, segment_b
    return segment_b, segment_a
