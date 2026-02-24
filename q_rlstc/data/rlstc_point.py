"""GPS trajectory point with spatial coordinates and timestamp.

Represents a single point in a GPS trajectory, storing longitude (x),
latitude (y), and a normalized timestamp (t).  Used throughout the
RLSTCcode-compatible data pipeline for distance computation, MDP state
construction, and cluster center representation.

See Also:
    :class:`Point_xy` — 2D-only variant used for segment geometry.
    :class:`Traj` — ordered collection of Points forming a trajectory.
"""

import math
from typing import Any


class Point:
    """A single GPS trajectory point with (x, y, t) coordinates.

    Attributes:
        x: Longitude or normalized x-coordinate.
        y: Latitude or normalized y-coordinate.
        t: Timestamp (typically z-score normalized).
    """

    def __init__(self, x: float, y: float, t: float) -> None:
        """Initialize a trajectory point.

        Args:
            x: Longitude or normalized x-coordinate.
            y: Latitude or normalized y-coordinate.
            t: Timestamp (raw or z-score normalized).
        """
        self.x: float = x
        self.y: float = y
        self.t: float = t

    def distance(self, other: "Point") -> float:
        """Compute Euclidean distance to another point in (x, y) space.

        Note:
            This ignores the temporal dimension — it measures purely
            spatial distance.

        Args:
            other: The point to measure distance to.

        Returns:
            Euclidean distance sqrt((x₁−x₂)² + (y₁−y₂)²).
        """
        delta_x = self.x - other.x
        delta_y = self.y - other.y
        return math.sqrt(delta_x * delta_x + delta_y * delta_y)

    def equal(self, other: "Point") -> bool:
        """Check exact equality on all three coordinates.

        Args:
            other: The point to compare against.

        Returns:
            True if x, y, and t all match exactly.
        """
        return self.x == other.x and self.y == other.y and self.t == other.t

    def __repr__(self) -> str:
        return f"Point(x={self.x:.6f}, y={self.y:.6f}, t={self.t:.6f})"
