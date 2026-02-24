"""Two-dimensional point for segment geometry calculations.

:class:`Point_xy` is a lightweight 2D point used exclusively for
computing geometric distances between trajectory segments (perpendicular,
parallel, and angle distances).  Unlike :class:`Point`, it does **not**
carry a timestamp — it represents a pure spatial coordinate.

The module also provides :func:`_point2line_distance`, a helper that
computes the perpendicular distance from a point to a line segment
(used in angle-distance computation).

See Also:
    :class:`Point` — full (x, y, t) trajectory point.
    :class:`Segment` — line segment built from two ``Point_xy`` endpoints.
"""

import math
from typing import Union

import numpy as np


class Point_xy:
    """A 2D point supporting arithmetic operators for segment geometry.

    Supports addition, subtraction, scalar multiplication/division,
    Euclidean distance, dot product, and conversion to NumPy array.

    Attributes:
        x: Horizontal coordinate (longitude or normalized x).
        y: Vertical coordinate (latitude or normalized y).
    """

    def __init__(self, x: float, y: float) -> None:
        """Initialize a 2D point.

        Args:
            x: Horizontal coordinate.
            y: Vertical coordinate.
        """
        self.x: float = x
        self.y: float = y

    def get_point(self) -> tuple:
        """Return coordinates as a (x, y) tuple.

        Returns:
            Tuple of (x, y) values.
        """
        return self.x, self.y

    def __add__(self, other: "Point_xy") -> "Point_xy":
        """Vector addition of two 2D points.

        Args:
            other: Point to add.

        Returns:
            New Point_xy with summed coordinates.

        Raises:
            TypeError: If other is not a Point_xy.
        """
        if not isinstance(other, Point_xy):
            raise TypeError("The other type is not 'Point_xy' type.")
        return Point_xy(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point_xy") -> "Point_xy":
        """Vector subtraction of two 2D points.

        Args:
            other: Point to subtract.

        Returns:
            New Point_xy with difference coordinates.

        Raises:
            TypeError: If other is not a Point_xy.
        """
        if not isinstance(other, Point_xy):
            raise TypeError("The other type is not 'Point_xy' type.")
        return Point_xy(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Point_xy":
        """Scalar multiplication.

        Args:
            scalar: Value to multiply both coordinates by.

        Returns:
            New Point_xy with scaled coordinates.

        Raises:
            TypeError: If scalar is not a float.
        """
        if isinstance(scalar, (float, int)):
            return Point_xy(self.x * scalar, self.y * scalar)
        raise TypeError("The scalar must be a numeric type (int or float).")

    def __truediv__(self, scalar: float) -> "Point_xy":
        """Scalar division.

        Args:
            scalar: Value to divide both coordinates by.

        Returns:
            New Point_xy with divided coordinates.

        Raises:
            TypeError: If scalar is not a float.
        """
        if isinstance(scalar, (float, int)):
            return Point_xy(self.x / scalar, self.y / scalar)
        raise TypeError("The scalar must be a numeric type (int or float).")

    def distance(self, other: "Point_xy") -> float:
        """Euclidean distance to another 2D point.

        Args:
            other: The point to measure distance to.

        Returns:
            Euclidean distance sqrt((x₁−x₂)² + (y₁−y₂)²).
        """
        delta_x = self.x - other.x
        delta_y = self.y - other.y
        return math.sqrt(delta_x * delta_x + delta_y * delta_y)

    def dot(self, other: "Point_xy") -> float:
        """Dot product with another 2D point (treated as vectors).

        Args:
            other: The other vector.

        Returns:
            Scalar dot product x₁·x₂ + y₁·y₂.
        """
        return self.x * other.x + self.y * other.y

    def as_array(self) -> np.ndarray:
        """Convert to a NumPy array of shape (2,).

        Returns:
            1D array ``[x, y]``.
        """
        return np.array((self.x, self.y))

    def __repr__(self) -> str:
        return f"Point_xy(x={self.x:.6f}, y={self.y:.6f})"


def _point2line_distance(
    point: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray,
) -> float:
    """Perpendicular distance from a point to a line segment.

    Uses the cross-product formula for point-to-line distance:
        d = ||(end − start) × (start − point)|| / ||end − start||

    If the line segment has zero length (start == end), returns the
    Euclidean distance from the point to that degenerate segment.

    Args:
        point: 2D point as ``np.array([x, y])``.
        line_start: Start of the line segment as ``np.array([x, y])``.
        line_end: End of the line segment as ``np.array([x, y])``.

    Returns:
        Perpendicular distance from the point to the line.
    """
    if np.all(np.equal(line_start, line_end)):
        return float(np.linalg.norm(point - line_start))

    cross_product = np.cross(line_end - line_start, line_start - point)
    line_length = np.linalg.norm(line_end - line_start)
    return float(np.abs(cross_product) / line_length)
