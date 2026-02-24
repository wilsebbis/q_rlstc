"""Trajectory container — an ordered sequence of GPS points.

A :class:`Traj` holds a list of :class:`Point` objects together with
metadata about the trajectory's size and temporal extent.  This is the
primary data structure used by the RLSTCcode-compatible MDP environment
and clustering pipeline.

See Also:
    :class:`Point` — individual GPS point.
    :class:`Segment` — a pair of points forming a directed line segment.
"""

from typing import List, Optional

from .rlstc_point import Point


class Traj:
    """An ordered sequence of GPS points forming a trajectory.

    Attributes:
        points: Ordered list of GPS points in this trajectory.
        size: Number of points (equal to ``len(points)``).
        ts: Timestamp of the first point (trajectory start time).
        te: Timestamp of the last point (trajectory end time).
        traj_id: Optional unique identifier for this trajectory.
    """

    def __init__(
        self,
        points: List[Point],
        size: int,
        ts: float,
        te: float,
        traj_id: Optional[int] = None,
    ) -> None:
        """Initialize a trajectory.

        Args:
            points: Ordered list of GPS points.
            size: Number of points in the trajectory.
            ts: Start timestamp (first point's ``t`` value).
            te: End timestamp (last point's ``t`` value).
            traj_id: Optional integer identifier for this trajectory.
        """
        self.points: List[Point] = points
        self.size: int = size
        self.ts: float = ts
        self.te: float = te
        self.traj_id: Optional[int] = traj_id

    def __repr__(self) -> str:
        return (
            f"Traj(id={self.traj_id}, size={self.size}, "
            f"ts={self.ts:.4f}, te={self.te:.4f})"
        )

    def __len__(self) -> int:
        """Return the number of points in this trajectory."""
        return self.size
