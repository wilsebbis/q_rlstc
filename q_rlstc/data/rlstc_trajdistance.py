"""Trajectory distance computations — IED, Fréchet, DTW, and segment distance.

This module implements several trajectory distance metrics used by
the RLSTCcode clustering pipeline:

- **IED (Integrated Euclidean Distance):** A time-aware distance that
  integrates point-wise Euclidean distance over the overlapping temporal
  window of two trajectories.  Handles non-overlapping tails by measuring
  distance to the nearest static endpoint.

- **Fréchet distance:** The "dog-walking" distance — the minimum leash
  length needed for a person and dog to traverse two curves simultaneously.

- **DTW (Dynamic Time Warping):** Classic elastic distance metric that
  allows non-linear alignment of two point sequences.

- **Segment (TRACLUS) distance:** Perpendicular + parallel + angle distance
  between endpoint-defined line segments.

- **MDL trajectory simplification cost:** Used by the preprocessing pipeline
  to decide when to merge consecutive points.

See Also:
    :mod:`clustering.trajdistance` — the q_rlstc-native IED implementation.
"""

import math
from typing import List, Optional, Tuple

import numpy as np

from .rlstc_point import Point
from .rlstc_point_xy import Point_xy
from .rlstc_segment import Segment, compare
from .rlstc_traj import Traj

# Minimum segment length below which distance contributions are zero.
SEGMENT_LENGTH_EPSILON: float = 1e-12


# ─── Interpolation helper ─────────────────────────────────────────────


def makemid(
    x_start: float,
    t_start: float,
    x_end: float,
    t_end: float,
    t_query: float,
) -> float:
    """Linearly interpolate a coordinate at a given timestamp.

    Given two points ``(t_start, x_start)`` and ``(t_end, x_end)``,
    returns the x-value at ``t_query`` via linear interpolation.

    Args:
        x_start: Coordinate value at ``t_start``.
        t_start: Timestamp at the start.
        x_end: Coordinate value at ``t_end``.
        t_end: Timestamp at the end.
        t_query: Timestamp to interpolate at.

    Returns:
        Interpolated coordinate value at ``t_query``.
    """
    return x_start + (t_query - t_start) / (t_end - t_start) * (x_end - x_start)


# ─── MDL simplification cost ──────────────────────────────────────────


def traj_mdl_comp(
    points: List[Point],
    start_index: int,
    curr_index: int,
    typed: str,
) -> float:
    """Compute the MDL (Minimum Description Length) cost for a trajectory segment.

    Used by :func:`data.preprocessing.simplify` to decide whether to keep
    or merge consecutive trajectory points.

    Two modes:
    - ``"simp"`` (simplified): Cost of representing ``points[start:curr+1]``
      as a single line segment from ``start`` to ``curr``.  Measures the
      log-length of the shortcut plus the reconstruction error (sum of
      per-point distances to the interpolated line).
    - ``"orign"`` (original): Cost of representing the points as their
      original piecewise-linear chain.  Sums ``log2(segment_length)``
      for each consecutive pair.

    Args:
        points: Full trajectory point list.
        start_index: First index of the segment.
        curr_index: Last index of the segment.
        typed: ``"simp"`` for simplified cost, ``"orign"`` for original cost.

    Returns:
        MDL cost (lower = more compressible).
    """
    shortcut = Segment(points[start_index], points[curr_index])
    header_cost = 0.0
    reconstruction_error = 0.0

    if typed == "simp":
        if shortcut.length > SEGMENT_LENGTH_EPSILON:
            time_span = abs(points[start_index].t - points[curr_index].t)
            header_cost = 0.5 * math.log2(shortcut.length) + 0.5 * time_span

    start_t = points[start_index].t
    end_t = points[curr_index].t
    start_x = points[start_index].x
    end_x = points[curr_index].x
    start_y = points[start_index].y
    end_y = points[curr_index].y

    for idx in range(start_index, curr_index):
        if typed == "simp":
            interp_x = makemid(start_x, start_t, end_x, end_t, points[idx].t)
            interp_y = makemid(start_y, start_t, end_y, end_t, points[idx].t)
            interpolated_point = Point(interp_x, interp_y, points[idx].t)
            reconstruction_error += points[idx].distance(interpolated_point)
        elif typed == "orign":
            pair_dist = 0.5 * points[idx].distance(points[idx + 1]) + 0.5 * abs(
                points[idx].t - points[idx + 1].t
            )
            if pair_dist > SEGMENT_LENGTH_EPSILON:
                header_cost += math.log2(pair_dist)

    if typed == "simp" and reconstruction_error > SEGMENT_LENGTH_EPSILON:
        header_cost += math.log2(reconstruction_error)

    return header_cost


# ─── Time-windowed trajectory extraction ───────────────────────────────


def timedTraj(
    points: List[Point],
    start_time: float,
    end_time: float,
) -> Optional[Traj]:
    """Extract the sub-trajectory within a time window [start_time, end_time].

    If the window boundaries fall between existing points, new interpolated
    boundary points are inserted so the returned trajectory starts and ends
    at exactly ``start_time`` and ``end_time``.

    Args:
        points: Ordered list of trajectory points (monotonic timestamps).
        start_time: Start of the extraction window.
        end_time: End of the extraction window.

    Returns:
        A new :class:`Traj` spanning ``[start_time, end_time]``,
        or ``None`` if the window does not overlap with the trajectory.
    """
    if start_time == end_time:
        return None
    if start_time > points[-1].t or end_time < points[0].t:
        return None

    # Find first point at or after start_time
    first_idx = 0
    while points[first_idx].t < start_time:
        first_idx += 1

    # Find last point at or before end_time
    last_idx = len(points) - 1
    while points[last_idx].t > end_time:
        last_idx -= 1

    new_points: List[Point] = []

    # Interpolate a boundary point at start_time if needed
    if first_idx != 0 and points[first_idx].t != start_time:
        interp_x = makemid(
            points[first_idx - 1].x, points[first_idx - 1].t,
            points[first_idx].x, points[first_idx].t,
            start_time,
        )
        interp_y = makemid(
            points[first_idx - 1].y, points[first_idx - 1].t,
            points[first_idx].y, points[first_idx].t,
            start_time,
        )
        new_points.append(Point(interp_x, interp_y, start_time))

    # Copy all interior points
    for idx in range(first_idx, last_idx + 1):
        new_points.append(points[idx])

    # Interpolate a boundary point at end_time if needed
    if last_idx != len(points) - 1 and points[last_idx].t != end_time:
        interp_x = makemid(
            points[last_idx].x, points[last_idx].t,
            points[last_idx + 1].x, points[last_idx + 1].t,
            end_time,
        )
        interp_y = makemid(
            points[last_idx].y, points[last_idx].t,
            points[last_idx + 1].y, points[last_idx + 1].t,
            end_time,
        )
        new_points.append(Point(interp_x, interp_y, end_time))

    return Traj(new_points, len(new_points), new_points[0].t, new_points[-1].t)


# ─── IED (Integrated Euclidean Distance) ───────────────────────────────


def line2lineIDE(
    point1_start: Point,
    point1_end: Point,
    point2_start: Point,
    point2_end: Point,
) -> float:
    """Integrated Euclidean Distance between two temporally-aligned line segments.

    Computes the trapezoidal approximation of the area between two line
    segments that share the same time interval:

        d = 0.5 × (dist(start₁, start₂) + dist(end₁, end₂)) × Δt

    Args:
        point1_start: Start point of the first segment.
        point1_end: End point of the first segment.
        point2_start: Start point of the second segment.
        point2_end: End point of the second segment.

    Returns:
        Non-negative integrated distance value.
    """
    dist_at_start = point1_start.distance(point2_start)
    dist_at_end = point1_end.distance(point2_end)
    time_span = point1_end.t - point1_start.t
    return 0.5 * (dist_at_start + dist_at_end) * time_span


def getstaticIED(
    points: List[Point],
    static_x: float,
    static_y: float,
    time_start: float,
    time_end: float,
) -> float:
    """IED from a trajectory to a static point over a time window.

    Used for the non-overlapping "tail" portions of two trajectories.
    Treats the static point as a zero-length trajectory at ``(static_x,
    static_y)`` and integrates the distance over ``[time_start, time_end]``.

    Args:
        points: Trajectory points.
        static_x: X-coordinate of the static reference point.
        static_y: Y-coordinate of the static reference point.
        time_start: Start of the integration window (must be < time_end).
        time_end: End of the integration window.

    Returns:
        Integrated distance, or ``1e10`` if no temporal overlap exists.
    """
    overlap_start = max(points[0].t, time_start)
    overlap_end = min(points[-1].t, time_end)
    total_distance = 0.0

    if overlap_start >= overlap_end:
        return 1e10

    static_start = Point(static_x, static_y, 0)
    static_end = Point(static_x, static_y, 0)

    windowed_traj = timedTraj(points, overlap_start, overlap_end)
    for idx in range(windowed_traj.size - 1):
        static_start.t = windowed_traj.points[idx].t
        static_end.t = windowed_traj.points[idx + 1].t
        segment_dist = line2lineIDE(
            windowed_traj.points[idx], windowed_traj.points[idx + 1],
            static_start, static_end,
        )
        total_distance += segment_dist

    return total_distance


def traj2trajIED(
    traj_points_a: List[Point],
    traj_points_b: List[Point],
) -> float:
    """Full IED between two trajectories.

    Computed in three parts:

    1. **Non-overlapping tails:** Where only one trajectory has data, the
       distance to the other's nearest endpoint is integrated.
    2. **Overlapping region:** Both trajectories are clipped to the common
       time window.  Points are merged into a joint timeline and the
       trapezoidal IED is summed over each micro-segment.

    Args:
        traj_points_a: Point list of the first trajectory.
        traj_points_b: Point list of the second trajectory.

    Returns:
        Total integrated Euclidean distance, or ``1e10`` if the
        trajectories have no temporal overlap.
    """
    time_a_start = traj_points_a[0].t
    time_a_end = traj_points_a[-1].t
    time_b_start = traj_points_b[0].t
    time_b_end = traj_points_b[-1].t

    # No temporal overlap at all
    if time_a_start >= time_b_end or time_a_end <= time_b_start:
        return 1e10

    total_distance = 0.0

    # Clip trajectory B to the time range of A
    timed_b = timedTraj(traj_points_b, time_a_start, time_a_end)
    common_start = timed_b.ts
    common_end = timed_b.te

    # Clip trajectory A to the common time range
    common_a = timedTraj(traj_points_a, common_start, common_end)

    # ── Non-overlapping tails ──────────────────────────────────────
    if time_a_start < common_start:
        total_distance += getstaticIED(
            traj_points_a,
            timed_b.points[0].x, timed_b.points[0].y,
            time_a_start, common_start,
        )
    if time_b_start < time_a_start:
        total_distance += getstaticIED(
            traj_points_b,
            traj_points_a[0].x, traj_points_a[0].y,
            time_b_start, time_a_start,
        )
    if time_a_end > common_end:
        total_distance += getstaticIED(
            traj_points_a,
            timed_b.points[-1].x, timed_b.points[-1].y,
            common_end, time_a_end,
        )
    if time_a_end < time_b_end:
        total_distance += getstaticIED(
            traj_points_b,
            traj_points_a[-1].x, traj_points_a[-1].y,
            time_a_end, time_b_end,
        )

    # ── Overlapping region: merge timelines ────────────────────────
    if common_a is not None and common_a.size != 0:
        current_time = common_a.ts
        iter_a = 0  # index into common_a
        iter_b = 0  # index into timed_b
        prev_point_a = common_a.points[0]
        prev_point_b = timed_b.points[0]

        while current_time != timed_b.te:
            next_time_b = timed_b.points[iter_b + 1].t
            next_time_a = common_a.points[iter_a + 1].t

            if next_time_b == next_time_a:
                # Both trajectories have a point at the same time
                next_point_a = common_a.points[iter_a + 1]
                next_point_b = timed_b.points[iter_b + 1]
                iter_a += 1
                iter_b += 1
                new_time = next_time_b
            elif next_time_b < next_time_a:
                # B has the next point — interpolate A at that time
                t_interp = timed_b.points[iter_b + 1].t
                interp_x = makemid(
                    common_a.points[iter_a].x, common_a.points[iter_a].t,
                    common_a.points[iter_a + 1].x, common_a.points[iter_a + 1].t,
                    t_interp,
                )
                interp_y = makemid(
                    common_a.points[iter_a].y, common_a.points[iter_a].t,
                    common_a.points[iter_a + 1].y, common_a.points[iter_a + 1].t,
                    t_interp,
                )
                next_point_a = Point(interp_x, interp_y, t_interp)
                next_point_b = timed_b.points[iter_b + 1]
                iter_b += 1
                new_time = t_interp
            else:
                # A has the next point — interpolate B at that time
                t_interp = common_a.points[iter_a + 1].t
                interp_x = makemid(
                    timed_b.points[iter_b].x, timed_b.points[iter_b].t,
                    timed_b.points[iter_b + 1].x, timed_b.points[iter_b + 1].t,
                    t_interp,
                )
                interp_y = makemid(
                    timed_b.points[iter_b].y, timed_b.points[iter_b].t,
                    timed_b.points[iter_b + 1].y, timed_b.points[iter_b + 1].t,
                    t_interp,
                )
                next_point_b = Point(interp_x, interp_y, t_interp)
                next_point_a = common_a.points[iter_a + 1]
                iter_a += 1
                new_time = t_interp

            current_time = new_time
            segment_dist = line2lineIDE(prev_point_a, next_point_a, prev_point_b, next_point_b)
            total_distance += segment_dist
            prev_point_a = next_point_a
            prev_point_b = next_point_b

    return total_distance


# ─── Fréchet Distance ──────────────────────────────────────────────────


class Distance:
    """Dynamic-programming computer for discrete Fréchet distance.

    Pre-allocates the DP table for trajectories of known lengths,
    then computes the Fréchet distance via :meth:`FRECHET`.

    Attributes:
        D0: Padded (N+1) × (M+1) DP cost matrix.
        flag: (N, M) matrix tracking which cells have been computed.
        D: View into D0[1:, 1:] — the active DP region.
    """

    def __init__(self, num_points_c: int, num_points_q: int) -> None:
        """Initialize the DP table.

        Args:
            num_points_c: Length of trajectory C.
            num_points_q: Length of trajectory Q.
        """
        self.D0: np.ndarray = np.zeros((num_points_c + 1, num_points_q + 1))
        self.flag: np.ndarray = np.zeros((num_points_c, num_points_q))
        self.D0[0, 1:] = np.inf
        self.D0[1:, 0] = np.inf
        self.D: np.ndarray = self.D0[1:, 1:]

    def FRECHET(
        self,
        traj_c: List[Point],
        traj_q: List[Point],
        skip: Optional[List[int]] = None,
    ) -> float:
        """Compute discrete Fréchet distance between two point lists.

        The Fréchet distance is the minimum "leash length" needed for
        a person and dog to walk along trajectories C and Q simultaneously,
        only moving forward.

        Args:
            traj_c: Points of trajectory C.
            traj_q: Points of trajectory Q.
            skip: Reserved for future use (indices to skip).

        Returns:
            Discrete Fréchet distance.
        """
        if skip is None:
            skip = []
        num_c = len(traj_c)
        num_q = len(traj_q)
        for i in range(num_c):
            for j in range(num_q):
                if self.flag[i, j] == 0:
                    pointwise_cost = traj_c[i].distance(traj_q[j])
                    best_predecessor = min(
                        self.D0[i, j],
                        self.D0[i, j + 1],
                        self.D0[i + 1, j],
                    )
                    self.D[i, j] = max(pointwise_cost, best_predecessor)
                    self.flag[i, j] = 1
        return float(self.D[num_c - 1, num_q - 1])


# ─── Dynamic Time Warping ─────────────────────────────────────────────


class Dtwdistance:
    """Dynamic-programming computer for DTW (Dynamic Time Warping) distance.

    Pre-allocates the DP table for known trajectory lengths.

    Attributes:
        D0: Padded (N+1) × (M+1) DP cost matrix.
        flag: (N, M) matrix tracking which cells have been computed.
        D: View into D0[1:, 1:] — the active DP region.
    """

    def __init__(self, num_points_c: int, num_points_q: int) -> None:
        """Initialize the DTW DP table.

        Args:
            num_points_c: Length of trajectory C.
            num_points_q: Length of trajectory Q.
        """
        self.D0: np.ndarray = np.zeros((num_points_c + 1, num_points_q + 1))
        self.flag: np.ndarray = np.zeros((num_points_c, num_points_q))
        self.D0[0, 1:] = np.inf
        self.D0[1:, 0] = np.inf
        self.D: np.ndarray = self.D0[1:, 1:]

    def DTW(
        self,
        traj_c: List[Point],
        traj_q: List[Point],
        skip: Optional[List[int]] = None,
    ) -> float:
        """Compute DTW distance between two point lists.

        DTW allows non-linear alignment: each point in C can be matched
        to one or more points in Q (and vice versa).  The total cost
        is the sum of all matched pairwise Euclidean distances.

        Args:
            traj_c: Points of trajectory C.
            traj_q: Points of trajectory Q.
            skip: Reserved for future use (indices to skip).

        Returns:
            DTW distance (cumulative aligned cost).
        """
        if skip is None:
            skip = []
        num_c = len(traj_c)
        num_q = len(traj_q)
        for i in range(num_c):
            for j in range(num_q):
                if self.flag[i, j] == 0:
                    delta = np.array([
                        traj_c[i].x - traj_q[j].x,
                        traj_c[i].y - traj_q[j].y,
                    ])
                    pointwise_cost = float(np.linalg.norm(delta))
                    best_predecessor = min(
                        self.D0[i, j],
                        self.D0[i, j + 1],
                        self.D0[i + 1, j],
                    )
                    self.D[i, j] = pointwise_cost + best_predecessor
                    self.flag[i, j] = 1
        return float(self.D[num_c - 1, num_q - 1])


# ─── Segment (TRACLUS) distance ───────────────────────────────────────


def wd_dist(
    traj_a: List[Point],
    traj_b: List[Point],
) -> float:
    """TRACLUS-style segment distance between two trajectories.

    Treats each trajectory as a single line segment from its first to
    last point, then computes the combined perpendicular + parallel +
    angle distance (with the longer segment as the reference).

    Args:
        traj_a: Points of the first trajectory.
        traj_b: Points of the second trajectory.

    Returns:
        TRACLUS segment distance.
    """
    seg_a = Segment(
        Point_xy(traj_a[0].x, traj_a[0].y),
        Point_xy(traj_a[-1].x, traj_a[-1].y),
    )
    seg_b = Segment(
        Point_xy(traj_b[0].x, traj_b[0].y),
        Point_xy(traj_b[-1].x, traj_b[-1].y),
    )
    longer_seg, shorter_seg = compare(seg_a, seg_b)
    return longer_seg.get_all_distance(shorter_seg)
