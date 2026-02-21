"""Inter-point Euclidean Distance (IED) for trajectory comparison.

Port of RLSTCcode's ``trajdistance.py`` into Q-RLSTC's clustering module.
IED is the primary distance metric used in the RLSTC paper for measuring
similarity between sub-trajectories and cluster centers.

Key functions:
    - ``traj2traj_ied``: Full IED between two trajectory point sequences
    - ``incremental_ied``: IED computed incrementally (used in MDP step)
    - ``line2line_ied``: Segment-pair distance building block
    - ``traj_mdl_comp``: MDL cost for trajectory simplification
"""

import math
from typing import List, Optional, Dict, Any
from collections import defaultdict

import numpy as np

from ..data.synthetic import Point, Trajectory


# ---------------------------------------------------------------------------
# Interpolation helper
# ---------------------------------------------------------------------------

def makemid(x1: float, t1: float, x2: float, t2: float, t: float) -> float:
    """Linear interpolation: value at time *t* between (x1,t1) and (x2,t2)."""
    if abs(t2 - t1) < 1e-15:
        return x1
    return x1 + (t - t1) / (t2 - t1) * (x2 - x1)


# ---------------------------------------------------------------------------
# Time-windowed trajectory extraction
# ---------------------------------------------------------------------------

def timed_traj(
    points: List[Point],
    ts: float,
    te: float,
) -> Optional[Trajectory]:
    """Extract sub-trajectory within time window [ts, te].

    Interpolates start/end points if boundaries fall between existing points.

    Args:
        points: Trajectory point list.
        ts: Start time.
        te: End time.

    Returns:
        New Trajectory within the time window, or None if empty.
    """
    if ts >= te:
        return None
    if ts > points[-1].t or te < points[0].t:
        return None

    s_i = 0
    e_i = len(points) - 1
    new_points: List[Point] = []

    while s_i < len(points) and points[s_i].t < ts:
        s_i += 1
    while e_i >= 0 and points[e_i].t > te:
        e_i -= 1

    # Interpolate start if needed
    if s_i != 0 and points[s_i].t != ts:
        x = makemid(points[s_i - 1].x, points[s_i - 1].t,
                     points[s_i].x, points[s_i].t, ts)
        y = makemid(points[s_i - 1].y, points[s_i - 1].t,
                     points[s_i].y, points[s_i].t, ts)
        new_points.append(Point(x=x, y=y, t=ts))

    for i in range(s_i, e_i + 1):
        new_points.append(points[i])

    # Interpolate end if needed
    if e_i != len(points) - 1 and points[e_i].t != te:
        x = makemid(points[e_i].x, points[e_i].t,
                     points[e_i + 1].x, points[e_i + 1].t, te)
        y = makemid(points[e_i].y, points[e_i].t,
                     points[e_i + 1].y, points[e_i + 1].t, te)
        new_points.append(Point(x=x, y=y, t=te))

    if len(new_points) < 2:
        return None

    return Trajectory(points=new_points)


# ---------------------------------------------------------------------------
# Segment-pair IED
# ---------------------------------------------------------------------------

def line2line_ied(p1s: Point, p1e: Point, p2s: Point, p2e: Point) -> float:
    """IED between two line segments sharing the same time interval.

    d = 0.5 * (d(p1s,p2s) + d(p1e,p2e)) * (t_e - t_s)
    """
    d1 = p1s.distance(p2s)
    d2 = p1e.distance(p2e)
    dt = abs(p1e.t - p1s.t)
    return 0.5 * (d1 + d2) * dt


def get_static_ied(
    points: List[Point],
    x: float,
    y: float,
    t1: float,
    t2: float,
) -> float:
    """IED from trajectory to a static point (x,y) over time [t1,t2].

    Used for the non-overlapping time portions of two trajectories.
    """
    s_t = max(points[0].t, t1)
    e_t = min(points[-1].t, t2)
    if s_t >= e_t:
        return 1e10

    timed = timed_traj(points, s_t, e_t)
    if timed is None or timed.size < 2:
        return 1e10

    total = 0.0
    ps = Point(x=x, y=y, t=0.0)
    pe = Point(x=x, y=y, t=0.0)

    for i in range(timed.size - 1):
        ps = Point(x=x, y=y, t=timed.points[i].t)
        pe = Point(x=x, y=y, t=timed.points[i + 1].t)
        total += line2line_ied(timed.points[i], timed.points[i + 1], ps, pe)

    return total


# ---------------------------------------------------------------------------
# Full trajectory-to-trajectory IED
# ---------------------------------------------------------------------------

def traj2traj_ied(traj1_pts: List[Point], traj2_pts: List[Point]) -> float:
    """Compute Inter-point Euclidean Distance between two trajectories.

    This is the primary distance metric from the RLSTC paper.  It accounts
    for temporal alignment and handles non-overlapping time ranges by
    computing static IED for the tails.

    Args:
        traj1_pts: Points of trajectory 1.
        traj2_pts: Points of trajectory 2.

    Returns:
        IED distance (lower = more similar).  Returns 1e10 if no overlap.
    """
    t1s = traj1_pts[0].t
    t1e = traj1_pts[-1].t
    t2s = traj2_pts[0].t
    t2e = traj2_pts[-1].t

    if t1s >= t2e or t1e <= t2s:
        return 1e10

    total = 0.0

    # Timed sub-trajectory of traj2 within traj1's time range
    timed = timed_traj(traj2_pts, t1s, t1e)
    if timed is None:
        return 1e10

    cut1 = timed.ts
    cut2 = timed.te

    # Common sub-trajectory of traj1 within the overlap
    common = timed_traj(traj1_pts, cut1, cut2)

    # Non-overlapping tails
    if t1s < cut1:
        total += get_static_ied(
            traj1_pts, timed.points[0].x, timed.points[0].y, t1s, cut1
        )
    if t2s < t1s:
        total += get_static_ied(
            traj2_pts, traj1_pts[0].x, traj1_pts[0].y, t2s, t1s
        )
    if t1e > cut2:
        total += get_static_ied(
            traj1_pts, timed.points[-1].x, timed.points[-1].y, cut2, t1e
        )
    if t1e < t2e:
        total += get_static_ied(
            traj2_pts, traj1_pts[-1].x, traj1_pts[-1].y, t1e, t2e
        )

    # Overlapping region: interleave by timestamp
    if common is not None and common.size > 0:
        iter1 = 0
        iter2 = 0
        lastp1 = common.points[0]
        lastp2 = timed.points[0]
        last_t = common.ts

        while last_t != timed.te:
            if iter2 + 1 >= len(timed.points) or iter1 + 1 >= len(common.points):
                break

            t_next_timed = timed.points[iter2 + 1].t
            t_next_common = common.points[iter1 + 1].t

            if abs(t_next_timed - t_next_common) < 1e-15:
                newp1 = common.points[iter1 + 1]
                newp2 = timed.points[iter2 + 1]
                iter1 += 1
                iter2 += 1
            elif t_next_timed < t_next_common:
                t = t_next_timed
                x = makemid(common.points[iter1].x, common.points[iter1].t,
                            common.points[iter1 + 1].x, common.points[iter1 + 1].t, t)
                y = makemid(common.points[iter1].y, common.points[iter1].t,
                            common.points[iter1 + 1].y, common.points[iter1 + 1].t, t)
                newp1 = Point(x=x, y=y, t=t)
                newp2 = timed.points[iter2 + 1]
                iter2 += 1
            else:
                t = t_next_common
                x = makemid(timed.points[iter2].x, timed.points[iter2].t,
                            timed.points[iter2 + 1].x, timed.points[iter2 + 1].t, t)
                y = makemid(timed.points[iter2].y, timed.points[iter2].t,
                            timed.points[iter2 + 1].y, timed.points[iter2 + 1].t, t)
                newp2 = Point(x=x, y=y, t=t)
                newp1 = common.points[iter1 + 1]
                iter1 += 1

            last_t = newp1.t
            total += line2line_ied(lastp1, newp1, lastp2, newp2)
            lastp1 = newp1
            lastp2 = newp2

    return total


# ---------------------------------------------------------------------------
# Incremental IED (used during RL training)
# ---------------------------------------------------------------------------

def incremental_sp(
    traj1: List[Point],
    traj2: List[Point],
    k_dict: Dict[int, Dict],
    k: int,
) -> Dict[int, Dict]:
    """First-point incremental IED computation.

    Called when the current segment is just starting (first extension).
    """
    t1s, t1e = traj1[0].t, traj1[-1].t
    t2s, t2e = traj2[0].t, traj2[-1].t

    if t1s >= t2e or t1e <= t2s:
        k_dict[k]["mid_dist"] = 1e10
        k_dict[k]["real_dist"] = 1e10
        k_dict[k]["lastp"] = traj2[0]
        k_dict[k]["j"] = 0
        return k_dict

    if t1e >= t2e:
        d = traj2traj_ied(traj1, traj2)
        k_dict[k]["mid_dist"] = d
        k_dict[k]["real_dist"] = d
        k_dict[k]["lastp"] = Point(x=traj2[-1].x, y=traj2[-1].y, t=t1e)
        k_dict[k]["j"] = len(traj2) - 1
        return k_dict

    e_i = len(traj2) - 1
    while traj2[e_i].t > t1e:
        e_i -= 1

    if traj2[e_i].t == t1e:
        lastp = traj2[e_i]
        front = traj2[:e_i + 1]
        mid_dist = traj2traj_ied(traj1, front)
        back = traj2[e_i:]
        back_dist = get_static_ied(back, traj1[-1].x, traj1[-1].y, t1e, t2e)
        k_dict[k]["mid_dist"] = mid_dist
        k_dict[k]["real_dist"] = mid_dist + back_dist
        k_dict[k]["lastp"] = lastp
        k_dict[k]["j"] = e_i

    if traj2[e_i].t < t1e:
        front = traj2[:e_i + 1]
        x = makemid(traj2[e_i].x, traj2[e_i].t, traj2[e_i + 1].x, traj2[e_i + 1].t, t1e)
        y = makemid(traj2[e_i].y, traj2[e_i].t, traj2[e_i + 1].y, traj2[e_i + 1].t, t1e)
        lastp = Point(x=x, y=y, t=t1e)
        front.append(lastp)
        back = traj2[e_i + 1:]
        back.insert(0, lastp)
        mid_dist = traj2traj_ied(traj1, front)
        back_dist = get_static_ied(back, traj1[-1].x, traj1[-1].y, t1e, t2e)
        k_dict[k]["mid_dist"] = mid_dist
        k_dict[k]["real_dist"] = mid_dist + back_dist
        k_dict[k]["lastp"] = lastp
        k_dict[k]["j"] = e_i

    return k_dict


def incremental_ied(
    traj1: List[Point],
    traj2: List[Point],
    k_dict: Dict[int, Dict],
    k: int,
    i: int,
    sp_i: int,
) -> Dict[int, Dict]:
    """Incremental IED computation.

    Dispatches to ``incremental_sp`` for the first extension, otherwise
    computes the incremental update for subsequent points.

    Args:
        traj1: Current segment (sub-trajectory being built).
        traj2: Cluster center trajectory.
        k_dict: Per-cluster distance state dictionary.
        k: Cluster index.
        i: Current point index within traj1.
        sp_i: Split point index.

    Returns:
        Updated k_dict.
    """
    if i == sp_i + 1:
        return incremental_sp(traj1, traj2, k_dict, k)

    # Non-start-point incremental update
    t1e = traj1[-1].t
    t2e = traj2[-1].t

    if k_dict[k]["mid_dist"] == 1e10:
        return incremental_sp(traj1, traj2, k_dict, k)

    temptraj1 = [traj1[i - 1], traj1[i]]

    if t2e == t1e or (t2e < t1e and t2e > traj1[i - 1].t):
        j = k_dict[k]["j"]
        temptraj2 = traj2[j:]
        if traj2[j].t <= k_dict[k]["lastp"].t:
            temptraj2[0] = k_dict[k]["lastp"]
        else:
            temptraj2.insert(0, k_dict[k]["lastp"])

        d = traj2traj_ied(temptraj1, temptraj2)
        k_dict[k]["mid_dist"] += d
        if t2e == t1e:
            k_dict[k]["real_dist"] = k_dict[k]["mid_dist"]
            k_dict[k]["lastp"] = traj2[-1]
        else:
            k_dict[k]["real_dist"] = k_dict[k]["mid_dist"]
            k_dict[k]["lastp"] = Point(x=traj2[-1].x, y=traj2[-1].y, t=t1e)
        k_dict[k]["j"] = len(traj2) - 1
        return k_dict

    if t2e < t1e and t2e <= traj1[i - 1].t:
        newp = Point(x=traj2[-1].x, y=traj2[-1].y, t=t1e)
        d = line2line_ied(traj1[i - 1], traj1[i], k_dict[k]["lastp"], newp)
        k_dict[k]["mid_dist"] += d
        k_dict[k]["real_dist"] = k_dict[k]["mid_dist"]
        k_dict[k]["lastp"] = newp
        k_dict[k]["j"] = len(traj2) - 1
        return k_dict

    if t1e < t2e:
        e_i = len(traj2) - 1
        while traj2[e_i].t > t1e:
            e_i -= 1
        j = k_dict[k]["j"]
        front = traj2[j:e_i + 1]
        if k_dict[k]["lastp"].t >= front[0].t:
            front[0] = k_dict[k]["lastp"]
        else:
            front.insert(0, k_dict[k]["lastp"])

        if traj2[e_i].t == t1e:
            lastp = traj2[e_i]
            mid_dist = traj2traj_ied(temptraj1, front)
            back = traj2[e_i:]
            back_dist = get_static_ied(back, traj1[-1].x, traj1[-1].y, t1e, t2e)
            k_dict[k]["mid_dist"] += mid_dist
            k_dict[k]["real_dist"] = k_dict[k]["mid_dist"] + back_dist
            k_dict[k]["lastp"] = lastp
            k_dict[k]["j"] = e_i
        else:
            x = makemid(traj2[e_i].x, traj2[e_i].t,
                        traj2[e_i + 1].x, traj2[e_i + 1].t, t1e)
            y = makemid(traj2[e_i].y, traj2[e_i].t,
                        traj2[e_i + 1].y, traj2[e_i + 1].t, t1e)
            lastp = Point(x=x, y=y, t=t1e)
            front.append(lastp)
            back = traj2[e_i + 1:]
            back.insert(0, lastp)
            mid_dist = traj2traj_ied(temptraj1, front)
            back_dist = get_static_ied(back, traj1[-1].x, traj1[-1].y, t1e, t2e)
            k_dict[k]["mid_dist"] += mid_dist
            k_dict[k]["real_dist"] = k_dict[k]["mid_dist"] + back_dist
            k_dict[k]["lastp"] = lastp
            k_dict[k]["j"] = e_i

        return k_dict

    return k_dict


# ---------------------------------------------------------------------------
# Min-distance cluster assignment (incremental)
# ---------------------------------------------------------------------------

def incremental_mindist(
    trajectory_points: List[Point],
    start_index: int,
    current_index: int,
    k_dict: Dict[int, Dict],
    cluster_dict: Dict[int, Any],
) -> tuple:
    """Find nearest cluster using incremental IED.

    Args:
        trajectory_points: Full trajectory points.
        start_index: Segment start index.
        current_index: Current point index.
        k_dict: Per-cluster distance state.
        cluster_dict: Cluster dictionary with centers at index [2].

    Returns:
        (min_distance, cluster_id)
    """
    min_dist = 1e10
    cluster_id = -1
    count = 0

    traj_pts = trajectory_points[start_index:current_index + 1]
    s_i = 0
    c_i = current_index - start_index

    for i in cluster_dict.keys():
        center = cluster_dict[i][2]
        if len(center) == 0:
            continue
        k_dict = incremental_ied(traj_pts, center, k_dict, i, c_i, s_i)

        if count == 0:
            min_dist = k_dict[i]["real_dist"]
            cluster_id = i
        else:
            if k_dict[i]["real_dist"] < min_dist:
                min_dist = k_dict[i]["real_dist"]
                cluster_id = i
        count += 1

    return min_dist, cluster_id


# ---------------------------------------------------------------------------
# MDL cost computation (used in preprocessing)
# ---------------------------------------------------------------------------

EPS = 1e-12


def traj_mdl_comp(
    points: List[Point],
    start_index: int,
    curr_index: int,
    mode: str = "simp",
) -> float:
    """Compute MDL cost for a trajectory segment.

    Two modes:
        - ``"simp"``: Cost of representing segment with start-end line.
          H = 0.5*log2(line_len) + 0.5*|dt|, plus log2(perpendicular error).
        - ``"orign"``: Cost of storing original points.
          H = sum of log2(inter-point distances).

    Args:
        points: Trajectory points.
        start_index: Start of segment.
        curr_index: End of segment.
        mode: "simp" or "orign".

    Returns:
        MDL cost.
    """
    p_start = points[start_index]
    p_curr = points[curr_index]

    seg_len = p_start.distance(p_curr)
    h = 0.0
    lh = 0.0

    t1 = p_start.t
    t2 = p_curr.t
    x1, y1 = p_start.x, p_start.y
    x2, y2 = p_curr.x, p_curr.y

    if mode == "simp":
        if seg_len > EPS:
            h = 0.5 * math.log2(seg_len) + 0.5 * abs(t1 - t2)

    for i in range(start_index, curr_index):
        if mode == "simp":
            t = points[i].t
            new_x = makemid(x1, t1, x2, t2, t)
            new_y = makemid(y1, t1, y2, t2, t)
            new_p = Point(x=new_x, y=new_y, t=t)
            lh += points[i].distance(new_p)
        elif mode == "orign":
            d = 0.5 * points[i].distance(points[i + 1]) + 0.5 * abs(points[i].t - points[i + 1].t)
            if d > EPS:
                h += math.log2(d)

    if mode == "simp":
        if lh > EPS:
            h += math.log2(lh)
        return h
    else:
        return h


# ---------------------------------------------------------------------------
# Fréchet and DTW distances (for evaluation / comparison)
# ---------------------------------------------------------------------------

class FrechetDistance:
    """Discrete Fréchet distance between two trajectories."""

    def __init__(self, n: int, m: int):
        self.D0 = np.zeros((n + 1, m + 1))
        self.flag = np.zeros((n, m))
        self.D0[0, 1:] = np.inf
        self.D0[1:, 0] = np.inf
        self.D = self.D0[1:, 1:]

    def compute(self, traj_c: List[Point], traj_q: List[Point]) -> float:
        n = len(traj_c)
        m = len(traj_q)
        for i in range(n):
            for j in range(m):
                if self.flag[i, j] == 0:
                    cost = traj_c[i].distance(traj_q[j])
                    self.D[i, j] = max(cost, min(self.D0[i, j], self.D0[i, j + 1], self.D0[i + 1, j]))
                    self.flag[i, j] = 1
        return float(self.D[n - 1, m - 1])


class DtwDistance:
    """Dynamic Time Warping distance between two trajectories."""

    def __init__(self, n: int, m: int):
        self.D0 = np.zeros((n + 1, m + 1))
        self.flag = np.zeros((n, m))
        self.D0[0, 1:] = np.inf
        self.D0[1:, 0] = np.inf
        self.D = self.D0[1:, 1:]

    def compute(self, traj_c: List[Point], traj_q: List[Point]) -> float:
        n = len(traj_c)
        m = len(traj_q)
        for i in range(n):
            for j in range(m):
                if self.flag[i, j] == 0:
                    cost = math.sqrt((traj_c[i].x - traj_q[j].x) ** 2 +
                                     (traj_c[i].y - traj_q[j].y) ** 2)
                    self.D[i, j] = cost + min(self.D0[i, j], self.D0[i, j + 1], self.D0[i + 1, j])
                    self.flag[i, j] = 1
        return float(self.D[n - 1, m - 1])
