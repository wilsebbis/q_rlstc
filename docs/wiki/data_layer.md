# Data Layer

---

## Overview

The **data layer** (`q_rlstc/data/`) contains all trajectory data structures, distance metrics, the MDP environment, and preprocessing utilities. These modules were originally ported from `RLSTCcode/subtrajcluster/` and have been fully documented with type hints and descriptive variable names.

## Module Map

| Module | Purpose |
|--------|---------|
| `rlstc_point.py` | `Point(x, y, t)` — GPS point with spatial coordinates and timestamp |
| `rlstc_point_xy.py` | `Point_xy(x, y)` — 2D point for segment geometry (no timestamp) |
| `rlstc_segment.py` | `Segment` — directed line segment with perpendicular, parallel, and angle distances |
| `rlstc_traj.py` | `Traj` — ordered sequence of Points forming a trajectory |
| `rlstc_trajdistance.py` | IED, Fréchet, DTW, segment distance, and MDL simplification cost |
| `rlstc_cluster.py` | Incremental IED clustering and cluster-center maintenance |
| `rlstc_mdp.py` | `TrajRLclus` — MDP environment for RL-based trajectory segmentation |
| `preprocessing.py` | GPS filtering, length normalization, z-score normalization, MDL simplification |
| `trajectory_scheduler.py` | `TrajectoryScheduler` — controls trajectory sampling across epochs |
| `dataset_loader.py` | `DatasetLoader` (ABC), `CustomDatasetLoader`, `DatasetConfig` — abstract dataset loading interface |

## Class Hierarchy

```
Point(x, y, t)
  └─ Used by: Traj, Segment (via Point_xy), distance functions

Point_xy(x, y)
  └─ Used by: Segment (2D geometry only)

Segment(start: Point_xy, end: Point_xy)
  └─ perpendicular_distance(), parallel_distance(), angle_distance()
  └─ Used by: IED computation, TRACLUS-style distance

Traj(points: List[Point], size, ts, te, traj_id)
  └─ Primary data container for trajectories
  └─ Used by: MDP, clustering, distance functions
```

## Distance Metrics

### IED (Integrated Euclidean Distance)

The primary distance metric. Integrates point-wise Euclidean distance over the overlapping temporal window of two trajectories, plus tail contributions for non-overlapping regions.

```
traj2trajIED(points_a, points_b) → float
```

**Incremental variant:** Instead of recomputing from scratch each time the agent extends a sub-trajectory, `incremental_sp` / `incremental_nsp` cache partial distances in a `k_dict` dictionary.

### Fréchet Distance

The "dog-walking" distance — minimum leash length for two entities traversing curves simultaneously. Computed via dynamic programming.

### DTW (Dynamic Time Warping)

Classic elastic alignment distance. Each point in one trajectory can match multiple points in the other. Uses DP to minimize total aligned cost.

### TRACLUS Segment Distance

Sum of three components between endpoint-defined line segments:
- **Perpendicular** — sideways separation
- **Parallel** — overshoot/undershoot
- **Angle** — directional divergence

## MDP Environment

`TrajRLclus` manages the RL training loop. Key properties:

| Property | Value |
|----------|-------|
| State dimension | 5 (or 4 if `ablate_odb=True`) |
| Actions | 2: EXTEND (0) or CUT (1) |
| Reward | `Δ(overall_sim)` — decrease in mean IED |
| Min segment length | Configurable (`min_seg_len`, default 3) |
| Clusters | Separate T (training) and E (evaluation) dicts |

**State vector:**
```
[overall_sim, split_overdist, ODB_feature, progress_ratio, remaining_ratio]
```

## Preprocessing Pipeline

The `preprocess_pipeline()` function chains 5 steps matching the original RLSTCcode:

```
raw GPS data → filter by bounding box → enforce length limits
→ z-score normalize coordinates → MDL simplify → pickle output
```

---

[→ RL Agents](rl_agents.md) | [→ Architecture](architecture.md)
