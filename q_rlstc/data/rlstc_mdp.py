"""RLSTCcode MDP environment for trajectory segmentation.

The :class:`TrajRLclus` class implements the Markov Decision Process
(MDP) used to train the RL agent for sub-trajectory clustering.  At each
step the agent observes a 5-dimensional state vector and chooses one of
two actions:

- **Action 0 (EXTEND):** Continue growing the current sub-trajectory
  by including the next point.
- **Action 1 (CUT):** End the current sub-trajectory at this point and
  start a new one.

**State vector** (5 features, unless ``ablate_odb=True`` → 4):
    ``[overall_sim, split_overdist, odb_feature, progress_ratio, remaining_ratio]``

- ``overall_sim`` — running mean IED across all assigned sub-trajectories
- ``split_overdist`` — projected mean IED if the current segment is cut here
- ``odb_feature`` — ``10 × overall_sim`` (ODB = "Overall Distance Boost")
- ``progress_ratio`` — fraction of the trajectory scanned so far
- ``remaining_ratio`` — fraction of the trajectory remaining

**Reward:** ``Δ(overall_sim)`` — decrease in overall distance after a cut.
Positive reward means the cut improved clustering quality.

**Minimum segment length:** Enforced by silently overriding CUT → EXTEND
when the current segment is shorter than ``min_seg_len``.

See Also:
    :mod:`data.rlstc_cluster` — incremental IED and cluster maintenance.
    :mod:`rl.vqdqn_agent` — agent that interacts with this MDP.
"""

import pickle
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .rlstc_cluster import (
    add2clusdict,
    compute_overdist,
    incremental_mindist,
    update_centers,
)
from .rlstc_point import Point
from .rlstc_segment import Segment
from .rlstc_traj import Traj
from .rlstc_trajdistance import traj2trajIED


# ─── Custom unpickler for RLSTCcode pickle files ──────────────────────


class _RLSTCUnpickler(pickle.Unpickler):
    """Redirect legacy RLSTCcode module paths during unpickling.

    Pickle stores the original module path (e.g. ``"traj.Traj"``).
    This unpickler rewrites those imports to point at the
    ``q_rlstc.data.rlstc_*`` modules so old pickle files load correctly.
    """

    _MODULE_MAP: Dict[str, str] = {
        "traj": "q_rlstc.data.rlstc_traj",
        "point": "q_rlstc.data.rlstc_point",
        "segment": "q_rlstc.data.rlstc_segment",
        "point_xy": "q_rlstc.data.rlstc_point_xy",
    }

    def find_class(self, module: str, name: str) -> type:
        """Override module resolution with the redirection map."""
        module = self._MODULE_MAP.get(module, module)
        return super().find_class(module, name)


def _load_pickle(path: str) -> Any:
    """Load an RLSTCcode pickle with module-name redirection.

    Args:
        path: Filesystem path to the pickle file.

    Returns:
        Deserialized Python object.
    """
    with open(path, "rb") as f:
        return _RLSTCUnpickler(f).load()


# ─── MDP Environment ──────────────────────────────────────────────────


class TrajRLclus:
    """MDP environment for RL-based trajectory segmentation and clustering.

    Manages two independent cluster dictionaries — one for **T**raining
    (``clusters_T``) and one for **E**valuation (``clusters_E``) — so
    that evaluation metrics are never contaminated by training updates.

    **Cluster dictionary structure** (per cluster ID):

    ======  ==========================================================
    Index   Contents
    ======  ==========================================================
    ``[0]``  List of IED distances for assigned sub-trajectories
    ``[1]``  List of :class:`Traj` sub-trajectories
    ``[2]``  Center trajectory points (``List[Point]``)
    ``[3]``  Time-point dict for center recomputation
    ``[4]``  List of segment lengths (parallel to ``[0]``)
    ======  ==========================================================

    Attributes:
        n_actions: Number of discrete actions (always 2).
        n_features: State dimension (5 normally, 4 if ablate_odb).
        min_seg_len: Minimum allowed segment length before a CUT.
        ablate_odb: If True, omit the ODB feature (4-dim state).
        clusters_T: Training cluster dictionary.
        clusters_E: Evaluation cluster dictionary.
        trajsdata: List of all candidate trajectories.
        basesim_T: Baseline similarity for training clusters.
        basesim_E: Baseline similarity for evaluation clusters.
    """

    def __init__(
        self,
        cand_train: str,
        base_centers_T: str,
        base_centers_E: str,
        min_seg_len: int = 3,
        ablate_odb: bool = False,
    ) -> None:
        """Initialize the MDP environment from pickle files.

        Args:
            cand_train: Path to pickle file of candidate training trajectories.
            base_centers_T: Path to pickle file of training cluster centers.
            base_centers_E: Path to pickle file of evaluation cluster centers.
            min_seg_len: Minimum sub-trajectory length before CUT is allowed.
                         Clamped to ≥ 1.
            ablate_odb: If True, omit the ODB feature (4-dim state).
        """
        self.n_actions: int = 2
        self.ablate_odb: bool = ablate_odb
        self.n_features: int = 4 if ablate_odb else 5
        self.min_seg_len: int = max(1, min_seg_len)

        # Load training centers to determine K
        centers_T = _load_pickle(base_centers_T)
        centers_data_T = centers_T[0][2]
        num_clusters = len(centers_data_T)

        self.RW: float = 0.0
        self.clusters_T: Dict[int, list] = defaultdict(list)
        self.clusters_E: Dict[int, list] = defaultdict(list)

        # Initialize empty cluster slots
        for cluster_id in range(num_clusters):
            self.clusters_T[cluster_id].append([])  # [0] distances
            self.clusters_T[cluster_id].append([])  # [1] sub-trajectories
        for cluster_id in range(num_clusters):
            self.clusters_E[cluster_id].append([])  # [0] distances
            self.clusters_E[cluster_id].append([])  # [1] sub-trajectories

        self.allsubtrajs_T: List[Traj] = []
        self.allsubtrajs_E: List[Traj] = []
        self.allsubindexes_E: List[List[int]] = []
        self._load(cand_train, base_centers_T, base_centers_E)

    def _load(
        self,
        cand_train: str,
        base_centers_T: str,
        base_centers_E: str,
    ) -> None:
        """Load trajectory data and cluster centers from pickle files.

        Populates ``self.trajsdata``, ``self.basesim_T/E``, and the
        cluster dictionaries with center points and time-point dicts.

        Args:
            cand_train: Path to candidate trajectories pickle.
            base_centers_T: Path to training centers pickle.
            base_centers_E: Path to evaluation centers pickle.
        """
        self.trajsdata: List[Traj] = _load_pickle(cand_train)

        centers_T = _load_pickle(base_centers_T)
        self.basesim_T: float = centers_T[0][1]
        centers_data_T = centers_T[0][2]

        centers_E = _load_pickle(base_centers_E)
        self.basesim_E: float = centers_E[0][1]
        centers_data_E = centers_E[0][2]

        for cluster_id in range(len(centers_data_T)):
            center_points = centers_data_T[cluster_id][1].points
            self.clusters_T[cluster_id].append(center_points)           # [2] center
            self.clusters_T[cluster_id].append(defaultdict(list))       # [3] time-point dict
            self.clusters_T[cluster_id].append([])                      # [4] segment lengths

        for cluster_id in range(len(centers_data_E)):
            center_points = centers_data_E[cluster_id][1].points
            self.clusters_E[cluster_id].append(center_points)           # [2] center
            self.clusters_E[cluster_id].append(defaultdict(list))       # [3] time-point dict
            self.clusters_E[cluster_id].append([])                      # [4] segment lengths

    def reset(
        self,
        episode: int,
        label: str = "T",
    ) -> Tuple[np.ndarray, int]:
        """Reset the environment for a new trajectory (episode).

        Initializes all internal state for scanning trajectory ``episode``:
        split point, incremental distance cache, cluster assignment, etc.

        Args:
            episode: Index into ``self.trajsdata`` for the trajectory
                     to be segmented.
            label: ``"T"`` for training clusters, ``"E"`` for evaluation.

        Returns:
            Tuple of ``(initial_observation, trajectory_length)``.
        """
        self.split_point: int = 0
        self.length: int = self.trajsdata[episode].size
        self.k_dict: Dict[int, Dict[str, Any]] = {}

        num_clusters = len(self.clusters_T)
        for cluster_id in range(num_clusters):
            self.k_dict[cluster_id] = {
                'mid_dist': 1e10,
                'real_dist': 1e10,
                'lastp': Point(0, 0, 0),
                'j': 0,
            }

        self.traj_num: int = 1
        self.minsim: float = 0.0
        self.k: int = 0

        # Compute initial distance to nearest cluster
        active_clusters = self.clusters_T if label == "T" else self.clusters_E
        self.minsim, self.k = incremental_mindist(
            self.trajsdata[episode], self.split_point, 1,
            self.k_dict, active_clusters, episode,
        )
        self.next_minsim: float = self.minsim

        center_points = active_clusters[self.k][2]
        self.overall_sim: float = traj2trajIED(
            self.trajsdata[episode].points, center_points,
        )
        self.split_overdist: float = self.minsim

        # Build initial observation
        observation = self._build_observation(
            index=1,
            split_point=self.split_point,
        )

        self.split: List[int] = []
        self.subtrajindex: List[List[int]] = []
        self.subtraj: List[Traj] = []
        self._seg_start_idx: int = 0  # tracks segment start for L_MIN

        return observation, self.length

    def step(
        self,
        episode: int,
        action: int,
        index: int,
        label: str = "T",
    ) -> Tuple[np.ndarray, float]:
        """Take one step in the MDP.

        The agent chooses to EXTEND (action=0) or CUT (action=1) at
        the given point index.  CUT is silently overridden to EXTEND
        if the minimum segment length constraint would be violated.

        Args:
            episode: Index of the trajectory being segmented.
            action: 0 = EXTEND, 1 = CUT.
            index: Current point index in the trajectory.
            label: ``"T"`` for training, ``"E"`` for evaluation.

        Returns:
            Tuple of ``(observation, reward)``.
            Reward is 0 for EXTEND and ``Δ(overall_sim)`` for CUT.
        """
        active_clusters = self.clusters_T if label == "T" else self.clusters_E

        # ── L_MIN enforcement: silently override CUT → EXTEND ──────
        if action == 1:
            segment_length = index - self._seg_start_idx + 1
            remaining_length = self.length - index
            if segment_length < self.min_seg_len or remaining_length < self.min_seg_len:
                action = 0  # force EXTEND
        self._last_action: int = action

        if action == 0:
            return self._step_extend(episode, index, label, active_clusters)

        if action == 1:
            return self._step_cut(episode, index, label, active_clusters)

        # Should never reach here
        return self._build_observation(index, self.split_point), 0.0

    def _step_extend(
        self,
        episode: int,
        index: int,
        label: str,
        active_clusters: Dict[int, list],
    ) -> Tuple[np.ndarray, float]:
        """Handle EXTEND action (action=0).

        If this is not the last point, update the incremental distance
        to the nearest cluster.  If it IS the last point, finalize the
        current sub-trajectory and assign it to the nearest cluster.

        Args:
            episode: Trajectory index.
            index: Current point index.
            label: ``"T"`` or ``"E"``.
            active_clusters: The cluster dict to update.

        Returns:
            Tuple of ``(observation, reward=0)``.
        """
        if index + 1 != self.length:
            # Not at end — just update incremental distance
            self.next_minsim, self.k = incremental_mindist(
                self.trajsdata[episode], self.split_point, index + 1,
                self.k_dict, active_clusters, episode,
            )
            if self.split_point == 0:
                self.split_overdist = self.next_minsim / self.traj_num
            else:
                self.split_overdist = (
                    (self.overall_sim * self.traj_num + self.next_minsim)
                    / (self.traj_num + 1)
                )
        else:
            # At the last point — finalize the remaining sub-trajectory
            self._finalize_subtraj(episode, index, label, active_clusters)

        observation = self._build_observation(index, self.split_point)
        return observation, 0.0

    def _step_cut(
        self,
        episode: int,
        index: int,
        label: str,
        active_clusters: Dict[int, list],
    ) -> Tuple[np.ndarray, float]:
        """Handle CUT action (action=1).

        Finalizes the current sub-trajectory, assigns it to the nearest
        cluster, computes the reward (improvement in overall_sim), and
        resets the split point for the next segment.

        Args:
            episode: Trajectory index.
            index: Current point index (cut location).
            label: ``"T"`` or ``"E"``.
            active_clusters: The cluster dict to update.

        Returns:
            Tuple of ``(observation, reward)``.
        """
        self.split.append(index)
        self.minsim = self.next_minsim

        # Finalize and assign the sub-trajectory
        self._finalize_subtraj(episode, index, label, active_clusters)

        previous_overall_sim = self.overall_sim
        if self.split_point != 0:
            self.traj_num += 1

        self.overall_sim = self.split_overdist
        self.split_point = index
        self._seg_start_idx = index  # new segment starts here

        # Begin incremental distance for the next segment
        if index + 1 != self.length:
            self.next_minsim, self.k = incremental_mindist(
                self.trajsdata[episode], self.split_point, index + 1,
                self.k_dict, active_clusters, episode,
            )
            self.split_overdist = (
                (self.overall_sim * self.traj_num + self.next_minsim)
                / (self.traj_num + 1)
            )

        observation = self._build_observation(index, self.split_point)
        reward = previous_overall_sim - self.overall_sim
        return observation, reward

    def _finalize_subtraj(
        self,
        episode: int,
        index: int,
        label: str,
        active_clusters: Dict[int, list],
    ) -> None:
        """Create a Traj from the current segment and assign to cluster.

        Args:
            episode: Trajectory index.
            index: End index of the sub-trajectory.
            label: ``"T"`` or ``"E"``.
            active_clusters: Cluster dict to update.
        """
        subtraj_points = self.trajsdata[episode].points[self.split_point:index + 1]
        size = len(subtraj_points)
        start_time = subtraj_points[0].t
        end_time = subtraj_points[-1].t
        subtraj = Traj(subtraj_points, size, start_time, end_time, self.trajsdata[episode].traj_id)

        self.subtrajindex.append([self.split_point, index])
        self.split.append(index)
        self.traj_num += 1

        # Assign to nearest cluster
        active_clusters[self.k][1].append(subtraj)
        if self.next_minsim != 1e10:
            active_clusters[self.k][0].append(self.next_minsim)
            active_clusters[self.k][4].append(subtraj.size)
            add2clusdict(subtraj.points, active_clusters, self.k)

        self.overall_sim = self.split_overdist

    def _build_observation(self, index: int, split_point: int) -> np.ndarray:
        """Build the state observation vector.

        Args:
            index: Current point index.
            split_point: Start index of the current segment.

        Returns:
            Observation array of shape ``(1, n_features)``.
        """
        progress = (index - split_point + 2) / self.length
        remaining = (self.length - (index + 1)) / self.length

        if self.ablate_odb:
            return np.array([
                self.overall_sim,
                self.split_overdist,
                progress,
                remaining,
            ]).reshape(1, -1)
        else:
            return np.array([
                self.overall_sim,
                self.split_overdist,
                self.overall_sim * 10,  # ODB feature
                progress,
                remaining,
            ]).reshape(1, -1)

    def output(self, label: str = "T") -> List[Any]:
        """Return current clustering results.

        Args:
            label: ``"T"`` for training results, ``"E"`` for evaluation.

        Returns:
            List of ``[overall_sim, cumulative_reward, cluster_dict]``.
        """
        if label == "T":
            return [self.overall_sim, self.RW, self.clusters_T]
        return [self.overall_sim, self.RW, self.clusters_E]

    def update_cluster(self, label: str = "T") -> None:
        """Recompute all cluster centers and reset per-epoch accumulators.

        Called at the end of each training epoch.  Recomputes centers
        from accumulated sub-trajectories, then clears the distance
        and sub-trajectory lists for the next epoch.

        Args:
            label: ``"T"`` for training clusters, ``"E"`` for evaluation.
        """
        if label == "T":
            self.basesim_T, self.clusters_T = update_centers(
                self.clusters_T, 3, 0.095,
            )
            for cluster_id in self.clusters_T.keys():
                self.clusters_T[cluster_id][0] = []
                self.clusters_T[cluster_id][1] = []
                self.clusters_T[cluster_id][3] = defaultdict(list)
        if label == "E":
            self.basesim_E, self.clusters_E = update_centers(
                self.clusters_E, 3, 0.095,
            )
            for cluster_id in self.clusters_E.keys():
                self.clusters_E[cluster_id][0] = []
                self.clusters_E[cluster_id][1] = []
                self.clusters_E[cluster_id][3] = defaultdict(list)