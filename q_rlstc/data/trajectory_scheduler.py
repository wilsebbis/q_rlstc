"""Trajectory Scheduler — controls which trajectories the agent sees each epoch.

Inspired by qDINA's WorkloadManager, this module provides three modes
for controlling the training data distribution:

  standard  — randomized shuffle of all trajectories (current behavior)
  drift     — re-weighted sampling that simulates geographic distribution shift
  low_data  — restricts training to a configurable fraction of trajectories

Usage:
    scheduler = TrajectoryScheduler(
        n_trajectories=100,
        validation_pct=0.1,
        mode="drift",
        data_fraction=0.5,
        seed=42,
    )
    for epoch in range(n_epochs):
        train_indices = scheduler.sample_epoch()
        val_start, val_end = scheduler.validation_range()
        scheduler.update()  # advance drift weights (no-op in other modes)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrajectoryScheduler:
    """Controls trajectory sampling across training epochs.

    Attributes:
        n_trajectories: Total number of trajectories available.
        validation_pct: Fraction held out for validation (default 0.1).
        mode: One of "standard", "drift", "low_data".
        data_fraction: Fraction of training set to use in low_data mode.
        seed: Random seed for reproducibility.
        n_clusters: Number of geographic clusters for drift simulation.
    """

    n_trajectories: int
    validation_pct: float = 0.1
    mode: str = "standard"
    data_fraction: float = 1.0
    seed: int = 42
    n_clusters: int = 5

    # ── Internal state (set in __post_init__) ─────────────────────
    _rng: random.Random = field(init=False, repr=False)
    _train_indices: List[int] = field(init=False, repr=False)
    _val_start: int = field(init=False, repr=False)
    _val_end: int = field(init=False, repr=False)
    _weights: List[float] = field(init=False, repr=False)
    _active_pool: List[int] = field(init=False, repr=False)
    _epoch: int = field(init=False, repr=False, default=0)

    def __post_init__(self):
        if self.mode not in ("standard", "drift", "low_data"):
            raise ValueError(
                f"Invalid mode '{self.mode}'. "
                "Choose from: 'standard', 'drift', 'low_data'"
            )

        self._rng = random.Random(self.seed)

        # Validation range (last validation_pct of trajectories)
        self._val_start = int(self.n_trajectories * (1 - self.validation_pct))
        self._val_end = self.n_trajectories

        # Full training pool
        all_train = list(range(self._val_start))
        self._train_indices = all_train

        # Per-trajectory sampling weights (uniform start)
        self._weights = [1.0] * len(all_train)

        # Active pool: restricted in low_data mode
        if self.mode == "low_data":
            n_active = max(1, int(len(all_train) * self.data_fraction))
            self._active_pool = sorted(
                self._rng.sample(all_train, n_active)
            )
        else:
            self._active_pool = list(all_train)

    # ── Public API ────────────────────────────────────────────────

    def sample_epoch(self) -> List[int]:
        """Return trajectory indices for this epoch's training.

        - standard: shuffled copy of all training indices
        - drift: weighted sample (with replacement) from full pool
        - low_data: shuffled copy of restricted pool
        """
        if self.mode == "standard":
            indices = list(self._active_pool)
            self._rng.shuffle(indices)
            return indices

        elif self.mode == "drift":
            # Weighted sample WITH replacement — simulates non-uniform
            # geographic distribution (some areas become "hot")
            k = len(self._active_pool)
            pool_weights = [self._weights[i] for i in self._active_pool]
            sampled = self._rng.choices(
                self._active_pool, weights=pool_weights, k=k
            )
            return sampled

        elif self.mode == "low_data":
            indices = list(self._active_pool)
            self._rng.shuffle(indices)
            return indices

        # Fallback (should never reach)
        return list(self._active_pool)

    def update(self) -> None:
        """Advance the scheduler state between epochs.

        - standard / low_data: no-op
        - drift: boost weights for a random cluster of trajectories,
          simulating geographic distribution shift (modeled on qDINA's
          WorkloadManager.update_workload()).
        """
        self._epoch += 1

        if self.mode != "drift":
            return

        # Partition training indices into geographic clusters
        # (proxy: contiguous blocks of trajectory IDs)
        n_train = len(self._train_indices)
        cluster_size = max(1, n_train // self.n_clusters)
        target_cluster = self._rng.randint(0, self.n_clusters - 1)

        # Boost weights for the target cluster
        start = target_cluster * cluster_size
        end = min(start + cluster_size, n_train)
        for i in range(start, end):
            self._weights[i] += 1.0

        # Mild decay on ALL weights to prevent runaway dominance
        decay = 0.95
        for i in range(n_train):
            self._weights[i] *= decay

    def validation_range(self) -> tuple:
        """Return (start_idx, end_idx) for validation trajectories."""
        return self._val_start, self._val_end

    @property
    def active_training_size(self) -> int:
        """Number of trajectories in the active training pool."""
        return len(self._active_pool)

    @property
    def epoch(self) -> int:
        """Current epoch counter."""
        return self._epoch

    def summary(self) -> dict:
        """Return a summary dict for logging."""
        return {
            "mode": self.mode,
            "n_trajectories": self.n_trajectories,
            "active_pool_size": len(self._active_pool),
            "data_fraction": self.data_fraction,
            "n_clusters": self.n_clusters if self.mode == "drift" else None,
            "epoch": self._epoch,
        }
