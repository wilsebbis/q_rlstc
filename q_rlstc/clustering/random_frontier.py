"""Random-policy frontier for budget-matched Δ_rand computation.

Precomputes the ValCR that a purely random CUT policy would achieve
at each CUT budget level, then exposes interpolation so Δ_rand(b)
can be computed at any evaluation budget b.

Usage:
    frontier = RandomFrontier(env, sidx, eidx, fold_basesim)
    frontier.build(n_samples=200, cut_probs=np.linspace(0.01, 0.50, 20))
    delta_rand = frontier.advantage(agent_valcr=0.85, agent_cut_pct=12.0)

Per user feedback: Δ_rand must be budget-matched (same CUT% ≤ b
constraint), NOT "whatever CUT% the random policy hit that epoch."
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class RandomFrontier:
    """Precomputed random-policy ValCR frontier across CUT budgets.

    For each CUT probability in a grid, runs the random policy multiple
    times through the validation trajectories, collects the resulting
    (CUT%, ValCR) pairs, and bins them.  The frontier is the mean ValCR
    in each CUT-% bin.

    Δ_rand(b) = agent_ValCR(b) − frontier_ValCR(b)

    Negative = agent outperforms random at budget b.
    """

    def __init__(
        self,
        fold_basesim: float,
        epsilon: float = 1e-8,
    ):
        """Initialize frontier (call build() to populate).

        Args:
            fold_basesim: Baseline OD for the validation fold.
            epsilon: Floor for basesim to prevent divide-by-zero.
        """
        self.fold_basesim = fold_basesim
        self.epsilon = epsilon

        # Populated by build()
        self._bin_centers: Optional[np.ndarray] = None
        self._bin_valcrs: Optional[np.ndarray] = None
        self._raw_points: List[Tuple[float, float]] = []

    def add_point(self, cut_pct: float, val_cr: float) -> None:
        """Add a raw (CUT%, ValCR) observation from a random-policy run.

        Use this to populate the frontier incrementally (e.g. during
        existing D1 ValCR-sweep experiments) instead of calling build().
        """
        self._raw_points.append((cut_pct, val_cr))

    def finalize(self, n_bins: int = 20, smoothing: int = 1) -> None:
        """Bin raw points and compute frontier curve.

        Args:
            n_bins: Number of CUT-% bins (0–100%).
            smoothing: If > 1, apply moving average over bins.
        """
        if not self._raw_points:
            return

        pts = np.array(self._raw_points)  # (N, 2): [cut_pct, val_cr]
        cut_pcts = pts[:, 0]
        val_crs = pts[:, 1]

        # Remove inf/nan
        valid = np.isfinite(val_crs)
        cut_pcts = cut_pcts[valid]
        val_crs = val_crs[valid]

        if len(cut_pcts) == 0:
            return

        bin_edges = np.linspace(0, 100, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_means = np.full(n_bins, np.nan)

        for i in range(n_bins):
            mask = (cut_pcts >= bin_edges[i]) & (cut_pcts < bin_edges[i + 1])
            if mask.any():
                bin_means[i] = np.mean(val_crs[mask])

        # Forward-fill NaN bins (extrapolate from nearest populated bin)
        for i in range(1, n_bins):
            if np.isnan(bin_means[i]) and not np.isnan(bin_means[i - 1]):
                bin_means[i] = bin_means[i - 1]
        # Backward-fill
        for i in range(n_bins - 2, -1, -1):
            if np.isnan(bin_means[i]) and not np.isnan(bin_means[i + 1]):
                bin_means[i] = bin_means[i + 1]

        self._bin_centers = bin_centers
        self._bin_valcrs = bin_means

    def interpolate(self, cut_pct: float) -> float:
        """Get frontier ValCR at a specific CUT budget via interpolation.

        Args:
            cut_pct: CUT budget (0–100%).

        Returns:
            Interpolated random-policy ValCR at that budget.
            Returns inf if frontier not yet built.
        """
        if self._bin_centers is None or self._bin_valcrs is None:
            return float("inf")

        # np.interp handles extrapolation via edge values
        valid = ~np.isnan(self._bin_valcrs)
        if not valid.any():
            return float("inf")

        return float(np.interp(
            cut_pct,
            self._bin_centers[valid],
            self._bin_valcrs[valid],
        ))

    def advantage(self, agent_valcr: float, agent_cut_pct: float) -> float:
        """Budget-matched Δ_rand.

        Δ_rand = frontier_ValCR(agent_CUT%) - agent_ValCR

        Positive = agent outperforms random at the same budget.
        Convention: lower ValCR = better, so positive Δ_rand = agent wins.

        Args:
            agent_valcr: Agent's ValCR.
            agent_cut_pct: Agent's CUT% (budget for matching).

        Returns:
            Advantage delta (positive = agent outperforms).
        """
        random_valcr = self.interpolate(agent_cut_pct)
        return random_valcr - agent_valcr

    @property
    def n_points(self) -> int:
        """Number of raw observations."""
        return len(self._raw_points)

    @property
    def is_built(self) -> bool:
        """Whether the frontier has been finalized."""
        return self._bin_centers is not None
