"""Tests for random frontier (budget-matched Δ_rand).

Verifies binning, interpolation, advantage sign convention,
and edge cases (empty frontier, all-NaN bins).
"""

import numpy as np
import pytest

from q_rlstc.clustering.random_frontier import RandomFrontier


class TestRandomFrontier:
    """Tests for RandomFrontier."""

    def _make_frontier(self):
        """Helper: build a frontier with known data."""
        rf = RandomFrontier(fold_basesim=5.0)
        # Add points: CUT 5% → ValCR ~1.2, CUT 15% → ValCR ~0.9
        for _ in range(20):
            rf.add_point(cut_pct=5.0 + np.random.uniform(-1, 1),
                         val_cr=1.2 + np.random.uniform(-0.05, 0.05))
            rf.add_point(cut_pct=15.0 + np.random.uniform(-1, 1),
                         val_cr=0.9 + np.random.uniform(-0.05, 0.05))
            rf.add_point(cut_pct=30.0 + np.random.uniform(-1, 1),
                         val_cr=0.85 + np.random.uniform(-0.05, 0.05))
        rf.finalize(n_bins=20)
        return rf

    def test_build_and_interpolate(self):
        """Built frontier should return finite values."""
        rf = self._make_frontier()
        assert rf.is_built
        v = rf.interpolate(cut_pct=10.0)
        assert np.isfinite(v)
        assert v > 0

    def test_advantage_positive_when_agent_better(self):
        """Agent outperforming random → positive Δ_rand."""
        rf = self._make_frontier()
        # Random at 15% is ~0.9; agent at 0.7 is better (lower)
        delta = rf.advantage(agent_valcr=0.7, agent_cut_pct=15.0)
        assert delta > 0

    def test_advantage_negative_when_agent_worse(self):
        """Agent underperforming random → negative Δ_rand."""
        rf = self._make_frontier()
        # Random at 15% is ~0.9; agent at 1.5 is worse (higher)
        delta = rf.advantage(agent_valcr=1.5, agent_cut_pct=15.0)
        assert delta < 0

    def test_empty_frontier_returns_inf(self):
        """Unbuilt frontier should return inf."""
        rf = RandomFrontier(fold_basesim=5.0)
        assert rf.interpolate(10.0) == float("inf")

    def test_n_points(self):
        """n_points tracks added observations."""
        rf = RandomFrontier(fold_basesim=5.0)
        assert rf.n_points == 0
        rf.add_point(5.0, 1.0)
        rf.add_point(10.0, 0.9)
        assert rf.n_points == 2

    def test_finalize_filters_inf(self):
        """Inf/NaN ValCR values should be filtered during finalize."""
        rf = RandomFrontier(fold_basesim=5.0)
        rf.add_point(5.0, float("inf"))
        rf.add_point(5.0, 1.0)
        rf.add_point(10.0, 0.9)
        rf.finalize()
        assert rf.is_built
        v = rf.interpolate(5.0)
        assert np.isfinite(v)

    def test_interpolation_monotonic_at_boundaries(self):
        """Extrapolation at edges should use boundary values."""
        rf = self._make_frontier()
        v_low = rf.interpolate(0.0)
        v_high = rf.interpolate(100.0)
        # Both should be finite (forward/backward fill handles empty bins)
        assert np.isfinite(v_low)
        assert np.isfinite(v_high)
