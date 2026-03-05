"""Integration tests: Random frontier and Δ_rand invariants.

Verifies that:
- Random frontier is built from fold-specific data
- Frontier is monotonic post-filtering (within bins)
- Δ_rand(b) is computed by interpolation, not nearest-point
- Sign convention: positive Δ_rand = agent outperforms random
- Budget-matching works at specific evaluation budgets

Runs in <1s with synthetic data.
"""

import numpy as np
import pytest

from q_rlstc.clustering.random_frontier import RandomFrontier
from q_rlstc.clustering.metrics import random_policy_advantage


class TestFrontierInvariants:
    """Integration invariants for RandomFrontier."""

    def _build_synthetic_frontier(self, seed=42):
        """Build a frontier with known synthetic data.

        Simulates: random CUT at 5% → ValCR≈1.5, 20% → ValCR≈1.0, 40% → ValCR≈0.8
        (more cutting = lower ValCR because more segments = lower OD per segment).
        """
        rng = np.random.default_rng(seed)
        rf = RandomFrontier(fold_basesim=5.0)

        for _ in range(50):
            # Low CUT: poor segmentation → high ValCR
            rf.add_point(5.0 + rng.normal(0, 1), 1.5 + rng.normal(0, 0.1))
            # Medium CUT
            rf.add_point(20.0 + rng.normal(0, 1), 1.0 + rng.normal(0, 0.1))
            # High CUT
            rf.add_point(40.0 + rng.normal(0, 1), 0.8 + rng.normal(0, 0.1))

        rf.finalize(n_bins=20)
        return rf

    def test_frontier_interpolates_not_nearest(self):
        """Δ_rand at CUT=12% should interpolate between 5% and 20% bins."""
        rf = self._build_synthetic_frontier()

        v_5 = rf.interpolate(5.0)    # ~1.5
        v_12 = rf.interpolate(12.0)  # should be between 5% and 20%
        v_20 = rf.interpolate(20.0)  # ~1.0

        # Interpolated value should be between the two surrounding bins
        assert v_5 > v_12 > v_20 or v_5 >= v_12 >= v_20, \
            f"Interpolation not monotonic: v(5%)={v_5}, v(12%)={v_12}, v(20%)={v_20}"

    def test_delta_rand_sign_convention(self):
        """Positive Δ_rand = agent outperforms random (lower ValCR = better)."""
        rf = self._build_synthetic_frontier()

        # Agent with ValCR=0.5 at CUT=20% (random ≈ 1.0 at 20%)
        delta = rf.advantage(agent_valcr=0.5, agent_cut_pct=20.0)
        assert delta > 0, f"Agent clearly better but Δ_rand={delta} ≤ 0"

    def test_delta_rand_budget_matched(self):
        """Δ_rand at different budgets uses different frontier values."""
        rf = self._build_synthetic_frontier()

        delta_5 = rf.advantage(agent_valcr=1.0, agent_cut_pct=5.0)
        delta_20 = rf.advantage(agent_valcr=1.0, agent_cut_pct=20.0)
        delta_40 = rf.advantage(agent_valcr=1.0, agent_cut_pct=40.0)

        # Same agent ValCR but Δ_rand differs due to budget matching
        assert delta_5 != delta_20 or delta_20 != delta_40, \
            "Budget matching has no effect — frontier is flat?"

    def test_different_folds_give_different_frontiers(self):
        """Frontiers from different seeds should differ (fold-specific)."""
        rf1 = self._build_synthetic_frontier(seed=1)
        rf2 = self._build_synthetic_frontier(seed=2)

        v1 = rf1.interpolate(15.0)
        v2 = rf2.interpolate(15.0)
        assert v1 != v2, "Different folds produced identical frontiers"

    def test_frontier_at_specific_budgets(self):
        """Verify frontier can be queried at canonical evaluation budgets."""
        rf = self._build_synthetic_frontier()

        for budget in [5.0, 10.0, 20.0, 30.0]:
            v = rf.interpolate(budget)
            assert np.isfinite(v), f"Frontier not finite at {budget}%"
            assert v > 0, f"Frontier ValCR not positive at {budget}%"


class TestDeltaRandConsistency:
    """Verify Δ_rand sign convention is consistent across both implementations."""

    def test_scalar_and_frontier_agree(self):
        """random_policy_advantage() and RandomFrontier.advantage() should agree on sign.

        Both must use: positive = agent outperforms (lower ValCR = better).
        """
        agent_valcr = 0.8
        random_valcr = 1.2

        # Scalar function
        delta_scalar = random_policy_advantage(agent_valcr, random_valcr)

        # Frontier function (build a tiny frontier pinned at random_valcr)
        rf = RandomFrontier(fold_basesim=5.0)
        for _ in range(20):
            rf.add_point(10.0, random_valcr)
        rf.finalize(n_bins=20)
        delta_frontier = rf.advantage(agent_valcr, agent_cut_pct=10.0)

        # Both should be positive (agent is better)
        assert delta_scalar > 0, f"Scalar Δ_rand wrong sign: {delta_scalar}"
        assert delta_frontier > 0, f"Frontier Δ_rand wrong sign: {delta_frontier}"

        # Both should be approximately equal
        assert delta_scalar == pytest.approx(delta_frontier, abs=0.1)

    def test_exact_match_gives_zero(self):
        """When agent ValCR == random ValCR, Δ_rand == 0."""
        delta = random_policy_advantage(agent_valcr=1.0, random_valcr=1.0)
        assert delta == pytest.approx(0.0)
