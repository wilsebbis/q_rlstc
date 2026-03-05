"""Tests for adaptive shot allocation.

Verifies three-regime shot scheduling, history tracking,
and statistics computation.
"""

import numpy as np
import pytest

from q_rlstc.rl.adaptive_shots import AdaptiveShotScheduler


class TestAdaptiveShotScheduler:
    """Tests for AdaptiveShotScheduler."""

    def test_narrow_margin_high_shots(self):
        """Narrow Q-margin → high shots."""
        s = AdaptiveShotScheduler(
            shots_low=64, shots_default=256, shots_high=1024,
            tau_low=0.2, tau_high=1.0
        )
        assert s.get_shots(q_margin=0.1) == 1024

    def test_wide_margin_low_shots(self):
        """Wide Q-margin → low shots."""
        s = AdaptiveShotScheduler(
            shots_low=64, shots_default=256, shots_high=1024,
            tau_low=0.2, tau_high=1.0
        )
        assert s.get_shots(q_margin=2.0) == 64

    def test_medium_margin_default_shots(self):
        """Medium Q-margin → default shots."""
        s = AdaptiveShotScheduler(
            shots_low=64, shots_default=256, shots_high=1024,
            tau_low=0.2, tau_high=1.0
        )
        assert s.get_shots(q_margin=0.5) == 256

    def test_negative_margin_uses_abs(self):
        """Negative Q-margin handled via abs()."""
        s = AdaptiveShotScheduler(
            shots_low=64, shots_default=256, shots_high=1024,
            tau_low=0.2, tau_high=1.0
        )
        assert s.get_shots(q_margin=-0.1) == 1024

    def test_history_tracking(self):
        """Shot history should track decisions."""
        s = AdaptiveShotScheduler()
        s.get_shots(0.1)
        s.get_shots(0.5)
        s.get_shots(2.0)
        assert len(s._history) == 3

    def test_stats(self):
        """Statistics should report correct values."""
        s = AdaptiveShotScheduler(
            shots_low=64, shots_default=256, shots_high=1024,
            tau_low=0.2, tau_high=1.0
        )
        s.get_shots(0.1)   # high
        s.get_shots(0.5)   # default
        s.get_shots(2.0)   # low
        stats = s.get_stats()
        assert stats["n_decisions"] == 3
        assert stats["total"] == 1024 + 256 + 64
        assert stats["mean"] == pytest.approx((1024 + 256 + 64) / 3, abs=1)

    def test_reset_clears_history(self):
        """reset() should clear shot history."""
        s = AdaptiveShotScheduler()
        s.get_shots(0.1)
        s.get_shots(0.5)
        s.reset()
        assert len(s._history) == 0
        assert s.get_stats()["n_decisions"] == 0

    def test_empty_stats(self):
        """Empty scheduler should return zero stats."""
        s = AdaptiveShotScheduler()
        stats = s.get_stats()
        assert stats["n_decisions"] == 0
        assert stats["total"] == 0
