"""Tests for DROP action extension.

Verifies action space sizing, drop guards, retention checks,
escalating penalty, and DROP% computation.
"""

import pytest

from q_rlstc.rl.drop_action import DropActionSpace, EXTEND, CUT, DROP


class TestDropActionSpace:
    """Tests for DropActionSpace."""

    def test_disabled_has_two_actions(self):
        """Disabled → binary action space."""
        das = DropActionSpace(enabled=False)
        assert das.n_actions == 2

    def test_enabled_has_three_actions(self):
        """Enabled → ternary action space."""
        das = DropActionSpace(enabled=True)
        assert das.n_actions == 3

    def test_action_indices(self):
        """Action constants match expected values."""
        assert EXTEND == 0
        assert CUT == 1
        assert DROP == 2

    def test_drop_not_allowed_when_disabled(self):
        """DROP never allowed when disabled."""
        das = DropActionSpace(enabled=False)
        assert not das.is_drop_allowed(consecutive_drops=0)

    def test_drop_allowed_within_limit(self):
        """DROP allowed when consecutive count < max."""
        das = DropActionSpace(enabled=True, max_consecutive_drops=3)
        assert das.is_drop_allowed(0)
        assert das.is_drop_allowed(1)
        assert das.is_drop_allowed(2)

    def test_drop_blocked_at_limit(self):
        """DROP blocked when consecutive count >= max."""
        das = DropActionSpace(enabled=True, max_consecutive_drops=3)
        assert not das.is_drop_allowed(3)
        assert not das.is_drop_allowed(5)

    def test_base_penalty(self):
        """First drop should use base penalty."""
        das = DropActionSpace(drop_penalty=0.5, drop_penalty_escalation=0.1)
        assert das.get_drop_penalty(consecutive_drops=0) == pytest.approx(0.5)

    def test_escalating_penalty(self):
        """Consecutive drops should increase penalty."""
        das = DropActionSpace(drop_penalty=0.5, drop_penalty_escalation=0.1)
        assert das.get_drop_penalty(0) == pytest.approx(0.5)
        assert das.get_drop_penalty(1) == pytest.approx(0.6)
        assert das.get_drop_penalty(2) == pytest.approx(0.7)

    def test_retention_satisfied(self):
        """Retention met when enough points kept."""
        das = DropActionSpace(retain_pct=70.0)
        assert das.check_retention(n_total=100, n_dropped=20)  # 80% kept
        assert das.check_retention(n_total=100, n_dropped=30)  # 70% kept

    def test_retention_violated(self):
        """Retention violated when too many points dropped."""
        das = DropActionSpace(retain_pct=70.0)
        assert not das.check_retention(n_total=100, n_dropped=31)  # 69% kept

    def test_retention_empty_trajectory(self):
        """Empty trajectory should satisfy retention."""
        das = DropActionSpace(retain_pct=70.0)
        assert das.check_retention(n_total=0, n_dropped=0)

    def test_drop_pct_computation(self):
        """DROP% should be correctly computed."""
        das = DropActionSpace()
        assert das.compute_drop_pct(100, 15) == pytest.approx(15.0)
        assert das.compute_drop_pct(100, 0) == pytest.approx(0.0)
        assert das.compute_drop_pct(0, 0) == pytest.approx(0.0)
