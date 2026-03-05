"""Tests for soft-DQN target computation (soft_targets.py).

Verifies numerical stability, limiting behavior (α→0 = hard max),
batch/scalar handling, and Boltzmann policy normalization.
"""

import numpy as np
import pytest

from q_rlstc.rl.soft_targets import soft_value, soft_policy


class TestSoftValue:
    """Tests for soft_value function."""

    def test_recovers_hard_max_small_alpha(self):
        """As α → 0, soft_value → max(Q)."""
        q = np.array([1.0, 3.0, 2.0])
        sv = soft_value(q, alpha=1e-6)
        assert sv == pytest.approx(3.0, abs=1e-4)

    def test_above_hard_max_for_positive_alpha(self):
        """Soft value ≥ max(Q) for α > 0 (entropy bonus)."""
        q = np.array([1.0, 2.0])
        sv = soft_value(q, alpha=0.5)
        assert sv >= max(q)

    def test_batch_shape(self):
        """Batch input should return batch-shaped output."""
        q = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, 0.5]])
        sv = soft_value(q, alpha=0.1)
        assert sv.shape == (3,)
        # Each soft value ≥ max of row
        for i in range(3):
            assert sv[i] >= q[i].max()

    def test_equal_q_values(self):
        """Equal Q-values → V = Q + α·log(n_actions)."""
        q = np.array([2.0, 2.0])
        alpha = 0.5
        sv = soft_value(q, alpha=alpha)
        expected = 2.0 + alpha * np.log(2)
        assert sv == pytest.approx(expected, abs=1e-10)

    def test_numerical_stability_large_q(self):
        """Should handle very large Q-values without overflow."""
        q = np.array([1000.0, 999.0])
        sv = soft_value(q, alpha=0.1)
        assert np.isfinite(sv)
        assert sv >= 1000.0

    def test_negative_q_values(self):
        """Should work with negative Q-values."""
        q = np.array([-5.0, -3.0])
        sv = soft_value(q, alpha=0.1)
        assert np.isfinite(sv)
        assert sv >= -3.0  # ≥ max(Q)

    def test_alpha_zero_raises(self):
        """α = 0 should raise ValueError."""
        with pytest.raises(ValueError):
            soft_value(np.array([1.0, 2.0]), alpha=0.0)

    def test_negative_alpha_raises(self):
        """Negative α should raise ValueError."""
        with pytest.raises(ValueError):
            soft_value(np.array([1.0, 2.0]), alpha=-0.1)


class TestSoftPolicy:
    """Tests for soft_policy / Boltzmann function."""

    def test_sums_to_one(self):
        """Probabilities must sum to 1."""
        q = np.array([1.0, 3.0, 2.0])
        pi = soft_policy(q, alpha=0.5)
        assert pi.sum() == pytest.approx(1.0, abs=1e-10)

    def test_max_q_gets_highest_prob(self):
        """Action with highest Q should get highest probability."""
        q = np.array([1.0, 5.0, 2.0])
        pi = soft_policy(q, alpha=0.1)
        assert np.argmax(pi) == 1

    def test_uniform_for_equal_q(self):
        """Equal Q-values → uniform policy."""
        q = np.array([3.0, 3.0])
        pi = soft_policy(q, alpha=0.5)
        assert pi[0] == pytest.approx(0.5, abs=1e-10)
        assert pi[1] == pytest.approx(0.5, abs=1e-10)

    def test_batch_normalization(self):
        """Each row in batch should sum to 1."""
        q = np.array([[1.0, 2.0], [5.0, 1.0]])
        pi = soft_policy(q, alpha=0.1)
        assert pi.shape == (2, 2)
        for i in range(2):
            assert pi[i].sum() == pytest.approx(1.0, abs=1e-10)

    def test_low_alpha_concentrates(self):
        """Very low α → near-deterministic (concentrated on max)."""
        q = np.array([1.0, 5.0])
        pi = soft_policy(q, alpha=0.001)
        assert pi[1] > 0.999

    def test_alpha_zero_raises(self):
        """α = 0 should raise ValueError."""
        with pytest.raises(ValueError):
            soft_policy(np.array([1.0, 2.0]), alpha=0.0)
