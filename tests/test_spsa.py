"""Tests for the SPSA optimizer.

Verifies learning-rate schedule, perturbation schedule, gradient
estimation on a known quadratic, gradient clipping, and parameter
update direction.
"""

import numpy as np
import pytest

from q_rlstc.rl.spsa import SPSAOptimizer


class TestSPSASchedules:
    """Learning rate and perturbation schedules."""

    def test_learning_rate_decreases(self):
        """a_k should decrease over iterations."""
        opt = SPSAOptimizer(a=0.12, c=0.10, A=20)
        rates = [opt._get_learning_rate(k) for k in range(1, 50)]
        # Should be monotonically non-increasing
        for i in range(len(rates) - 1):
            assert rates[i] >= rates[i + 1], \
                f"a_k not decreasing: a_{i+1}={rates[i]} > a_{i+2}={rates[i+1]}"

    def test_perturbation_decreases(self):
        """c_k should decrease over iterations."""
        opt = SPSAOptimizer(a=0.12, c=0.10, A=20)
        perts = [opt._get_perturbation_magnitude(k) for k in range(1, 50)]
        for i in range(len(perts) - 1):
            assert perts[i] >= perts[i + 1]

    def test_initial_learning_rate(self):
        """First learning rate should be close to a / (1 + A)^alpha."""
        opt = SPSAOptimizer(a=0.12, c=0.10, A=20)
        a1 = opt._get_learning_rate(1)
        expected = 0.12 / (20 + 1 + 1) ** 0.602
        assert abs(a1 - expected) < 1e-6


class TestSPSAGradientEstimation:
    """Gradient estimation on known functions."""

    def test_averaged_gradient_direction_quadratic(self):
        """Averaged SPSA gradient should align with true gradient of x^2.

        SPSA is stochastic — individual samples can have wrong per-component
        signs. We average 50 independent estimates and check that the overall
        direction (dot product with true gradient 2x) is positive.
        """
        params = np.array([1.0, -2.0, 0.5, -0.3])
        true_gradient = 2.0 * params  # analytical gradient of sum(x^2)

        def loss_fn(p):
            return float(np.sum(p ** 2))

        accumulated = np.zeros_like(params)
        for i in range(50):
            opt = SPSAOptimizer(a=0.5, c=0.1, A=0, seed=i)
            grad = opt.compute_gradient(loss_fn, params)
            accumulated += grad

        avg_grad = accumulated / 50
        dot = np.dot(avg_grad, true_gradient)
        assert dot > 0, (
            f"Averaged gradient not aligned with true gradient: "
            f"dot={dot:.4f}, avg_grad={avg_grad}, true={true_gradient}"
        )

    def test_multiple_steps_reduce_loss(self):
        """Multiple SPSA steps should reduce loss on a simple quadratic.

        A single stochastic step can overshoot, so we verify the trend
        over 20 small steps with a conservative learning rate.
        """
        opt = SPSAOptimizer(a=0.05, c=0.1, A=10, seed=42)
        params = np.array([2.0, -3.0, 1.5, -1.0])

        def loss_fn(p):
            return float(np.sum(p ** 2))

        initial_loss = loss_fn(params)

        for _ in range(20):
            params, _ = opt.step(loss_fn, params)

        final_loss = loss_fn(params)
        assert final_loss < initial_loss, \
            f"Loss did not decrease after 20 steps: {initial_loss:.4f} -> {final_loss:.4f}"


class TestSPSAGradientClipping:
    """Gradient clipping behaviour (max_grad_norm)."""

    def test_clipping_limits_gradient_norm(self):
        """Gradient clipping should limit the gradient magnitude."""
        opt = SPSAOptimizer(a=0.12, c=0.10, A=20, max_grad_norm=1.0, seed=42)
        params = np.array([100.0, -100.0])

        def loss_fn(p):
            return float(np.sum(p ** 2))

        # step() applies gradient clipping internally
        _, grad_norm = opt.step(loss_fn, params)
        assert grad_norm <= 1.0 + 1e-6

    def test_large_max_grad_norm_no_clipping(self):
        """With large max_grad_norm, gradient should pass through."""
        opt = SPSAOptimizer(a=0.12, c=0.10, A=20, max_grad_norm=1e10, seed=42)
        params = np.array([0.1, -0.1])

        def loss_fn(p):
            return float(np.sum(p ** 2))

        _, grad_norm = opt.step(loss_fn, params)
        # Should not have been clipped (gradient of small values is small)
        assert grad_norm < 100
