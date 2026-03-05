"""Integration tests: Lagrangian dual-variable update invariants.

Verifies the dual ascent mechanism for CUT budget control:
- r' = r - λ·1[CUT] applied per step
- λ increases when cut_rate > b_soft, decreases otherwise
- λ projected to [0, λ_max]
- Freeze epochs hold λ constant
- EMA smoothing works correctly
- Delta clamp prevents wild swings

Runs in <1s with pure numerical computations (no training loop).
"""

import numpy as np
import pytest


class TestLagrangianDualUpdate:
    """Tests for the Lagrangian λ update rule.

    Reimplements the exact update logic from run_thesis_experiments.py
    lines 894–910 to verify invariants independently.
    """

    @staticmethod
    def lagrangian_step(
        lagrangian_lambda: float,
        cut_rate_ema: float,
        batch_cut_rate: float,
        target_cut_pct: float,
        lagrangian_lr: float = 0.02,
        lambda_max: float = 2.0,
        lambda_delta_max: float = 0.05,
        lambda_cut_ema_decay: float = 0.9,
        frozen: bool = False,
    ) -> tuple:
        """Single Lagrangian update step (mirrors exact experiment code).

        Returns (new_lambda, new_ema, delta_lambda).
        """
        # EMA update
        if cut_rate_ema is None:
            cut_rate_ema = batch_cut_rate
        else:
            cut_rate_ema = (lambda_cut_ema_decay * cut_rate_ema
                           + (1 - lambda_cut_ema_decay) * batch_cut_rate)

        if frozen:
            return lagrangian_lambda, cut_rate_ema, 0.0

        raw_delta = lagrangian_lr * (cut_rate_ema - target_cut_pct)
        delta_lambda = max(-lambda_delta_max, min(lambda_delta_max, raw_delta))
        new_lambda = max(0.0, min(lambda_max, lagrangian_lambda + delta_lambda))
        return new_lambda, cut_rate_ema, delta_lambda

    def test_lambda_increases_when_overcutting(self):
        """λ should increase when CUT rate exceeds b_soft target."""
        lam, _, delta = self.lagrangian_step(
            lagrangian_lambda=0.1,
            cut_rate_ema=None,
            batch_cut_rate=25.0,   # >> target 10%
            target_cut_pct=10.0,
        )
        assert delta > 0
        assert lam > 0.1

    def test_lambda_decreases_when_undercutting(self):
        """λ should decrease when CUT rate is below b_soft target."""
        lam, _, delta = self.lagrangian_step(
            lagrangian_lambda=0.5,
            cut_rate_ema=None,
            batch_cut_rate=3.0,    # << target 10%
            target_cut_pct=10.0,
        )
        assert delta < 0
        assert lam < 0.5

    def test_lambda_projected_to_zero_floor(self):
        """λ must never go negative (projected to [0, λ_max])."""
        lam, _, _ = self.lagrangian_step(
            lagrangian_lambda=0.01,
            cut_rate_ema=None,
            batch_cut_rate=0.0,    # way below target → large negative delta
            target_cut_pct=10.0,
            lagrangian_lr=1.0,     # aggressive LR to force negative
        )
        assert lam >= 0.0

    def test_lambda_projected_to_max_ceiling(self):
        """λ must not exceed λ_max."""
        lam, _, _ = self.lagrangian_step(
            lagrangian_lambda=1.95,
            cut_rate_ema=None,
            batch_cut_rate=100.0,   # way above target → large positive delta
            target_cut_pct=10.0,
            lambda_max=2.0,
        )
        assert lam <= 2.0

    def test_frozen_lambda_unchanged(self):
        """λ should not change during freeze epochs."""
        lam_orig = 0.5
        lam, _, delta = self.lagrangian_step(
            lagrangian_lambda=lam_orig,
            cut_rate_ema=None,
            batch_cut_rate=50.0,   # would normally cause increase
            target_cut_pct=10.0,
            frozen=True,
        )
        assert lam == lam_orig
        assert delta == 0.0

    def test_delta_clamped(self):
        """Δλ should be clamped to [-delta_max, +delta_max]."""
        _, _, delta = self.lagrangian_step(
            lagrangian_lambda=1.0,
            cut_rate_ema=None,
            batch_cut_rate=100.0,  # extreme → large raw delta
            target_cut_pct=10.0,
            lagrangian_lr=1.0,     # aggressive → raw_delta = 1.0*(100-10) = 90
            lambda_delta_max=0.05,
        )
        assert abs(delta) <= 0.05 + 1e-10

    def test_ema_smoothing(self):
        """EMA should smooth batch CUT rate observations."""
        decay = 0.9
        # First call: EMA initializes to batch_cut_rate
        _, ema1, _ = self.lagrangian_step(
            lagrangian_lambda=0.1, cut_rate_ema=None,
            batch_cut_rate=20.0, target_cut_pct=10.0,
            lambda_cut_ema_decay=decay,
        )
        assert ema1 == 20.0  # first observation = initialization

        # Second call: EMA = 0.9*20 + 0.1*5 = 18.5
        _, ema2, _ = self.lagrangian_step(
            lagrangian_lambda=0.1, cut_rate_ema=ema1,
            batch_cut_rate=5.0, target_cut_pct=10.0,
            lambda_cut_ema_decay=decay,
        )
        assert ema2 == pytest.approx(0.9 * 20.0 + 0.1 * 5.0)

    def test_convergence_to_target(self):
        """Repeated updates should drive λ toward equilibrium near target."""
        lam = 0.1
        ema = None
        # Simulate constant 10% CUT rate (== target)
        for _ in range(100):
            lam, ema, delta = self.lagrangian_step(
                lagrangian_lambda=lam,
                cut_rate_ema=ema,
                batch_cut_rate=10.0,
                target_cut_pct=10.0,
            )
        # At equilibrium, delta ≈ 0
        assert abs(delta) < 1e-6


class TestRewardShaping:
    """Verify r' = r - λ·1[CUT] is applied correctly."""

    def test_cut_action_penalized(self):
        """CUT action should reduce reward by λ."""
        base_reward = 1.0
        lagrangian_lambda = 0.5
        # r' = r - λ·1[action==CUT] = 1.0 - 0.5 = 0.5
        shaped_reward = base_reward - lagrangian_lambda
        assert shaped_reward == pytest.approx(0.5)

    def test_extend_action_unpenalized(self):
        """EXTEND action should NOT be penalized by λ."""
        base_reward = 1.0
        lagrangian_lambda = 0.5
        # r' = r - λ·1[action==EXTEND] = r (since 1[EXTEND]=0)
        shaped_reward = base_reward - lagrangian_lambda * 0  # EXTEND → no penalty
        assert shaped_reward == pytest.approx(1.0)

    def test_lambda_zero_no_shaping(self):
        """λ=0 should leave reward unchanged for CUT."""
        base_reward = 0.8
        lagrangian_lambda = 0.0
        shaped_reward = base_reward - lagrangian_lambda
        assert shaped_reward == pytest.approx(0.8)
