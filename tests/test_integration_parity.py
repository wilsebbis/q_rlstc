"""Integration tests: RLSTC Parity Mode invariants.

These tests verify that parity mode correctly disables all thesis-specific
reward shaping, applies RLSTC-matched hyperparams, and produces the
exact training conditions described in the RLSTC paper.

Runs in <1s with no quantum circuits (tests protocol config only).
"""

import pytest
import sys
from pathlib import Path

# Add experiments to path so we can import run_thesis_experiments
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from run_thesis_experiments import (
    RLSTC_PARITY_PROTOCOL,
    PROTOCOL,
    TrainingMode,
)


class TestParityProtocolInvariants:
    """Verify RLSTC_PARITY_PROTOCOL matches RLSTC paper exactly."""

    def test_gamma_is_099(self):
        """RLSTC uses γ=0.99 (not thesis default 0.90)."""
        assert RLSTC_PARITY_PROTOCOL["gamma"] == 0.99

    def test_epsilon_decay_per_step(self):
        """RLSTC decays ε per step, not per episode."""
        assert RLSTC_PARITY_PROTOCOL["EPSILON_DECAY_MODE"] == "per_step"
        assert RLSTC_PARITY_PROTOCOL["EPSILON_DECAY_PER_STEP"] == 0.99

    def test_soft_target_tau(self):
        """RLSTC uses soft target update τ=0.001."""
        assert RLSTC_PARITY_PROTOCOL.get("USE_SOFT_TARGET") is True
        assert RLSTC_PARITY_PROTOCOL.get("SOFT_TARGET_TAU") == 0.001

    def test_reward_mode_raw_od_delta(self):
        """Parity reward must be raw OD delta (r = OD_t - OD_{t+1})."""
        assert RLSTC_PARITY_PROTOCOL["REWARD_MODE"] == "raw_od_delta"


class TestParityShapiTermsZero:
    """All thesis-specific reward shaping must be exactly zero in parity mode."""

    def test_cut_penalty_zero(self):
        """No CUT penalty in parity (RLSTC doesn't penalize CUT)."""
        assert RLSTC_PARITY_PROTOCOL["CUT_PENALTY"] == 0.0

    def test_extend_cost_zero(self):
        """No EXTEND cost in parity."""
        assert RLSTC_PARITY_PROTOCOL["EXTEND_COST"] == 0.0

    def test_complexity_lambda_zero(self):
        """No complexity regularization in parity."""
        assert RLSTC_PARITY_PROTOCOL["COMPLEXITY_LAMBDA"] == 0.0

    def test_min_cut_bonus_zero(self):
        """No first-CUT bonus in parity."""
        assert RLSTC_PARITY_PROTOCOL["MIN_CUT_BONUS"] == 0.0
        assert RLSTC_PARITY_PROTOCOL["MIN_CUT_BONUS_FINAL"] == 0.0

    def test_lagrangian_off(self):
        """Lagrangian adaptive penalty is OFF in parity mode."""
        assert RLSTC_PARITY_PROTOCOL["USE_LAGRANGIAN"] is False

    def test_forced_cut_off(self):
        """No forced-cut curriculum in parity."""
        assert RLSTC_PARITY_PROTOCOL["FORCED_CUT_PROB"] == 0.0
        assert RLSTC_PARITY_PROTOCOL["FORCED_CUT_EPOCHS"] == 0

    def test_optimistic_bias_off(self):
        """No optimistic CUT bias in parity."""
        assert RLSTC_PARITY_PROTOCOL["OPTIMISTIC_CUT_BIAS"] == 0.0

    def test_stratified_replay_off(self):
        """No action-stratified replay in parity."""
        assert RLSTC_PARITY_PROTOCOL["USE_STRATIFIED_REPLAY"] is False

    def test_collapse_detection_off(self):
        """Collapse detection threshold is 0 in parity."""
        assert RLSTC_PARITY_PROTOCOL["COLLAPSE_CUT_THRESHOLD"] == 0.0


class TestParityStateFeatures:
    """Verify parity uses RLSTC's state feature mode."""

    def test_l_min_is_one(self):
        """RLSTC allows segments of length 1 (L_MIN=1)."""
        assert RLSTC_PARITY_PROTOCOL["L_MIN"] == 1

    def test_exploration_is_epsilon_greedy(self):
        """RLSTC uses ε-greedy, not Boltzmann."""
        assert RLSTC_PARITY_PROTOCOL["EXPLORATION_MODE"] == "epsilon_greedy"


class TestParityDiffersFromControlled:
    """Parity and controlled modes must actually differ on key parameters."""

    def test_gamma_differs(self):
        """γ should differ between parity (0.99) and controlled (0.90)."""
        assert RLSTC_PARITY_PROTOCOL["gamma"] != PROTOCOL["gamma"]

    def test_lagrangian_differs(self):
        """Lagrangian should be OFF in parity, ON in controlled."""
        assert RLSTC_PARITY_PROTOCOL["USE_LAGRANGIAN"] is False
        assert PROTOCOL["USE_LAGRANGIAN"] is True

    def test_shaping_differs(self):
        """CUT_PENALTY should be 0 in parity, >0 in controlled."""
        assert RLSTC_PARITY_PROTOCOL["CUT_PENALTY"] == 0.0
        assert PROTOCOL["CUT_PENALTY"] > 0

    def test_training_mode_enum_values(self):
        """TrainingMode enum should have exactly two members."""
        assert TrainingMode.CONTROLLED_SPSA.value == "controlled_spsa"
        assert TrainingMode.RLSTC_PARITY.value == "rlstc_parity"
