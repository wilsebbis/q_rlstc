"""DROP action extension for the Q-RLSTC MDP.

Expands the action space from EXTEND/CUT to EXTEND/CUT/DROP when
enabled. DROP discards the current point entirely — it is not included
in any cluster segment.

Cost model (per user feedback):
    - DROP incurs a penalty reward: r_drop = -drop_penalty
    - Consecutive DROPs are guarded: max_consecutive_drops limits
      degenerate "drop everything" policies
    - A retention constraint ensures ≥ retain_pct% of trajectory
      points are kept (not dropped)

Tracking:
    - drop_pct is logged alongside cut_pct as a third evaluation axis
    - budget_violated (hard constraint) can check both CUT% and DROP%

Usage:
    action_space = DropActionSpace(
        enabled=True,
        drop_penalty=0.5,
        max_consecutive_drops=3,
        retain_pct=70.0,
    )
    # During step:
    if action == action_space.DROP:
        if action_space.is_drop_allowed(consecutive_drops):
            # Apply drop semantics
        else:
            # Force EXTEND instead
"""

from dataclasses import dataclass

# Action indices
EXTEND = 0
CUT = 1
DROP = 2


@dataclass
class DropActionSpace:
    """Configuration for the DROP action extension.

    Attributes:
        enabled: If False, action space remains binary (EXTEND/CUT).
        n_actions: 3 when enabled, 2 when disabled.
        drop_penalty: Reward penalty for each DROP action.
        max_consecutive_drops: Max consecutive DROPs before forced EXTEND.
        retain_pct: Minimum percentage of trajectory points that must
            be retained (not dropped). Evaluated post-episode.
        drop_penalty_escalation: If > 0, penalty increases by this
            amount for each consecutive drop within a sequence.
    """
    enabled: bool = False
    drop_penalty: float = 0.5
    max_consecutive_drops: int = 3
    retain_pct: float = 70.0
    drop_penalty_escalation: float = 0.1

    @property
    def n_actions(self) -> int:
        """Number of available actions."""
        return 3 if self.enabled else 2

    def is_drop_allowed(self, consecutive_drops: int) -> bool:
        """Check if DROP is currently allowed.

        Args:
            consecutive_drops: Number of consecutive DROPs so far.

        Returns:
            True if another DROP is allowed.
        """
        if not self.enabled:
            return False
        return consecutive_drops < self.max_consecutive_drops

    def get_drop_penalty(self, consecutive_drops: int) -> float:
        """Get the penalty for dropping at this point.

        Penalty increases with consecutive drops to discourage runs.

        Args:
            consecutive_drops: Number of consecutive DROPs before this one.

        Returns:
            Penalty value (positive, to be subtracted from reward).
        """
        return self.drop_penalty + self.drop_penalty_escalation * consecutive_drops

    def check_retention(self, n_total: int, n_dropped: int) -> bool:
        """Check if retention constraint is satisfied.

        Args:
            n_total: Total number of trajectory points.
            n_dropped: Number of points dropped.

        Returns:
            True if retention constraint is met.
        """
        if n_total == 0:
            return True
        retain_actual = 100.0 * (n_total - n_dropped) / n_total
        return retain_actual >= self.retain_pct

    def compute_drop_pct(self, n_total: int, n_dropped: int) -> float:
        """Compute DROP percentage.

        Args:
            n_total: Total trajectory points.
            n_dropped: Points dropped.

        Returns:
            DROP percentage (0-100).
        """
        if n_total == 0:
            return 0.0
        return 100.0 * n_dropped / n_total
