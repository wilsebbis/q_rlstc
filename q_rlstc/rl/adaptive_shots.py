"""Adaptive shot allocation for VQ-DQN agents.

Drop-in wrapper that dynamically adjusts measurement shots based on
Q-value margins:

    |Q(s,extend) - Q(s,cut)| < τ₁  →  shots_high   (need precision)
    |Q(s,extend) - Q(s,cut)| > τ₂  →  shots_low    (confident, save shots)
    else                            →  shots_default

This trades faster evaluation on confident states for higher precision
on margin-critical states, reducing total measurement budget while
maintaining decision quality.

Usage:
    scheduler = AdaptiveShotScheduler(
        shots_low=64, shots_default=256, shots_high=1024,
        tau_low=1.0, tau_high=0.2
    )
    shots = scheduler.get_shots(q_margin=0.3)
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class AdaptiveShotScheduler:
    """Q-margin-based shot allocation scheduler.

    Attributes:
        shots_low: Shot count for confident decisions (wide margin).
        shots_default: Shot count for moderate decisions.
        shots_high: Shot count for uncertain decisions (narrow margin).
        tau_low: Margin threshold below which shots_high is used.
        tau_high: Margin threshold above which shots_low is used.
    """
    shots_low: int = 64
    shots_default: int = 256
    shots_high: int = 1024
    tau_low: float = 0.2     # |ΔQ| < τ_low → high shots
    tau_high: float = 1.0    # |ΔQ| > τ_high → low shots

    # Tracking (for logging / histogram)
    _history: List[int] = field(default_factory=list, repr=False)

    def get_shots(self, q_margin: float) -> int:
        """Determine shot count from Q-value margin.

        Args:
            q_margin: |Q(s, extend) - Q(s, cut)|. Non-negative.

        Returns:
            Number of measurement shots to use.
        """
        margin = abs(q_margin)

        if margin < self.tau_low:
            shots = self.shots_high
        elif margin > self.tau_high:
            shots = self.shots_low
        else:
            shots = self.shots_default

        self._history.append(shots)
        return shots

    def get_stats(self) -> dict:
        """Return shot allocation statistics.

        Returns:
            Dict with mean, std, histogram bins, and total shot budget.
        """
        if not self._history:
            return {"mean": 0, "std": 0, "total": 0, "n_decisions": 0}

        h = np.array(self._history)
        return {
            "mean": float(np.mean(h)),
            "std": float(np.std(h)),
            "total": int(np.sum(h)),
            "n_decisions": len(h),
            "pct_low": float(100 * np.mean(h == self.shots_low)),
            "pct_default": float(100 * np.mean(h == self.shots_default)),
            "pct_high": float(100 * np.mean(h == self.shots_high)),
        }

    def reset(self) -> None:
        """Clear shot history for a new episode/epoch."""
        self._history = []
