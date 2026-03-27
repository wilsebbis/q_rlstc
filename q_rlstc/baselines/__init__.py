"""Non-learned baseline policies for trajectory clustering evaluation.

These baselines provide standard reference points for the learned RL policies,
proving that the neural/quantum agents actually exploit spatial geometry rather
than just exploiting the ValCR metric length bias via random cutting.
"""

from .random_policy import run_random_policy, BaselineResult
from .fixed_window import run_fixed_window
from .heading_change import run_heading_change

__all__ = [
    "BaselineResult",
    "run_random_policy",
    "run_fixed_window",
    "run_heading_change"
]
