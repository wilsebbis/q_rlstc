"""Seed-stability reporting for reinforcement learning experiments.

Computes metrics over multiple random seeds to identify narrow-regime
operation, collapse behaviors (0% or 100% CUT), and robust performance.
"""

import numpy as np
from typing import List, Tuple, Dict, Any

def collapse_rate(
    runs_cut_rates: List[float],
    threshold_low: float = 0.01,
    threshold_high: float = 0.99
) -> Dict[str, float]:
    """Measure how often a model collapses into extreme degenerate behavior.

    Args:
        runs_cut_rates: List of final CUT% selected by the policy across seeds.
        threshold_low: CUT% below which the model is considered collapsed (does nothing).
        threshold_high: CUT% above which the model is considered collapsed (shatters everything).

    Returns:
        Dict with collapse rates and total collapse percentage.
    """
    total_runs = len(runs_cut_rates)
    if total_runs == 0:
        return {"low_collapse": 0.0, "high_collapse": 0.0, "total_collapse": 0.0}
        
    low_collapses = sum(1 for c in runs_cut_rates if c <= threshold_low)
    high_collapses = sum(1 for c in runs_cut_rates if c >= threshold_high)
    
    return {
        "low_collapse": low_collapses / total_runs,
        "high_collapse": high_collapses / total_runs,
        "total_collapse": (low_collapses + high_collapses) / total_runs
    }

def best_worst_spread(runs_metrics: List[float]) -> float:
    """Calculate the spread between the best and worst seed performance.

    Args:
        runs_metrics: List of metrics across seeds.

    Returns:
        Difference between maximum and minimum metric (lower spread usually means more stable).
    """
    if not runs_metrics:
        return 0.0
    return max(runs_metrics) - min(runs_metrics)

def iqr(runs_metrics: List[float]) -> float:
    """Calculate the Interquartile Range (IQR) of a metric across seeds.

    Robust measure of variance that ignores massive outliers caused by seed collapse.

    Args:
        runs_metrics: List of metrics across seeds.

    Returns:
        IQR (75th percentile - 25th percentile).
    """
    if len(runs_metrics) < 2:
        return 0.0
    q75, q25 = np.percentile(runs_metrics, [75, 25])
    return float(q75 - q25)

def probability_beat_baseline(
    agent_metrics: List[float],
    baseline_metrics: List[float],
    lower_is_better: bool = True
) -> float:
    """Calculate the empirical probability that an agent seed beats a random baseline seed.

    Assumes independent seeds. Measures P(Agent > Baseline) across all pairwise seed combinations.

    Args:
        agent_metrics: List of agent metrics.
        baseline_metrics: List of baseline metrics.
        lower_is_better: True if a lower metric implies "beating".

    Returns:
        Probability in [0, 1].
    """
    if not agent_metrics or not baseline_metrics:
        return 0.0
        
    wins = 0
    total = len(agent_metrics) * len(baseline_metrics)
    
    for am in agent_metrics:
        for bm in baseline_metrics:
            if lower_is_better:
                if am < bm:
                    wins += 1
                elif am == bm:
                    wins += 0.5  # Tie
            else:
                if am > bm:
                    wins += 1
                elif am == bm:
                    wins += 0.5  # Tie
                    
    return wins / total
