"""Budget-constrained evaluation metrics for trajectory clustering.

Provides methods for computing metrics that respect maximum CUT budgets,
extracting Pareto frontiers, and measuring rank instability when moving
from raw metrics to budgeted metrics.
"""

import numpy as np
from typing import List, Dict, Tuple, Set

def compute_valcr(cluster_dict: Dict[int, list], base_sim: float) -> float:
    from q_rlstc.data.rlstc_cluster import compute_overdist
    od = compute_overdist(cluster_dict)
    return od / base_sim if base_sim > 0 else float('inf')

def compute_nvalcr(cluster_dict: Dict[int, list], base_sim: float) -> float:
    from q_rlstc.data.rlstc_cluster import compute_overdist_per_point
    n_od = compute_overdist_per_point(cluster_dict)
    return n_od / base_sim if base_sim > 0 else float('inf')

def compute_wvalcr(cluster_dict: Dict[int, list], base_sim: float) -> float:
    from q_rlstc.data.rlstc_cluster import compute_overdist_length_weighted
    w_od = compute_overdist_length_weighted(cluster_dict)
    return w_od / base_sim if base_sim > 0 else float('inf')


def best_metric_under_cut_budget(
    cut_rates: List[float],
    metrics: List[float],
    budget: float,
    lower_is_better: bool = True
) -> float:
    """Find the best metric achieved while respecting the maximum CUT% budget.

    Args:
        cut_rates: List of CUT% for each configuration/epoch.
        metrics: List of corresponding metrics (e.g., ValCR).
        budget: Maximum allowable CUT%.
        lower_is_better: True if lower metric is better (e.g., ValCR).

    Returns:
        Best metric under budget, or inf / -inf if no runs met the budget.
    """
    valid_metrics = [m for c, m in zip(cut_rates, metrics) if c <= budget]
    if not valid_metrics:
        return float('inf') if lower_is_better else float('-inf')
    return min(valid_metrics) if lower_is_better else max(valid_metrics)

def pareto_frontier(
    cut_rates: List[float],
    metrics: List[float],
    lower_is_better: bool = True
) -> Tuple[List[float], List[float]]:
    """Extract the Pareto frontier from a set of CUT% and metrics.

    Args:
        cut_rates: List of CUT%.
        metrics: List of metrics.
        lower_is_better: True if lower metric is better.

    Returns:
        Tuple of (frontier_cuts, frontier_metrics) sorted by CUT%.
    """
    points = sorted(zip(cut_rates, metrics), key=lambda x: x[0])
    frontier_cuts = []
    frontier_metrics = []
    
    best_metric = float('inf') if lower_is_better else float('-inf')
    
    for c, m in points:
        if lower_is_better:
            if m < best_metric:
                frontier_cuts.append(c)
                frontier_metrics.append(m)
                best_metric = m
        else:
            if m > best_metric:
                frontier_cuts.append(c)
                frontier_metrics.append(m)
                best_metric = m
                
    return frontier_cuts, frontier_metrics

def pareto_auc(
    frontier_cuts: List[float],
    frontier_metrics: List[float],
    max_cut: float = 1.0
) -> float:
    """Compute the Area Under the Curve (AUC) for the Pareto frontier up to max_cut.
    
    Assumes step function behavior: a metric is held until a better one is found at a higher CUT%.

    Args:
        frontier_cuts: Sorted list of CUT% on the frontier.
        frontier_metrics: Corresponding metrics on the frontier.
        max_cut: Maximum CUT% to consider for the area.

    Returns:
        Area under the curve.
    """
    if not frontier_cuts:
        return 0.0

    auc = 0.0
    for i in range(len(frontier_cuts) - 1):
        width = frontier_cuts[i+1] - frontier_cuts[i]
        height = frontier_metrics[i]
        auc += width * height
        
    if frontier_cuts[-1] < max_cut:
        width = max_cut - frontier_cuts[-1]
        height = frontier_metrics[-1]
        auc += width * height

    return auc

def delta_vs_random(
    matched_budget_score: float,
    random_budget_score: float,
    lower_is_better: bool = True
) -> float:
    """Compute fractional improvement over the random baseline at a matched budget.

    Negative output means improvement if lower_is_better (e.g., -0.10 is a 10% reduction in ValCR).
    """
    if random_budget_score == 0 or random_budget_score == float('inf'):
        return 0.0
    return (matched_budget_score - random_budget_score) / random_budget_score


def rank_instability(
    raw_scores: Dict[str, float],
    budgeted_scores: Dict[str, float],
    lower_is_better: bool = True
) -> Tuple[float, int]:
    """Calculate the Spearman correlation and absolute rank reversals.

    Args:
        raw_scores: Dict mapping model IDs to their unconstrained best metrics.
        budgeted_scores: Dict mapping model IDs to their budgeted metrics.
        lower_is_better: True if lower score is better.

    Returns:
        Tuple of (spearman_correlation, number_of_pairwise_reversals).
    """
    from scipy.stats import spearmanr
    from itertools import combinations
    
    models = list(raw_scores.keys())
    if not models or len(models) < 2:
        return 1.0, 0
    
    # Sort models. Best models come first.
    raw_sorted = sorted(models, key=lambda x: raw_scores[x], reverse=not lower_is_better)
    budgeted_sorted = sorted(models, key=lambda x: budgeted_scores.get(x, float('inf') if lower_is_better else float('-inf')), reverse=not lower_is_better)
    
    raw_ranks = [raw_sorted.index(m) for m in models]
    bud_ranks = [budgeted_sorted.index(m) for m in models]
    
    correlation, _ = spearmanr(raw_ranks, bud_ranks)
    
    reversals = 0
    for m1, m2 in combinations(models, 2):
        raw_dir = raw_scores[m1] < raw_scores[m2] if lower_is_better else raw_scores[m1] > raw_scores[m2]
        m1_bud = budgeted_scores.get(m1, float('inf') if lower_is_better else float('-inf'))
        m2_bud = budgeted_scores.get(m2, float('inf') if lower_is_better else float('-inf'))
        bud_dir = m1_bud < m2_bud if lower_is_better else m1_bud > m2_bud
        
        if raw_dir != bud_dir:
            reversals += 1
            
    return correlation, reversals
