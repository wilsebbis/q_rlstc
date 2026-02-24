"""Statistical utilities for experiment analysis.

Provides bootstrap confidence intervals and paired significance testing
for multi-seed experiment aggregation.
"""

import numpy as np
from typing import Tuple


def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Args:
        data: 1D array of observations (e.g., 5 seed results).
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (0.95 = 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        (mean, ci_low, ci_high) tuple.
    """
    data = np.asarray(data)
    rng = np.random.default_rng(seed)
    n = len(data)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = 1 - ci
    ci_low = np.percentile(boot_means, 100 * alpha / 2)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.mean(data)), float(ci_low), float(ci_high)


def paired_bootstrap_test(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> float:
    """Two-sided paired bootstrap significance test.

    Tests H0: mean(a) == mean(b) by resampling the paired differences.

    Args:
        a: Results from model A (one per seed).
        b: Results from model B (one per seed).
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        Two-sided p-value.
    """
    a, b = np.asarray(a), np.asarray(b)
    assert len(a) == len(b), "Must have same number of seeds"
    diffs = a - b
    observed = np.abs(np.mean(diffs))

    rng = np.random.default_rng(seed)
    n = len(diffs)
    count = 0
    for _ in range(n_bootstrap):
        signs = rng.choice([-1, 1], size=n)
        boot_mean = np.abs(np.mean(diffs * signs))
        if boot_mean >= observed:
            count += 1

    return count / n_bootstrap
