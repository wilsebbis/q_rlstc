#!/usr/bin/env python3
"""Post-process multi-seed experiment results with significance tests.

Loads the JSON from a multi-seed thesis run and computes:
  - Mann-Whitney U test (VQ-DQN vs each control)
  - Cohen's d effect size
  - Bootstrap 95% confidence intervals
  - Summary table with mean ± std and p-values

Usage:
    python experiments/run_significance_test.py \
        results/thesis_multiseed/thesis_results_*.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def cohens_d(a: list, b: list) -> float:
    """Compute Cohen's d effect size (positive = a < b = a is better)."""
    na, nb = np.array(a), np.array(b)
    pooled_std = np.sqrt(((len(na)-1)*np.var(na, ddof=1) +
                          (len(nb)-1)*np.var(nb, ddof=1)) /
                         (len(na) + len(nb) - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(nb) - np.mean(na)) / pooled_std)


def bootstrap_ci(data: list, n_boot: int = 10000, ci: float = 0.95,
                 seed: int = 42) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(seed)
    arr = np.array(data)
    means = np.array([
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return float(np.percentile(means, 100*alpha)), \
           float(np.percentile(means, 100*(1-alpha)))


def mann_whitney_u(a: list, b: list) -> tuple[float, float]:
    """Mann-Whitney U test. Returns (U statistic, p-value).

    Uses scipy if available, otherwise a simple approximation.
    """
    try:
        from scipy.stats import mannwhitneyu
        stat, p = mannwhitneyu(a, b, alternative='two-sided')
        return float(stat), float(p)
    except ImportError:
        # Simple rank-sum approximation
        combined = [(v, 0) for v in a] + [(v, 1) for v in b]
        combined.sort(key=lambda x: x[0])
        rank_sum_a = sum(i+1 for i, (_, g) in enumerate(combined) if g == 0)
        n1, n2 = len(a), len(b)
        u1 = rank_sum_a - n1*(n1+1)/2
        # Normal approximation
        mu = n1*n2/2
        sigma = np.sqrt(n1*n2*(n1+n2+1)/12)
        if sigma == 0:
            return float(u1), 1.0
        z = abs(u1 - mu) / sigma
        # Two-sided p-value from normal approx
        from math import erfc
        p = erfc(z / np.sqrt(2))
        return float(u1), float(p)


def effect_size_label(d: float) -> str:
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def main():
    parser = argparse.ArgumentParser(
        description="Significance tests for multi-seed Q-RLSTC results"
    )
    parser.add_argument("json_path", help="Path to thesis_results JSON")
    parser.add_argument("--output", default=None,
                        help="Output markdown path (default: alongside JSON)")
    args = parser.parse_args()

    with open(args.json_path) as f:
        data = json.load(f)

    e1 = data.get("E1")
    if not e1:
        print("ERROR: No E1 results found in JSON.", file=sys.stderr)
        return 1

    # Find quantum model
    quantum = [r for r in e1 if r.get("kind") == "quantum"]
    controls = [r for r in e1 if r.get("kind") in ("classical", "adam")]

    if not quantum:
        print("ERROR: No quantum model found in E1.", file=sys.stderr)
        return 1

    q = quantum[0]
    q_crs = q.get("per_seed_crs", [q["val_cr"]])
    q_name = q["model"]
    n_seeds = q.get("n_seeds", len(q_crs))

    print(f"\n{'='*80}")
    print(f"  Significance Analysis: {q_name} vs Controls")
    print(f"  Seeds: {n_seeds}")
    print(f"{'='*80}\n")

    # Reference: quantum stats
    q_mean = np.mean(q_crs)
    q_std = np.std(q_crs, ddof=1) if len(q_crs) > 1 else 0
    q_lo, q_hi = bootstrap_ci(q_crs) if len(q_crs) > 1 else (q_mean, q_mean)

    print(f"  {q_name}: CR = {q_mean:.4f} +/- {q_std:.4f}  "
          f"[95% CI: {q_lo:.4f} - {q_hi:.4f}]")
    print(f"  Per-seed CRs: {[f'{c:.4f}' for c in q_crs]}\n")

    lines = []
    lines.append(f"# Significance Analysis: Multi-Seed E1 Results\n")
    lines.append(f"Seeds: {n_seeds}\n")
    lines.append(f"## {q_name}\n")
    lines.append(f"- Mean CR: **{q_mean:.4f} +/- {q_std:.4f}**")
    lines.append(f"- 95% Bootstrap CI: [{q_lo:.4f}, {q_hi:.4f}]")
    lines.append(f"- Per-seed: {[round(c, 4) for c in q_crs]}\n")
    lines.append("## Pairwise Comparisons\n")
    lines.append("| Control | Params | Mean CR | Std | U-stat | p-value | "
                 "Cohen's d | Effect | Sig? |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    print(f"{'Control':<30s} {'Params':>6s} {'Mean CR':>10s} {'Std':>8s} "
          f"{'U':>8s} {'p':>8s} {'d':>8s} {'Effect':>10s} {'Sig?':>5s}")
    print("-" * 100)

    for c in controls:
        c_crs = c.get("per_seed_crs", [c["val_cr"]])
        c_name = c["model"]
        c_mean = np.mean(c_crs)
        c_std = np.std(c_crs, ddof=1) if len(c_crs) > 1 else 0

        if len(q_crs) > 1 and len(c_crs) > 1:
            u, p = mann_whitney_u(q_crs, c_crs)
            d = cohens_d(q_crs, c_crs)
        else:
            u, p, d = 0, 1.0, 0.0

        eff = effect_size_label(d)
        sig = "YES" if p < 0.05 else "no"

        print(f"{c_name:<30s} {c['params']:>6d} {c_mean:>10.4f} {c_std:>8.4f} "
              f"{u:>8.1f} {p:>8.4f} {d:>+8.3f} {eff:>10s} {sig:>5s}")

        lines.append(f"| {c_name} | {c['params']} | {c_mean:.4f} | {c_std:.4f} "
                     f"| {u:.1f} | {p:.4f} | {d:+.3f} | {eff} | {sig} |")

    print("-" * 100)

    # Interpretation
    lines.append("\n## Interpretation\n")
    spsa_controls = [c for c in controls if c.get("kind") == "classical"]
    if spsa_controls:
        best_ctrl = min(spsa_controls, key=lambda c: np.mean(c.get("per_seed_crs", [c["val_cr"]])))
        bc_crs = best_ctrl.get("per_seed_crs", [best_ctrl["val_cr"]])
        bc_mean = np.mean(bc_crs)
        if len(q_crs) > 1 and len(bc_crs) > 1:
            _, p = mann_whitney_u(q_crs, bc_crs)
            d = cohens_d(q_crs, bc_crs)
            if p < 0.05:
                lines.append(
                    f"The VQ-DQN ({q['params']} params) significantly outperforms "
                    f"the best SPSA control ({best_ctrl['model']}, {best_ctrl['params']} params) "
                    f"with p={p:.4f}, Cohen's d={d:+.3f} ({effect_size_label(d)} effect). "
                    f"This supports **Claim B** (parameter-efficient inductive bias under SPSA)."
                )
            else:
                lines.append(
                    f"The difference between VQ-DQN and {best_ctrl['model']} is "
                    f"not statistically significant (p={p:.4f}). "
                    f"**Claim B requires more seeds or a larger training budget to confirm.**"
                )
        else:
            lines.append("Insufficient seeds for significance testing.")

    # Write output
    out_path = args.output
    if not out_path:
        out_path = str(Path(args.json_path).parent / "significance_analysis.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
