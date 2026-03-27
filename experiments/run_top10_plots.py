#!/usr/bin/env python3
"""Generate the 10 most important thesis plots for Q-RLSTC.

Runs experiments D1, E1, E2, E3, AB1, RA1 and produces 10 publication-quality
figures in a single invocation.

Usage::

    # Full run (5 seeds, ~90 min)
    python experiments/run_top10_plots.py --amount 30 --epochs 2 \
        --seeds 42,123,7,99,2025 --output-dir results/top10

    # Smoke test (~5 min)
    python experiments/run_top10_plots.py --amount 30 --epochs 1 \
        --seeds 42 --output-dir results/top10_smoke

    # Re-plot from saved JSON
    python experiments/run_top10_plots.py \
        --plots-only results/top10/top10_results.json
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ── Project bootstrap ─────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.run_thesis_experiments import (
    ModelSpec,
    PROTOCOL,
    build_agent,
    compute_fold_basesim,
    get_ablation_entanglement_specs,
    get_e1_specs,
    get_e2_specs,
    get_e3_specs,
    run_d1_valcr_sweep,
    run_ra1_reward_ablation,
    train_and_evaluate,
    _collect_env_metadata,
)

_DATA_DIR = _PROJECT_ROOT / "q_rlstc" / "data"

# ── Matplotlib setup ──────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

STYLE = {
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "lines.linewidth": 2,
    "lines.markersize": 8,
}
plt.rcParams.update(STYLE)

# Colour palette
C_QUANTUM = "#e6194B"
C_SPSA    = "#4363d8"
C_ADAM    = "#3cb44b"
C_RANDOM  = "#aaaaaa"
C_EAGLE   = "#f58231"
C_HERON   = "#911eb4"
C_IDEAL   = "#3cb44b"
C_NOCNOT  = "#42d4f4"
C_LINEAR  = "#e6194B"
C_SHAPED  = "#4363d8"
C_NAIVE   = "#f58231"


# ═════════════════════════════════════════════════════════════════════════
#  Experiment runners
# ═════════════════════════════════════════════════════════════════════════

def run_experiments(
    traj_path: str,
    centers_path: str,
    n_traj: int,
    n_epochs: int,
    seeds: List[int],
    output_dir: Path,
) -> Dict[str, Any]:
    """Run all 6 experiment groups and return collected results dict."""

    results: Dict[str, Any] = {
        "protocol": PROTOCOL,
        "env": _collect_env_metadata(),
        "config": {
            "n_traj": n_traj,
            "n_epochs": n_epochs,
            "seeds": seeds,
        },
    }

    json_path = output_dir / "top10_results.json"

    def _save():
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  [saved] {json_path}")

    # ── D1: Random policy CUT% sweep ─────────────────────────────
    print(f"\n{'═'*70}")
    print("  [1/6] D1: ValCR vs CUT% sweep")
    print(f"{'═'*70}")
    d1_n = min(n_traj, 100) if n_traj > 100 else n_traj
    results["D1"] = run_d1_valcr_sweep(traj_path, centers_path, d1_n, seeds[0])
    _save()

    # ── E1: Core Quantum Utility (all 10 models × seeds) ─────────
    print(f"\n{'═'*70}")
    print("  [2/6] E1: Core Quantum Utility (multi-seed)")
    print(f"{'═'*70}")
    e1_all = []   # flat list: every (model, seed) run
    for spec in get_e1_specs():
        for si, seed in enumerate(seeds):
            print(f"\n  [{si+1}/{len(seeds)}] seed={seed} → {spec.name}")
            agent = build_agent(spec, seed)
            r = train_and_evaluate(
                agent, spec, traj_path, centers_path,
                n_traj, n_epochs, seed,
            )
            r["seed"] = seed
            e1_all.append(r)
    results["E1"] = e1_all
    _save()

    # ── E2: NISQ Viability (Eagle / Heron noise) ─────────────────
    print(f"\n{'═'*70}")
    print("  [3/6] E2: NISQ Viability")
    print(f"{'═'*70}")
    e2_all = []
    for spec in get_e2_specs():
        for si, seed in enumerate(seeds):
            print(f"\n  [{si+1}/{len(seeds)}] seed={seed} → {spec.name}")
            agent = build_agent(spec, seed)
            r = train_and_evaluate(
                agent, spec, traj_path, centers_path,
                n_traj, n_epochs, seed,
            )
            r["seed"] = seed
            e2_all.append(r)
    results["E2"] = e2_all
    _save()

    # ── E3: Shot Sensitivity (128 / 512 / 2048) ──────────────────
    print(f"\n{'═'*70}")
    print("  [4/6] E3: Shot Sensitivity")
    print(f"{'═'*70}")
    e3_all = []
    for spec in get_e3_specs():
        for si, seed in enumerate(seeds):
            print(f"\n  [{si+1}/{len(seeds)}] seed={seed} → {spec.name}")
            agent = build_agent(spec, seed)
            r = train_and_evaluate(
                agent, spec, traj_path, centers_path,
                n_traj, n_epochs, seed,
            )
            r["seed"] = seed
            e3_all.append(r)
    results["E3"] = e3_all
    _save()

    # ── AB1: Entanglement Ablation ────────────────────────────────
    print(f"\n{'═'*70}")
    print("  [5/6] AB1: Entanglement Ablation")
    print(f"{'═'*70}")
    ab1_all = []
    for spec in get_ablation_entanglement_specs():
        for si, seed in enumerate(seeds):
            print(f"\n  [{si+1}/{len(seeds)}] seed={seed} → {spec.name}")
            agent = build_agent(spec, seed)
            r = train_and_evaluate(
                agent, spec, traj_path, centers_path,
                n_traj, n_epochs, seed,
            )
            r["seed"] = seed
            ab1_all.append(r)
    results["AB1"] = ab1_all
    _save()

    # ── RA1: Reward Stabilisation Ablation ────────────────────────
    print(f"\n{'═'*70}")
    print("  [6/6] RA1: Reward Stabilisation Ablation")
    print(f"{'═'*70}")
    results["RA1"] = run_ra1_reward_ablation(
        traj_path, centers_path,
        min(n_traj, 30), n_epochs, seeds[0],
    )
    _save()

    return results


# ═════════════════════════════════════════════════════════════════════════
#  Plotting helpers
# ═════════════════════════════════════════════════════════════════════════

def _agg_by_model(flat_results: List[Dict]) -> Dict[str, List[Dict]]:
    """Group a flat list of per-seed results by model name."""
    by_model: Dict[str, List[Dict]] = {}
    for r in flat_results:
        by_model.setdefault(r["model"], []).append(r)
    return by_model


def _bootstrap_ci(a: np.ndarray, b: np.ndarray, n: int = 10000, alpha: float = 0.05):
    """Bootstrap 95% CI on mean(a) - mean(b)."""
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(n):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs.append(np.mean(sa) - np.mean(sb))
    diffs = np.sort(diffs)
    lo = diffs[int(n * alpha / 2)]
    hi = diffs[int(n * (1 - alpha / 2))]
    return float(np.mean(diffs)), float(lo), float(hi)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size."""
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def _effect_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"


# ═════════════════════════════════════════════════════════════════════════
#  10 Plot generators
# ═════════════════════════════════════════════════════════════════════════

def plot_01_pareto_frontier(data: Dict, out: Path):
    """Plot 1: ValCR vs CUT% Pareto Frontier (E1 + D1)."""
    fig, ax = plt.subplots(figsize=(11, 7))

    # D1 random baseline curve
    d1 = data.get("D1", {})
    if d1 and "results" in d1:
        d1r = d1["results"]
        d1_cuts = [r["actual_cut_pct"] for r in d1r]
        d1_crs = [r["val_cr"] for r in d1r]
        ax.plot(d1_cuts, d1_crs, "o--", color=C_RANDOM, linewidth=1.5,
                markersize=6, label="Random policy (D1)", zorder=1)
        ax.fill_between(d1_cuts, d1_crs, max(d1_crs) * 1.1,
                        alpha=0.05, color=C_RANDOM)

    # E1 agent points (aggregate per model: mean across seeds)
    e1 = data.get("E1", [])
    by_model = _agg_by_model(e1)

    kind_style = {
        "quantum":   (C_QUANTUM, "D", 180),
        "classical": (C_SPSA,    "s", 120),
        "adam":       (C_ADAM,    "^", 120),
        "original":  ("#808080", "P", 120),
    }

    for model_name, runs in by_model.items():
        kind = runs[0].get("kind", "classical")
        params = runs[0].get("params", 34)
        color, marker, base_size = kind_style.get(kind, ("#808080", "o", 100))

        mean_cut = float(np.mean([r["cut_pct"] for r in runs]))
        mean_cr  = float(np.mean([r["val_cr"] for r in runs]))
        std_cr   = float(np.std([r["val_cr"] for r in runs])) if len(runs) > 1 else 0

        size = max(60, min(300, params * 0.8))
        ax.scatter(mean_cut, mean_cr, s=size, c=color, marker=marker,
                   edgecolors="black", linewidth=0.8, zorder=5)
        if std_cr > 0:
            ax.errorbar(mean_cut, mean_cr, yerr=std_cr, fmt="none",
                        ecolor=color, alpha=0.5, capsize=3, zorder=4)
        # Label
        short = model_name.split("(")[0].strip()
        ax.annotate(f"{short}\n({params}p)",
                    (mean_cut, mean_cr),
                    textcoords="offset points", xytext=(10, 6),
                    fontsize=6.5, color=color, alpha=0.9)

    # Budget threshold lines
    for thresh, ls in [(5, ":"), (10, "--"), (20, "-.")]:
        ax.axvline(x=thresh, color="#999", linestyle=ls, alpha=0.35,
                   label=f"CUT≤{thresh}%")

    ax.set_xlabel("CUT% (segmentation aggressiveness)")
    ax.set_ylabel("ValCR (lower = better)")
    ax.set_title("Plot 1 — Pareto Frontier: Learned Agents vs Random Baseline (E1+D1)")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(str(out / "01_pareto_frontier.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 01_pareto_frontier.png")


def plot_02_d1_length_sensitivity(data: Dict, out: Path):
    """Plot 2: D1 Length-Sensitivity Sweep (ValCR / nValCR / wValCR)."""
    d1 = data.get("D1", {})
    if not d1 or "results" not in d1:
        print("  ✗ Plot 2 skipped — no D1 data")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    d1r = d1["results"]
    probs = [r["cut_prob"] * 100 for r in d1r]

    for key, label, color, marker in [
        ("val_cr",   "ValCR (raw)",       C_QUANTUM, "o"),
        ("n_val_cr", "nValCR (per-point)", C_SPSA,    "s"),
        ("w_val_cr", "wValCR (weighted)",  C_ADAM,    "^"),
    ]:
        vals = [r.get(key, float("inf")) for r in d1r]
        finite = [v for v in vals if v < 1e6]
        if not finite:
            continue
        ax.plot(probs, vals, f"{marker}-", color=color, linewidth=2,
                markersize=8, label=label)
        for x, y in zip(probs, vals):
            if y < 1e6:
                ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=7, color=color)

    ax.set_xlabel("Random CUT Probability (%)")
    ax.set_ylabel("Metric Value (lower = better)")
    ax.set_title("Plot 2 — D1: ValCR Length-Sensitivity (Contribution C1)\n"
                 "Metric degeneracy: ValCR drops with CUT% even without learning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(out / "02_d1_length_sensitivity.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 02_d1_length_sensitivity.png")


def plot_03_od_basesim_decomposition(data: Dict, out: Path):
    """Plot 3: OD / BaseSim Decomposition Heatmap (E1)."""
    e1 = data.get("E1", [])
    if not e1:
        print("  ✗ Plot 3 skipped — no E1 data")
        return

    by_model = _agg_by_model(e1)
    models = sorted(by_model.keys(),
                    key=lambda m: np.mean([r["val_cr"] for r in by_model[m]]))

    col_labels = ["Model", "Params", "OD", "basesim", "CR(mean)", "CR(med)",
                  "CUT%", "#segs"]
    cell_data = []
    cell_colors = []

    for m in models:
        runs = by_model[m]
        od = np.mean([r.get("val_od", 0) for r in runs])
        bs = np.mean([r.get("val_basesim", 0) for r in runs])
        cr_mean = np.mean([r["val_cr"] for r in runs])
        cr_med = np.mean([r.get("val_cr_median", r["val_cr"]) for r in runs])
        cut = np.mean([r["cut_pct"] for r in runs])
        segs = int(np.mean([r["n_segs"] for r in runs]))
        params = runs[0].get("params", "?")

        row = [m[:22], str(params), f"{od:.4f}", f"{bs:.4f}",
               f"{cr_mean:.4f}", f"{cr_med:.4f}", f"{cut:.1f}%", str(segs)]
        cell_data.append(row)

        # Flag denominator pathology
        row_colors = ["white"] * 8
        if bs < 0.1:
            row_colors[3] = "#ffcccc"  # red tint for low basesim
        if abs(cr_mean - cr_med) > 0.5 * cr_mean and cr_mean > 0:
            row_colors[4] = "#fff3cc"  # yellow for mean≠median divergence
        cell_colors.append(row_colors)

    fig, ax = plt.subplots(figsize=(14, max(4, 0.45 * len(models) + 2)))
    ax.axis("off")
    table = ax.table(cellText=cell_data, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.5)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4363d8")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Apply cell colours
    for i, row_colors in enumerate(cell_colors):
        for j, c in enumerate(row_colors):
            if c != "white":
                table[i + 1, j].set_facecolor(c)
        # Alternate row shading
        if i % 2 == 1:
            for j in range(len(col_labels)):
                if cell_colors[i][j] == "white":
                    table[i + 1, j].set_facecolor("#f0f4ff")

    ax.set_title("Plot 3 — OD/BaseSim Decomposition (Contribution C5)\n"
                 "Red = basesim<0.1 (denominator pathology)  •  "
                 "Yellow = mean≠median CR divergence",
                 fontsize=11, pad=20)
    fig.tight_layout()
    fig.savefig(str(out / "03_od_basesim_decomposition.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 03_od_basesim_decomposition.png")


def plot_04_significance_forest(data: Dict, out: Path):
    """Plot 4: Significance Forest Plot (E1 multi-seed)."""
    e1 = data.get("E1", [])
    if not e1:
        print("  ✗ Plot 4 skipped — no E1 data")
        return

    by_model = _agg_by_model(e1)
    vqdqn_runs = by_model.get("VQ-DQN (5q×3L)", [])
    if len(vqdqn_runs) < 2:
        print("  ✗ Plot 4 skipped — need ≥2 seeds for significance")
        return

    vq_crs = np.array([r["val_cr"] for r in vqdqn_runs])

    # Compare against each SPSA control
    controls = [
        "MLP-34 (SPSA)", "Control A (linear)",
        "Control B (h=64)", "Control C (h=32×32)",
    ]

    comparisons = []
    for ctrl_name in controls:
        ctrl_runs = by_model.get(ctrl_name, [])
        if len(ctrl_runs) < 2:
            continue
        ctrl_crs = np.array([r["val_cr"] for r in ctrl_runs])

        # Mann-Whitney U
        try:
            from scipy.stats import mannwhitneyu
            stat, p_val = mannwhitneyu(vq_crs, ctrl_crs, alternative="two-sided")
        except (ImportError, ValueError):
            stat, p_val = 0, 1.0

        d = _cohens_d(vq_crs, ctrl_crs)
        mean_diff, ci_lo, ci_hi = _bootstrap_ci(vq_crs, ctrl_crs)

        comparisons.append({
            "label": f"VQ-DQN vs {ctrl_name}",
            "mean_diff": mean_diff,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "p": p_val,
            "d": d,
            "effect": _effect_label(d),
        })

    if not comparisons:
        print("  ✗ Plot 4 skipped — no valid control comparisons")
        return

    fig, ax = plt.subplots(figsize=(10, max(3, len(comparisons) * 0.8 + 2)))

    y_pos = list(range(len(comparisons)))
    for i, c in enumerate(comparisons):
        color = C_QUANTUM if c["p"] < 0.05 else "#999999"
        ax.errorbar(c["mean_diff"], i, xerr=[[c["mean_diff"] - c["ci_lo"]],
                    [c["ci_hi"] - c["mean_diff"]]], fmt="o", color=color,
                    capsize=5, markersize=8, linewidth=2)

        sig_str = "**" if c["p"] < 0.01 else "*" if c["p"] < 0.05 else "ns"
        ax.annotate(f"p={c['p']:.3f} {sig_str}  d={c['d']:+.2f} ({c['effect']})",
                    (c["ci_hi"] + 0.02, i), fontsize=8, va="center")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([c["label"] for c in comparisons], fontsize=9)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Mean ValCR Difference (VQ-DQN − Control)\n"
                  "Negative = VQ-DQN is better")
    ax.set_title("Plot 4 — Significance Forest Plot (Mann-Whitney U + Bootstrap 95% CI)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(str(out / "04_significance_forest.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 04_significance_forest.png")


def plot_05_param_efficiency(data: Dict, out: Path):
    """Plot 5: Parameter Efficiency Scatter."""
    e1 = data.get("E1", [])
    if not e1:
        print("  ✗ Plot 5 skipped — no E1 data")
        return

    by_model = _agg_by_model(e1)

    fig, ax = plt.subplots(figsize=(9, 6))

    kind_style = {
        "quantum":   (C_QUANTUM, "D", "VQ-DQN (SPSA)"),
        "classical": (C_SPSA,    "s", "Classical (SPSA)"),
        "adam":       (C_ADAM,    "^", "Classical (Adam)"),
        "original":  ("#808080", "P", "Original RLSTC"),
    }

    plotted_labels = set()
    for model_name, runs in by_model.items():
        kind = runs[0].get("kind", "classical")
        params = runs[0].get("params", 34)
        color, marker, legend_label = kind_style.get(kind, ("#808080", "o", "Other"))

        mean_cr = float(np.mean([r["val_cr"] for r in runs]))
        std_cr = float(np.std([r["val_cr"] for r in runs])) if len(runs) > 1 else 0

        label = legend_label if legend_label not in plotted_labels else None
        plotted_labels.add(legend_label)

        ax.scatter(params, mean_cr, s=140, c=color, marker=marker,
                   edgecolors="black", linewidth=0.8, zorder=5, label=label)
        if std_cr > 0:
            ax.errorbar(params, mean_cr, yerr=std_cr, fmt="none",
                        ecolor=color, alpha=0.4, capsize=3, zorder=4)

        short = model_name.split("(")[0].strip()[:12]
        ax.annotate(short, (params, mean_cr),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=7, alpha=0.8)

    ax.set_xscale("log")
    ax.set_xlabel("Parameter Count (log scale)")
    ax.set_ylabel("Best ValCR (lower = better)")
    ax.set_title("Plot 5 — Parameter Efficiency: 15–38× Compression Story")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(str(out / "05_param_efficiency.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 05_param_efficiency.png")


def plot_06_qmargin_evolution(data: Dict, out: Path):
    """Plot 6: Q-Margin Evolution (D2 piggyback from E1)."""
    e1 = data.get("E1", [])
    if not e1:
        print("  ✗ Plot 6 skipped — no E1 data")
        return

    by_model = _agg_by_model(e1)

    # Pick representative models (one per category) from first seed
    targets = [
        "VQ-DQN (5q×3L)", "MLP-34 (SPSA)",
        "Control B (h=64)", "Control C (h=32×32)",
    ]
    palette = [C_QUANTUM, C_SPSA, C_ADAM, C_EAGLE]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for model_name, color in zip(targets, palette):
        runs = by_model.get(model_name, [])
        if not runs:
            continue
        # Use first seed for per-epoch trajectory
        qm = runs[0].get("q_margins", [])
        if len(qm) < 1:
            continue
        epochs = list(range(1, len(qm) + 1))
        ax.plot(epochs, qm, "o-", color=color, linewidth=2, markersize=6,
                label=f"{model_name} ({runs[0].get('params', '?')}p)")

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5,
               label="Neutral (Q_ext = Q_cut)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Q-Margin = Q(extend) − Q(cut)")
    ax.set_title("Plot 6 — Q-Margin Evolution (D2)\n"
                 "Negative → policy prefers CUT (fragmentation attractor)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out / "06_qmargin_evolution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 06_qmargin_evolution.png")


def plot_07_nisq_noise(data: Dict, out: Path):
    """Plot 7: NISQ Noise Degradation (E2 + E1 ideal)."""
    e1 = data.get("E1", [])
    e2 = data.get("E2", [])
    if not e2:
        print("  ✗ Plot 7 skipped — no E2 data")
        return

    # Get ideal VQ-DQN from E1
    by_e1 = _agg_by_model(e1)
    ideal_runs = by_e1.get("VQ-DQN (5q×3L)", [])
    ideal_cr = float(np.mean([r["val_cr"] for r in ideal_runs])) if ideal_runs else None

    # E2 results
    by_e2 = _agg_by_model(e2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                    gridspec_kw={"width_ratios": [2, 1]})

    # Left panel: ValCR bars
    labels, crs, stds, colors = [], [], [], []
    if ideal_cr is not None:
        labels.append("Ideal\n(statevector)")
        crs.append(ideal_cr)
        stds.append(float(np.std([r["val_cr"] for r in ideal_runs])) if len(ideal_runs) > 1 else 0)
        colors.append(C_IDEAL)

    noise_colors = {"eagle": C_EAGLE, "heron": C_HERON}
    for model_name, runs in by_e2.items():
        noise = runs[0].get("noise", "unknown")
        labels.append(f"{noise.capitalize()}\nnoise model")
        crs.append(float(np.mean([r["val_cr"] for r in runs])))
        stds.append(float(np.std([r["val_cr"] for r in runs])) if len(runs) > 1 else 0)
        colors.append(noise_colors.get(noise, "#999"))

    x = np.arange(len(labels))
    bars = ax1.bar(x, crs, yerr=stds, color=colors, alpha=0.85,
                   capsize=5, edgecolor="black", linewidth=0.5)
    for bar, cr in zip(bars, crs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{cr:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("ValCR (lower = better)")
    ax1.set_title("ValCR Under Noise Models")

    # Right panel: Resilience ratio
    if ideal_cr and ideal_cr > 0:
        ratios = []
        ratio_labels = []
        ratio_colors = []
        for label, cr, color in zip(labels[1:], crs[1:], colors[1:]):
            ratio = cr / ideal_cr
            ratios.append(ratio)
            ratio_labels.append(label.replace("\n", " "))
            ratio_colors.append(color)

        x2 = np.arange(len(ratios))
        bars2 = ax2.barh(x2, ratios, color=ratio_colors, alpha=0.85,
                         edgecolor="black", linewidth=0.5)
        ax2.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)
        ax2.axvline(x=1.2, color="red", linestyle=":", alpha=0.4,
                    label="20% degradation")
        for bar, ratio in zip(bars2, ratios):
            status = "✓" if ratio < 1.2 else "⚠"
            ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{ratio:.2f}× {status}", va="center", fontsize=9)
        ax2.set_yticks(x2)
        ax2.set_yticklabels(ratio_labels)
        ax2.set_xlabel("CR_noisy / CR_ideal (1.0 = no degradation)")
        ax2.set_title("Resilience Ratio")
        ax2.legend(fontsize=8)

    fig.suptitle("Plot 7 — NISQ Noise Degradation (E2, Contribution C4)", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(out / "07_nisq_noise.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 07_nisq_noise.png")


def plot_08_shot_sensitivity(data: Dict, out: Path):
    """Plot 8: Shot Sensitivity Curve (E3 + statevector baseline)."""
    e1 = data.get("E1", [])
    e3 = data.get("E3", [])
    if not e3:
        print("  ✗ Plot 8 skipped — no E3 data")
        return

    # Statevector baseline from E1
    by_e1 = _agg_by_model(e1)
    ideal_runs = by_e1.get("VQ-DQN (5q×3L)", [])
    ideal_cr = float(np.mean([r["val_cr"] for r in ideal_runs])) if ideal_runs else None

    # E3 results: group by shot count
    shot_map: Dict[int, List[float]] = {}
    for r in e3:
        shots = r.get("config", {}).get("shots", 0)
        if shots == 0:
            # Try to extract from model name
            name = r.get("model", "")
            for s in [128, 512, 2048]:
                if str(s) in name:
                    shots = s
                    break
        shot_map.setdefault(shots, []).append(r["val_cr"])

    if not shot_map:
        print("  ✗ Plot 8 skipped — no valid shot data")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))

    shots_sorted = sorted(shot_map.keys())
    mean_crs = [float(np.mean(shot_map[s])) for s in shots_sorted]
    std_crs = [float(np.std(shot_map[s])) for s in shots_sorted] if any(len(v) > 1 for v in shot_map.values()) else [0] * len(shots_sorted)

    ax.errorbar(shots_sorted, mean_crs, yerr=std_crs, fmt="o-",
                color=C_QUANTUM, linewidth=2, markersize=10, capsize=5,
                label="VQ-DQN (finite shots)")

    for x, y in zip(shots_sorted, mean_crs):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9)

    # Statevector baseline
    if ideal_cr is not None:
        ax.axhline(y=ideal_cr, color=C_IDEAL, linestyle="--", linewidth=1.5,
                   label=f"Statevector (exact): {ideal_cr:.3f}")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Shots per Circuit Evaluation (log₂ scale)")
    ax.set_ylabel("ValCR (lower = better)")
    ax.set_title("Plot 8 — Shot Sensitivity (E3): Minimal Shot Floor")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(out / "08_shot_sensitivity.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 08_shot_sensitivity.png")


def plot_09_entanglement_ablation(data: Dict, out: Path):
    """Plot 9: Entanglement Ablation (AB1)."""
    ab1 = data.get("AB1", [])
    if not ab1:
        print("  ✗ Plot 9 skipped — no AB1 data")
        return

    by_model = _agg_by_model(ab1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    models = list(by_model.keys())
    palette = {
        "VQ-DQN (no-CNOT)": C_NOCNOT,
        "VQ-DQN (linear)":  C_LINEAR,
    }

    # Left panel: ValCR
    x = np.arange(len(models))
    crs = []
    stds = []
    colors = []
    for m in models:
        runs = by_model[m]
        crs.append(float(np.mean([r["val_cr"] for r in runs])))
        stds.append(float(np.std([r["val_cr"] for r in runs])) if len(runs) > 1 else 0)
        colors.append(palette.get(m, "#999"))

    bars = ax1.bar(x, crs, yerr=stds, color=colors, alpha=0.85,
                   capsize=5, edgecolor="black", linewidth=0.5)
    for bar, cr in zip(bars, crs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{cr:.4f}", ha="center", fontsize=10, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace("VQ-DQN ", "") for m in models])
    ax1.set_ylabel("Best ValCR (lower = better)")
    ax1.set_title("ValCR Comparison")

    # Right panel: CUT%
    cuts = []
    cut_stds = []
    for m in models:
        runs = by_model[m]
        cuts.append(float(np.mean([r["cut_pct"] for r in runs])))
        cut_stds.append(float(np.std([r["cut_pct"] for r in runs])) if len(runs) > 1 else 0)

    bars2 = ax2.bar(x, cuts, yerr=cut_stds, color=colors, alpha=0.85,
                    capsize=5, edgecolor="black", linewidth=0.5)
    for bar, cut in zip(bars2, cuts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{cut:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace("VQ-DQN ", "") for m in models])
    ax2.set_ylabel("CUT%")
    ax2.set_title("CUT% Comparison")

    # Verdict
    if len(crs) >= 2:
        diff_pct = 100 * (crs[0] - crs[1]) / max(crs[1], 1e-8)
        if abs(diff_pct) > 10:
            verdict = "Entanglement contributes to policy quality"
        else:
            verdict = "Performance similar → efficiency from generic architecture"
        fig.suptitle(f"Plot 9 — Entanglement Ablation (AB1)\n{verdict}",
                     fontsize=12)
    else:
        fig.suptitle("Plot 9 — Entanglement Ablation (AB1)", fontsize=12)

    fig.tight_layout()
    fig.savefig(str(out / "09_entanglement_ablation.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 09_entanglement_ablation.png")


def plot_10_reward_stabilisation(data: Dict, out: Path):
    """Plot 10: Reward Stabilisation Ablation (RA1)."""
    ra1 = data.get("RA1", {})
    if not ra1 or "results" not in ra1:
        print("  ✗ Plot 10 skipped — no RA1 data")
        return

    ra1r = ra1["results"]
    if len(ra1r) < 2:
        print("  ✗ Plot 10 skipped — need ≥2 RA1 conditions")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    names = [r["condition"] for r in ra1r]
    colors = [C_SHAPED if "shaped" in n.lower() else C_NAIVE for n in names]

    # Panel 1: Best ValCR
    ax = axes[0]
    crs = [r["best_val_cr"] for r in ra1r]
    bars = ax.bar(range(len(names)), crs, color=colors, alpha=0.85,
                  edgecolor="black", linewidth=0.5)
    for bar, cr in zip(bars, crs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{cr:.4f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("VQ-DQN ", "") for n in names], fontsize=9)
    ax.set_ylabel("Best ValCR")
    ax.set_title("ValCR (lower = better)")

    # Panel 2: Final CUT%
    ax = axes[1]
    cuts = [r["final_cut_pct"] for r in ra1r]
    bars = ax.bar(range(len(names)), cuts, color=colors, alpha=0.85,
                  edgecolor="black", linewidth=0.5)
    for bar, cut in zip(bars, cuts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{cut:.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("VQ-DQN ", "") for n in names], fontsize=9)
    ax.set_ylabel("Final CUT%")
    ax.set_title("Segmentation Rate")

    # Panel 3: Reward shaping components table
    ax = axes[2]
    ax.axis("off")
    table_data = [
        ["Component", "Shaped", "Naive"],
        ["L_MIN", str(ra1r[0].get("l_min", "?")), str(ra1r[1].get("l_min", "?"))],
        ["CUT_PENALTY", f"{ra1r[0].get('cut_penalty', 0):.2f}",
         f"{ra1r[1].get('cut_penalty', 0):.2f}"],
        ["EXTEND_COST", f"{ra1r[0].get('extend_cost', 0):.2f}",
         f"{ra1r[1].get('extend_cost', 0):.2f}"],
        ["COMPLEXITY_λ", f"{ra1r[0].get('complexity_lambda', 0):.2f}",
         f"{ra1r[1].get('complexity_lambda', 0):.2f}"],
    ]
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for j in range(3):
        table[0, j].set_facecolor("#4363d8")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Ablation Config", fontsize=10)

    # Verdict
    naive = next((r for r in ra1r if "naive" in r["condition"].lower()), None)
    shaped = next((r for r in ra1r if "shaped" in r["condition"].lower()), None)
    if naive and shaped:
        naive_cut = naive["final_cut_pct"]
        verdict = (f"Naive → CUT={naive_cut:.0f}% "
                   f"({'DEGENERATE' if naive_cut > 50 else 'partial'})")
    else:
        verdict = ""

    fig.suptitle(f"Plot 10 — Reward Stabilisation Ablation (RA1/E6)\n{verdict}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out / "10_reward_stabilisation.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 10_reward_stabilisation.png")


# ═════════════════════════════════════════════════════════════════════════
#  Master plot generator
# ═════════════════════════════════════════════════════════════════════════

ALL_PLOTS = [
    ("01", plot_01_pareto_frontier),
    ("02", plot_02_d1_length_sensitivity),
    ("03", plot_03_od_basesim_decomposition),
    ("04", plot_04_significance_forest),
    ("05", plot_05_param_efficiency),
    ("06", plot_06_qmargin_evolution),
    ("07", plot_07_nisq_noise),
    ("08", plot_08_shot_sensitivity),
    ("09", plot_09_entanglement_ablation),
    ("10", plot_10_reward_stabilisation),
]


def generate_all_plots(data: Dict, plot_dir: Path):
    """Generate all 10 thesis plots."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'═'*60}")
    print(f"  Generating 10 Thesis Plots → {plot_dir}")
    print(f"{'═'*60}\n")

    n_ok = 0
    for num, fn in ALL_PLOTS:
        try:
            fn(data, plot_dir)
            n_ok += 1
        except Exception as e:
            print(f"  ✗ Plot {num} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n  Generated {n_ok}/10 plots → {plot_dir}\n")
    return n_ok


# ═════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate the 10 most important Q-RLSTC thesis plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--traj-path",
                        default=str(_DATA_DIR / "Tdrive_norm_traj"))
    parser.add_argument("--centers-path",
                        default=str(_DATA_DIR / "tdrive_clustercenter"))
    parser.add_argument("--amount", type=int, default=30,
                        help="Number of trajectories (default: 30)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Training epochs per model (default: 2)")
    parser.add_argument("--seeds", default="42",
                        help="Comma-separated seeds (default: 42)")
    parser.add_argument("--output-dir", default="results/top10",
                        help="Output directory (default: results/top10)")
    parser.add_argument("--plots-only", default=None,
                        help="Path to existing JSON — regenerate plots only")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"

    seed_list = [int(s.strip()) for s in args.seeds.split(",")]

    # ── Plots-only mode ──────────────────────────────────────────
    if args.plots_only:
        print(f"  Loading results from {args.plots_only}")
        with open(args.plots_only) as f:
            data = json.load(f)
        generate_all_plots(data, plot_dir)
        return 0

    # ── Full experiment run ──────────────────────────────────────
    t0 = time.time()
    print(f"\n{'═'*70}")
    print(f"  Q-RLSTC TOP-10 THESIS PLOTS")
    print(f"  {datetime.now().isoformat()}")
    print(f"  {args.amount} trajectories, {args.epochs} epochs, seeds={seed_list}")
    print(f"  Output: {output_dir}")
    print(f"{'═'*70}\n")

    data = run_experiments(
        traj_path=args.traj_path,
        centers_path=args.centers_path,
        n_traj=args.amount,
        n_epochs=args.epochs,
        seeds=seed_list,
        output_dir=output_dir,
    )

    # ── Generate plots ───────────────────────────────────────────
    n_ok = generate_all_plots(data, plot_dir)

    elapsed = time.time() - t0
    print(f"{'═'*70}")
    print(f"  ✓ COMPLETE — {n_ok}/10 plots in {elapsed:.0f}s")
    print(f"  JSON:  {output_dir / 'top10_results.json'}")
    print(f"  Plots: {plot_dir}")
    print(f"{'═'*70}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
