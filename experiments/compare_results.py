#!/usr/bin/env python3
"""Direct head-to-head comparison of RLSTCcode vs Q-RLSTC results.

Loads both JSON result files, normalises metrics, and produces a unified
comparison report with publication-ready plots.

Usage:
    python experiments/compare_results.py \
        --classical results/classical_baseline/classical_results.json \
        --quantum   results/thesis_qfix/thesis_results_20260224_121334.json \
        --output    results/comparison
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch, FancyBboxPatch
import numpy as np


# ── Palette ───────────────────────────────────────────────────────────────

C = {
    "bg":        "#0F172A",
    "card":      "#1E293B",
    "text":      "#E2E8F0",
    "grid":      "#334155",
    "blue":      "#3B82F6",
    "purple":    "#8B5CF6",
    "slate":     "#94A3B8",
    "amber":     "#F59E0B",
    "green":     "#10B981",
    "red":       "#EF4444",
    "cyan":      "#06B6D4",
}


def _ax(ax: plt.Axes) -> None:
    ax.set_facecolor(C["card"])
    ax.tick_params(colors=C["text"], labelsize=9)
    ax.xaxis.label.set_color(C["text"])
    ax.yaxis.label.set_color(C["text"])
    ax.title.set_color(C["text"])
    for s in ax.spines.values():
        s.set_color(C["grid"])
    ax.grid(axis="y", color=C["grid"], alpha=0.4, linewidth=0.5)


# ── Data loading ──────────────────────────────────────────────────────────

def load_results(classical_path: str, quantum_path: str) -> tuple[dict, dict]:
    with open(classical_path) as f:
        cl = json.load(f)
    with open(quantum_path) as f:
        qu = json.load(f)
    return cl, qu


def build_comparison_rows(cl: dict, qu: dict) -> list[dict]:
    """Build a flat list of {label, system, kind, params, val_cr, data, optimizer, ...}."""
    rows = []

    # Classical: best data-size
    if "data_size_experiment" in cl:
        best = min(cl["data_size_experiment"], key=lambda x: x["val_cr"])
        rows.append({
            "label": f"RLSTCcode (best, {best['trajectories']//1000}k)",
            "system": "RLSTCcode", "kind": "classical",
            "params": 514, "val_cr": best["val_cr"],
            "training_data": best["trajectories"],
            "optimizer": "SGD", "detail": "best data-size",
        })

    # Classical: 5-fold mean
    if "kfold_cv" in cl:
        cv = cl["kfold_cv"]
        rows.append({
            "label": "RLSTCcode (5-fold CV)",
            "system": "RLSTCcode", "kind": "classical",
            "params": 514, "val_cr": cv["mean_cr"],
            "val_cr_std": cv["std_cr"],
            "training_data": 4000,
            "optimizer": "SGD", "detail": "5-fold CV mean",
        })

    # Classical: all individual data-size experiments
    if "data_size_experiment" in cl:
        for e in cl["data_size_experiment"]:
            rows.append({
                "label": f"RLSTCcode ({e['trajectories']//1000}k)",
                "system": "RLSTCcode", "kind": "classical",
                "params": 514, "val_cr": e["val_cr"],
                "training_data": e["trajectories"],
                "optimizer": "SGD", "detail": "data-size",
            })

    # Classical: cluster-count
    if "cluster_count_experiment" in cl:
        for e in cl["cluster_count_experiment"]:
            rows.append({
                "label": f"RLSTCcode (k={e['k']})",
                "system": "RLSTCcode", "kind": "classical",
                "params": 514, "val_cr": e["val_cr"],
                "training_data": "varies",
                "optimizer": "SGD", "detail": "cluster-count",
            })

    # Quantum E1 models
    if "E1" in qu:
        for entry in qu["E1"]:
            rows.append({
                "label": entry["model"],
                "system": "Q-RLSTC", "kind": entry["kind"],
                "params": entry["params"], "val_cr": entry["val_cr"],
                "cut_pct": entry.get("cut_pct", 0),
                "n_segs": entry.get("n_segs", 0),
                "wall_time": entry.get("wall_time", 0),
                "q_margin": entry.get("q_margins", [None])[-1],
                "training_data": 30,
                "optimizer": "SPSA", "detail": "E1",
            })

    # Other models (modelstate4)
    if "other_models" in cl:
        for e in cl["other_models"]:
            rows.append({
                "label": f"RLSTCcode ({e['label']})",
                "system": "RLSTCcode", "kind": "classical",
                "params": 514, "val_cr": e["val_cr"],
                "training_data": "unknown",
                "optimizer": "SGD", "detail": "other",
            })

    return rows


# ── Plot 1: Unified CR bar chart ─────────────────────────────────────────

def plot_unified_bars(rows: list[dict], out: Path) -> None:
    """All models side-by-side, grouped by system."""
    # Pick representative rows
    display = [r for r in rows if r["detail"] in ("best data-size", "5-fold CV mean", "E1")]
    display.sort(key=lambda r: r["val_cr"])

    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor(C["bg"])
    _ax(ax)

    labels, crs, colors = [], [], []
    for r in display:
        labels.append(f"{r['label']}\n({r['params']}p, {r['optimizer']})")
        crs.append(r["val_cr"])
        if r["kind"] == "quantum":
            colors.append(C["purple"])
        elif r["system"] == "RLSTCcode":
            colors.append(C["blue"])
        else:
            colors.append(C["slate"])

    bars = ax.bar(range(len(labels)), crs, color=colors, width=0.65,
                  edgecolor="white", linewidth=0.5)

    for i, (bar, cr) in enumerate(zip(bars, crs)):
        va = "bottom"
        y = cr + 0.08
        if cr > 5:
            va = "top"
            y = cr - 0.3
        ax.text(i, y, f"{cr:.4f}", ha="center", va=va,
                fontsize=9, color=C["text"], fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Competitive Ratio (lower = better)")
    ax.set_title("Direct Comparison: All Models Ranked by Clustering Quality")

    legend = [
        Patch(facecolor=C["blue"], label="RLSTCcode (SGD, 500–5k trajs)"),
        Patch(facecolor=C["purple"], label="Q-RLSTC Quantum (SPSA, 30 trajs)"),
        Patch(facecolor=C["slate"], label="Q-RLSTC Classical Controls (SPSA, 30 trajs)"),
    ]
    ax.legend(handles=legend, loc="upper left",
              facecolor=C["card"], edgecolor=C["grid"],
              labelcolor=C["text"], fontsize=8)

    plt.tight_layout()
    fig.savefig(out / "unified_comparison.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Plot 2: Parameter efficiency (improved) ──────────────────────────────

def plot_param_efficiency(rows: list[dict], out: Path) -> None:
    display = [r for r in rows if r["detail"] in ("best data-size", "5-fold CV mean", "E1")]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(C["bg"])
    _ax(ax)

    for r in display:
        if r["kind"] == "quantum":
            color, marker, sz = C["purple"], "D", 180
        elif r["system"] == "RLSTCcode":
            color, marker, sz = C["blue"], "s", 140
        else:
            color, marker, sz = C["slate"], "o", 100

        ax.scatter(r["params"], r["val_cr"], s=sz, color=color, zorder=5,
                   marker=marker, edgecolors="white", linewidth=1.2)

        # Label positioning
        cr = r["val_cr"]
        p = r["params"]
        if p < 20:
            xytext = (14, 4)
        elif p > 1000:
            xytext = (-10, 12)
        elif cr > 5:
            xytext = (14, -8)
        else:
            xytext = (14, 4)

        ax.annotate(
            f"{r['label']}\nCR={cr:.4f}",
            (p, cr), textcoords="offset points", xytext=xytext,
            fontsize=7.5, color=C["text"],
            arrowprops=dict(arrowstyle="-", color=C["grid"], lw=0.5),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters (log scale)")
    ax.set_ylabel("Competitive Ratio (lower = better)")
    ax.set_title("Parameter Efficiency: Quality per Parameter")

    # Draw efficiency frontier arrow from VQ-DQN to best classical
    quantum = [r for r in display if r["kind"] == "quantum"]
    classical_best = [r for r in display if r["system"] == "RLSTCcode"]
    if quantum and classical_best:
        q = quantum[0]
        cb = min(classical_best, key=lambda r: r["val_cr"])
        ax.annotate("",
                    xy=(cb["params"], cb["val_cr"]),
                    xytext=(q["params"], q["val_cr"]),
                    arrowprops=dict(arrowstyle="->", color=C["amber"],
                                   lw=1.5, linestyle="--"))
        mid_p = (q["params"] * cb["params"]) ** 0.5
        mid_cr = (q["val_cr"] + cb["val_cr"]) / 2
        ax.text(mid_p, mid_cr + 0.15,
                f'15× fewer params\n{cb["val_cr"]/q["val_cr"]:.1f}× better CR\n(100× more data)',
                fontsize=7, color=C["amber"], ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=C["card"],
                          edgecolor=C["amber"], alpha=0.8))

    legend = [
        Patch(facecolor=C["blue"], label="RLSTCcode (SGD)"),
        Patch(facecolor=C["purple"], label="Quantum VQ-DQN (SPSA)"),
        Patch(facecolor=C["slate"], label="Classical Controls (SPSA)"),
    ]
    ax.legend(handles=legend, loc="upper right",
              facecolor=C["card"], edgecolor=C["grid"],
              labelcolor=C["text"], fontsize=8)

    plt.tight_layout()
    fig.savefig(out / "parameter_efficiency.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Plot 3: Training data scaling ────────────────────────────────────────

def plot_data_scaling(rows: list[dict], out: Path) -> None:
    """Show how CR scales with training data, with Q-RLSTC as a reference."""
    data_rows = [r for r in rows if r["detail"] == "data-size"]
    quantum = [r for r in rows if r["kind"] == "quantum" and r["detail"] == "E1"]
    ctrl_c = [r for r in rows if r["detail"] == "E1" and "32×32" in r.get("label", "")]

    if not data_rows:
        return

    data_rows.sort(key=lambda r: r["training_data"])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(C["bg"])
    _ax(ax)

    xs = [r["training_data"] for r in data_rows]
    ys = [r["val_cr"] for r in data_rows]

    ax.plot(xs, ys, "o-", color=C["blue"], linewidth=2, markersize=8,
            label="RLSTCcode DQN (514 params, SGD)", zorder=5)
    for x, y in zip(xs, ys):
        ax.text(x, y - 0.025, f"{y:.4f}", ha="center", va="top",
                fontsize=8, color=C["blue"], fontweight="bold")

    # Q-RLSTC VQ-DQN reference line
    if quantum:
        q_cr = quantum[0]["val_cr"]
        ax.axhline(q_cr, color=C["purple"], linestyle="--", linewidth=1.5, alpha=0.8)
        ax.text(xs[-1], q_cr + 0.02, f"VQ-DQN (34p, SPSA, 30 trajs) = {q_cr:.4f}",
                fontsize=8, color=C["purple"], ha="right", va="bottom")

    # Control C reference
    if ctrl_c:
        cc_cr = ctrl_c[0]["val_cr"]
        ax.axhline(cc_cr, color=C["slate"], linestyle=":", linewidth=1.2, alpha=0.7)
        ax.text(xs[-1], cc_cr + 0.02, f"Control C (1314p, SPSA, 30 trajs) = {cc_cr:.4f}",
                fontsize=8, color=C["slate"], ha="right", va="bottom")

    # CR=1.0 reference
    ax.axhline(1.0, color=C["amber"], linestyle="-.", linewidth=1, alpha=0.5)
    ax.text(xs[0], 1.02, "CR=1.0 (baseline parity)", fontsize=7,
            color=C["amber"], alpha=0.7)

    ax.set_xlabel("Training Trajectories")
    ax.set_ylabel("Competitive Ratio (lower = better)")
    ax.set_title("Data Scaling: More Data Closes the Gap")
    ax.legend(facecolor=C["card"], edgecolor=C["grid"],
              labelcolor=C["text"], fontsize=8, loc="upper right")

    plt.tight_layout()
    fig.savefig(out / "data_scaling.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Plot 4: Controlled comparison (Q-RLSTC internal) ─────────────────────

def plot_controlled_comparison(qu: dict, out: Path) -> None:
    """Bar chart of only Q-RLSTC models under identical conditions."""
    if "E1" not in qu:
        return

    entries = qu["E1"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                    gridspec_kw={"width_ratios": [3, 2]})
    fig.patch.set_facecolor(C["bg"])

    # Left: ValCR
    _ax(ax1)
    labels = [e["model"] for e in entries]
    crs = [e["val_cr"] for e in entries]
    params = [e["params"] for e in entries]
    kinds = [e["kind"] for e in entries]
    colors = [C["purple"] if k == "quantum" else C["slate"] for k in kinds]

    bars = ax1.bar(range(len(labels)), crs, color=colors, width=0.6,
                   edgecolor="white", linewidth=0.5)

    for i, (cr, p) in enumerate(zip(crs, params)):
        y = cr + 0.12 if cr < 5 else cr + 0.2
        ax1.text(i, y, f"CR={cr:.2f}\n{p} params", ha="center", va="bottom",
                 fontsize=8, color=C["text"], fontweight="bold")

    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Competitive Ratio (lower = better)")
    ax1.set_title("Controlled Comparison (identical training conditions)")

    # Right: CUT% and segment count
    _ax(ax2)
    cut_pcts = [e.get("cut_pct", 0) for e in entries]
    bar_colors = [C["purple"] if k == "quantum" else C["slate"] for k in kinds]

    bars2 = ax2.bar(range(len(labels)), cut_pcts, color=bar_colors, width=0.6,
                    edgecolor="white", linewidth=0.5)

    for i, (cp, ns) in enumerate(zip(cut_pcts, [e.get("n_segs", 0) for e in entries])):
        ax2.text(i, cp + 0.8, f"{cp:.1f}%\n({ns} segs)", ha="center", va="bottom",
                 fontsize=8, color=C["text"], fontweight="bold")

    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Cut Percentage (%)")
    ax2.set_title("Segmentation Activity")

    legend = [
        Patch(facecolor=C["purple"], label="Quantum (VQ-DQN)"),
        Patch(facecolor=C["slate"], label="Classical Controls"),
    ]
    for a in (ax1, ax2):
        a.legend(handles=legend, loc="upper right",
                 facecolor=C["card"], edgecolor=C["grid"],
                 labelcolor=C["text"], fontsize=7)

    plt.tight_layout()
    fig.savefig(out / "controlled_comparison.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Plot 5: Summary dashboard ────────────────────────────────────────────

def plot_summary_dashboard(cl: dict, qu: dict, out: Path) -> None:
    """Compact summary figure for thesis / presentation."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("Q-RLSTC vs RLSTCcode — Results Dashboard",
                 color=C["text"], fontsize=14, fontweight="bold", y=0.98)

    # Panel A: Classical data-size
    ax = axes[0, 0]
    _ax(ax)
    if "data_size_experiment" in cl:
        entries = cl["data_size_experiment"]
        xs = [e["trajectories"] for e in entries]
        ys = [e["val_cr"] for e in entries]
        best_i = ys.index(min(ys))
        cols = [C["amber"] if i == best_i else C["blue"] for i in range(len(ys))]
        ax.bar([str(x) for x in xs], ys, color=cols, width=0.6, edgecolor="white", linewidth=0.5)
        for i, y in enumerate(ys):
            ax.text(i, y + 0.008, f"{y:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color=C["text"])
    ax.set_title("A) RLSTCcode: Data-Size Ablation", fontsize=10)
    ax.set_ylabel("CR")

    # Panel B: Quantum E1 comparison
    ax = axes[0, 1]
    _ax(ax)
    if "E1" in qu:
        entries = qu["E1"]
        labels = [e["model"].replace(" ", "\n") for e in entries]
        crs = [e["val_cr"] for e in entries]
        cols = [C["purple"] if e["kind"] == "quantum" else C["slate"] for e in entries]
        ax.bar(range(len(labels)), crs, color=cols, width=0.6, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7)
        for i, cr in enumerate(crs):
            y = cr + 0.15 if cr < 5 else cr + 0.2
            ax.text(i, y, f"{cr:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=C["text"])
    ax.set_title("B) Q-RLSTC: Controlled Experiment (30 trajs)", fontsize=10)
    ax.set_ylabel("CR")

    # Panel C: Head-to-head
    ax = axes[1, 0]
    _ax(ax)
    head_to_head = []
    if "data_size_experiment" in cl:
        best_ds = min(cl["data_size_experiment"], key=lambda x: x["val_cr"])
        head_to_head.append(("RLSTCcode\n(best)", best_ds["val_cr"], C["blue"]))
    if "kfold_cv" in cl:
        head_to_head.append(("RLSTCcode\n(5-fold CV)", cl["kfold_cv"]["mean_cr"], C["blue"]))
    if "E1" in qu:
        for e in qu["E1"]:
            color = C["purple"] if e["kind"] == "quantum" else C["slate"]
            head_to_head.append((e["model"].replace(" ", "\n"), e["val_cr"], color))

    head_to_head.sort(key=lambda x: x[1])
    ax.barh(range(len(head_to_head)),
            [h[1] for h in head_to_head],
            color=[h[2] for h in head_to_head],
            height=0.6, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(head_to_head)))
    ax.set_yticklabels([h[0] for h in head_to_head], fontsize=7.5)
    for i, h in enumerate(head_to_head):
        ax.text(h[1] + 0.05, i, f"{h[1]:.4f}", va="center", fontsize=7.5,
                color=C["text"], fontweight="bold")
    ax.axvline(1.0, color=C["amber"], linestyle="-.", alpha=0.5)
    ax.set_xlabel("CR (lower = better)")
    ax.set_title("C) All Models Ranked", fontsize=10)

    # Panel D: Parameter efficiency scatter
    ax = axes[1, 1]
    _ax(ax)
    points = []
    if "data_size_experiment" in cl:
        best_ds = min(cl["data_size_experiment"], key=lambda x: x["val_cr"])
        points.append((514, best_ds["val_cr"], C["blue"], "s", "RLSTCcode\n(best)"))
    if "kfold_cv" in cl:
        points.append((514, cl["kfold_cv"]["mean_cr"], C["cyan"], "s", "RLSTCcode\n(CV)"))
    if "E1" in qu:
        for e in qu["E1"]:
            color = C["purple"] if e["kind"] == "quantum" else C["slate"]
            marker = "D" if e["kind"] == "quantum" else "o"
            points.append((e["params"], e["val_cr"], color, marker, e["model"]))

    for p, cr, color, marker, label in points:
        ax.scatter(p, cr, s=100, color=color, marker=marker, zorder=5,
                   edgecolors="white", linewidth=1)
        # Smart label positioning
        dx, dy = 12, 0
        if cr > 5:
            dy = -0.5
        elif p > 1000:
            dx = -80
        ax.annotate(f"{label}\n{cr:.2f}", (p, cr),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=6.5, color=C["text"])

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log)")
    ax.set_ylabel("CR")
    ax.set_title("D) Parameter Efficiency", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "summary_dashboard.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Markdown report ───────────────────────────────────────────────────────

def build_report(cl: dict, qu: dict, rows: list[dict]) -> str:
    lines: list[str] = []
    w = lines.append

    w("# Direct Head-to-Head: RLSTCcode vs Q-RLSTC")
    w("")
    w(f"Generated: {datetime.now().isoformat()}")
    w("")

    # ── Summary table ──
    w("## All Models Ranked by Competitive Ratio")
    w("")
    w("| Rank | Model | System | Params | Val CR | Optimizer | Training Data |")
    w("|---|---|---|---|---|---|---|")

    ranked = [r for r in rows if r["detail"] in ("best data-size", "5-fold CV mean", "E1", "other")]
    ranked.sort(key=lambda r: r["val_cr"])
    for i, r in enumerate(ranked):
        data = r["training_data"]
        data_str = f"{data:,}" if isinstance(data, int) else str(data)
        std_str = f" ± {r['val_cr_std']:.4f}" if "val_cr_std" in r else ""
        w(f"| {i+1} | {r['label']} | {r['system']} | {r['params']} | "
          f"**{r['val_cr']:.4f}**{std_str} | {r['optimizer']} | {data_str} trajs |")
    w("")

    # ── Key comparisons ──
    w("## Key Comparisons")
    w("")

    # 1. Same architecture
    rlstc_514 = [r for r in rows if r["system"] == "RLSTCcode" and r["detail"] == "best data-size"]
    ctrl_b = [r for r in rows if "Control B" in r.get("label", "")]
    if rlstc_514 and ctrl_b:
        w("### 1. Same Architecture (Dense 5→64→2, 514 params)")
        w("")
        w("| Model | Val CR | Optimizer | Data | Difference |")
        w("|---|---|---|---|---|")
        r = rlstc_514[0]
        c = ctrl_b[0]
        ratio = c["val_cr"] / r["val_cr"]
        w(f"| RLSTCcode | **{r['val_cr']:.4f}** | SGD | {r['training_data']:,} | — |")
        w(f"| Q-RLSTC Control B | **{c['val_cr']:.4f}** | SPSA | 30 | {ratio:.1f}× worse |")
        w("")
        w("> The 514-param MLP works well with SGD+backprop (3k data) but fails ")
        w("> under SPSA (30 data). The optimizer and data, not the architecture, ")
        w("> determine performance.")
        w("")

    # 2. Quantum vs best classical control
    vqdqn = [r for r in rows if r["kind"] == "quantum"]
    ctrl_c = [r for r in rows if "Control C" in r.get("label", "")]
    if vqdqn and ctrl_c:
        w("### 2. Quantum vs Best Classical Control (same SPSA, same 30 trajs)")
        w("")
        w("| Model | Params | Val CR | CUT% | Advantage |")
        w("|---|---|---|---|---|")
        q = vqdqn[0]
        c = ctrl_c[0]
        w(f"| VQ-DQN | **{q['params']}** | **{q['val_cr']:.4f}** | {q.get('cut_pct', 0):.1f}% | ★ best under SPSA |")
        w(f"| Control C | {c['params']} | {c['val_cr']:.4f} | {c.get('cut_pct', 0):.1f}% | {c['params']/q['params']:.0f}× more params |")
        diff = ((c["val_cr"] - q["val_cr"]) / c["val_cr"]) * 100
        w("")
        w(f"> VQ-DQN achieves **{diff:.1f}% lower CR** with **{c['params']/q['params']:.0f}× fewer parameters**.")
        w(f"> This demonstrates quantum parameter efficiency under gradient-free optimization.")
        w("")

    # 3. Absolute best vs quantum
    if rlstc_514 and vqdqn:
        w("### 3. Absolute Best Classical vs Quantum")
        w("")
        r = rlstc_514[0]
        q = vqdqn[0]
        w("| Dimension | RLSTCcode (best) | VQ-DQN |")
        w("|---|---|---|")
        w(f"| Val CR | **{r['val_cr']:.4f}** | {q['val_cr']:.4f} |")
        w(f"| Parameters | {r['params']} | **{q['params']}** |")
        w(f"| Training data | {r['training_data']:,} trajs | {q['training_data']} trajs |")
        w(f"| Optimizer | SGD (gradient) | SPSA (gradient-free) |")
        w(f"| CR ratio | 1.0× | {q['val_cr']/r['val_cr']:.1f}× |")
        w(f"| Data ratio | 1.0× | {r['training_data']/q['training_data']:.0f}× less |")
        w(f"| Param ratio | 1.0× | {r['params']/q['params']:.0f}× fewer |")
        w("")
        w(f"> RLSTCcode wins on absolute CR ({r['val_cr']:.4f} vs {q['val_cr']:.4f}) thanks to ")
        w(f"> {r['training_data']/q['training_data']:.0f}× more training data and gradient-based optimization. ")
        w(f"> VQ-DQN wins on parameter efficiency ({q['params']} vs {r['params']} params).")
        w("")

    # ── Conclusion ──
    w("## Conclusions")
    w("")
    w("1. **RLSTCcode achieves superior absolute clustering quality** (CR 0.59–0.82) due to ")
    w("   100× more training data and SGD with backpropagation gradients.")
    w("")
    w("2. **Under identical SPSA training conditions**, the 34-parameter VQ-DQN outperforms ")
    w("   all classical controls including one with 38× more parameters (1,314 → 1.64 CR vs 34 → 1.48 CR).")
    w("")
    w("3. **The quantum advantage is optimizer-specific**: classical networks degrade catastrophically ")
    w("   under SPSA (CR 5.11–9.21 for small/medium networks), while the quantum circuit structure ")
    w("   provides sufficient inductive bias to learn effective policies.")
    w("")
    w("4. **The data gap is the primary confound**: a fair same-data-same-optimizer comparison ")
    w("   shows quantum winning. A fair same-data-same-optimizer-same-epochs comparison at scale ")
    w("   remains an open experiment.")
    w("")

    w("## Plots")
    w("")
    w("![Unified Comparison](plots/unified_comparison.png)")
    w("")
    w("![Data Scaling](plots/data_scaling.png)")
    w("")
    w("![Controlled Comparison](plots/controlled_comparison.png)")
    w("")
    w("![Parameter Efficiency](plots/parameter_efficiency.png)")
    w("")
    w("![Summary Dashboard](plots/summary_dashboard.png)")
    w("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Direct comparison: RLSTCcode vs Q-RLSTC")
    parser.add_argument("--classical", required=True, help="classical_results.json path")
    parser.add_argument("--quantum", required=True, help="Q-RLSTC thesis_results JSON path")
    parser.add_argument("--output", default="results/comparison", help="output dir")
    args = parser.parse_args()

    cl, qu = load_results(args.classical, args.quantum)
    rows = build_comparison_rows(cl, qu)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    plots = out / "plots"
    plots.mkdir(exist_ok=True)

    # JSON
    json_path = out / "comparison.json"
    with open(json_path, "w") as f:
        serialisable = [
            {k: v for k, v in r.items() if not callable(v)}
            for r in rows
        ]
        json.dump({"timestamp": datetime.now().isoformat(), "models": serialisable}, f, indent=2)
    print(f"✓ {json_path}")

    # Plots
    print("Generating plots...")
    plot_unified_bars(rows, plots)
    print("  ✓ unified_comparison.png")
    plot_param_efficiency(rows, plots)
    print("  ✓ parameter_efficiency.png")
    plot_data_scaling(rows, plots)
    print("  ✓ data_scaling.png")
    plot_controlled_comparison(qu, plots)
    print("  ✓ controlled_comparison.png")
    plot_summary_dashboard(cl, qu, plots)
    print("  ✓ summary_dashboard.png")

    # Markdown
    md = build_report(cl, qu, rows)
    md_path = out / "comparison_report.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"✓ {md_path}")

    print(f"\nDone → {out}")


if __name__ == "__main__":
    main()
