#!/usr/bin/env python3
"""Generate Q-RLSTC-format JSON / Markdown / plots from RLSTCcode savemodels.

Scans a savemodels directory where h5 filenames encode the best validation
competitive ratio, then produces output matching the thesis experiment report
format used by Q-RLSTC.

Usage:
    python experiments/generate_classical_report.py \
        --savemodels /path/to/savemodels \
        --quantum-results results/thesis_qfix/thesis_results_20260224_121334.json \
        --output results/classical_baseline
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── Scanning ──────────────────────────────────────────────────────────────

def scan_savemodels(root: str) -> dict:
    """Walk *root* and extract competitive ratios from h5 filenames.

    Returns a dict of {folder_name: best_cr}.
    """
    results: dict[str, float] = {}
    root_path = Path(root)
    for subdir in sorted(root_path.iterdir()):
        if not subdir.is_dir():
            continue
        for h5 in subdir.glob("*.h5"):
            m = re.search(r"sub-RL-([\d.]+)\.h5", h5.name)
            if m:
                cr = float(m.group(1))
                name = subdir.name
                if name not in results or cr < results[name]:
                    results[name] = cr
    return results


def classify_experiments(results: dict) -> dict:
    """Group the scanned results into experiment categories."""
    categories: dict[str, dict] = {
        "data_size": {},
        "kfold_cv": {},
        "cluster_count": {},
        "other": {},
    }
    for name, cr in results.items():
        if re.match(r"^\d+kmodels$", name):
            k = int(re.match(r"^(\d+)k", name).group(1))
            categories["data_size"][f"{k}k trajectories"] = {
                "folder": name,
                "val_cr": round(cr, 10),
                "trajectories": k * 1000,
            }
        elif re.match(r"^kfoldmodels\d+$", name):
            fold = int(re.search(r"(\d+)$", name).group(1))
            categories["kfold_cv"][f"fold_{fold}"] = {
                "folder": name,
                "val_cr": round(cr, 10),
                "fold": fold,
            }
        elif re.match(r"^modelsk\d+$", name):
            k = int(re.search(r"(\d+)$", name).group(1))
            categories["cluster_count"][f"k={k}"] = {
                "folder": name,
                "val_cr": round(cr, 10),
                "k": k,
            }
        else:
            categories["other"][name] = {
                "folder": name,
                "val_cr": round(cr, 10),
            }
    return categories


# ── JSON generation ───────────────────────────────────────────────────────

def build_json(categories: dict, savemodels_path: str) -> dict:
    """Build a JSON structure mirroring Q-RLSTC thesis results."""
    timestamp = datetime.now().isoformat()

    doc = {
        "protocol": {
            "source": "RLSTCcode (classical baseline)",
            "architecture": "DQN Dense(5→64→2)",
            "params": 514,
            "optimizer": "SGD (lr=0.001)",
            "gamma": 0.99,
            "target_update": "soft (τ=0.05)",
            "double_dqn": False,
            "batch_size": 32,
            "memory_size": 5000,
            "epsilon_start": 1.0,
            "epsilon_min": 0.1,
            "epsilon_decay": 0.99,
        },
        "env": {
            "timestamp": timestamp,
            "savemodels_path": savemodels_path,
        },
    }

    # Data-size experiment
    if categories["data_size"]:
        entries = []
        for label, info in sorted(
            categories["data_size"].items(),
            key=lambda x: x[1]["trajectories"],
        ):
            entries.append({
                "label": label,
                "folder": info["folder"],
                "trajectories": info["trajectories"],
                "val_cr": info["val_cr"],
            })
        doc["data_size_experiment"] = entries

    # K-fold cross-validation
    if categories["kfold_cv"]:
        folds = []
        crs = []
        for label, info in sorted(
            categories["kfold_cv"].items(),
            key=lambda x: x[1]["fold"],
        ):
            folds.append({
                "fold": info["fold"],
                "folder": info["folder"],
                "val_cr": info["val_cr"],
            })
            crs.append(info["val_cr"])
        doc["kfold_cv"] = {
            "k": len(folds),
            "folds": folds,
            "mean_cr": round(float(np.mean(crs)), 10),
            "std_cr": round(float(np.std(crs)), 10),
        }

    # Cluster-count experiment
    if categories["cluster_count"]:
        entries = []
        for label, info in sorted(
            categories["cluster_count"].items(),
            key=lambda x: x[1]["k"],
        ):
            entries.append({
                "label": label,
                "k": info["k"],
                "folder": info["folder"],
                "val_cr": info["val_cr"],
            })
        doc["cluster_count_experiment"] = entries

    # Other
    if categories["other"]:
        entries = []
        for label, info in categories["other"].items():
            entries.append({
                "label": label,
                "folder": info["folder"],
                "val_cr": info["val_cr"],
            })
        doc["other_models"] = entries

    return doc


# ── Markdown report generation ────────────────────────────────────────────

def build_markdown(doc: dict, quantum_results: dict | None) -> str:
    """Generate a markdown report from the JSON structure."""
    lines: list[str] = []
    w = lines.append

    w("# Classical RLSTCcode — Experiment Report")
    w("")
    w(f"Generated: {doc['env']['timestamp']}")
    w("")
    w("## Protocol")
    w("")
    w("```json")
    w(json.dumps(doc["protocol"], indent=2))
    w("```")
    w("")

    # ── Data-size ──
    if "data_size_experiment" in doc:
        w("## Data-Size Experiment (T-Drive)")
        w("")
        w("| Trajectories | Best Val CR | Folder |")
        w("|---|---|---|")
        best = min(doc["data_size_experiment"], key=lambda x: x["val_cr"])
        for e in doc["data_size_experiment"]:
            marker = " ★" if e is best else ""
            w(f"| {e['trajectories']:,} | **{e['val_cr']:.4f}**{marker} | `{e['folder']}` |")
        w("")

    # ── K-fold ──
    if "kfold_cv" in doc:
        cv = doc["kfold_cv"]
        w("## 5-Fold Cross-Validation")
        w("")
        w("| Fold | Val CR | Folder |")
        w("|---|---|---|")
        best_fold = min(cv["folds"], key=lambda x: x["val_cr"])
        for f in cv["folds"]:
            marker = " ★" if f is best_fold else ""
            w(f"| {f['fold']} | **{f['val_cr']:.4f}**{marker} | `{f['folder']}` |")
        w(f"| **Mean ± Std** | **{cv['mean_cr']:.4f} ± {cv['std_cr']:.4f}** | |")
        w("")

    # ── Cluster-count ──
    if "cluster_count_experiment" in doc:
        w("## Cluster-Count (k) Experiment")
        w("")
        w("| k | Val CR | Folder |")
        w("|---|---|---|")
        best_k = min(doc["cluster_count_experiment"], key=lambda x: x["val_cr"])
        for e in doc["cluster_count_experiment"]:
            marker = " ★" if e is best_k else ""
            w(f"| {e['k']} | **{e['val_cr']:.4f}**{marker} | `{e['folder']}` |")
        w("")

    # ── Other ──
    if "other_models" in doc:
        w("## Other Models")
        w("")
        w("| Label | Val CR | Folder |")
        w("|---|---|---|")
        for e in doc["other_models"]:
            w(f"| {e['label']} | **{e['val_cr']:.4f}** | `{e['folder']}` |")
        w("")

    # ── Cross-system comparison ──
    if quantum_results:
        w("## Cross-System Comparison")
        w("")
        w("> **Note**: Training conditions differ significantly (see bottom).")
        w("")
        w("| System | Model | Params | Val CR | Optimizer | Training Data |")
        w("|---|---|---|---|---|---|")

        # Classical rows
        if "data_size_experiment" in doc:
            best_ds = min(doc["data_size_experiment"], key=lambda x: x["val_cr"])
            w(f"| RLSTCcode | DQN (5→64→2) | 514 | **{best_ds['val_cr']:.4f}** | SGD | {best_ds['trajectories']:,} trajs |")
        if "kfold_cv" in doc:
            cv = doc["kfold_cv"]
            w(f"| RLSTCcode | DQN (5-fold CV) | 514 | **{cv['mean_cr']:.4f}** | SGD | ~4,000 trajs |")

        # Quantum rows
        if "E1" in quantum_results:
            for entry in quantum_results["E1"]:
                model = entry["model"]
                params = entry["params"]
                cr = entry["val_cr"]
                kind = entry["kind"]
                prefix = "Q-RLSTC" if kind == "quantum" else "Q-RLSTC ctrl"
                w(f"| {prefix} | {model} | {params} | **{cr:.4f}** | SPSA | 30 trajs |")
        w("")

        w("### Conditions Caveat")
        w("")
        w("| Dimension | RLSTCcode | Q-RLSTC |")
        w("|---|---|---|")
        w("| Training trajectories | 500–5,000 | 30 |")
        w("| Epochs / Rounds | 2 full rounds | 2 epochs |")
        w("| Optimizer | SGD (backprop) | SPSA (gradient-free) |")
        w("| Target update | Soft (τ=0.05) | Hard copy (every 10 eps) |")
        w("| Double DQN | No | Yes |")
        w("| γ | 0.99 | 0.9 |")
        w("| Reward shaping | ΔOD raw | ΔOD + cut penalty + extend cost |")
        w("")

    w("## Plots")
    w("")
    w("![Data-Size CR](plots/data_size_cr.png)")
    w("")
    w("![K-Fold CV](plots/kfold_cv.png)")
    w("")
    w("![Cluster-Count CR](plots/cluster_count_cr.png)")
    w("")
    if quantum_results:
        w("![Cross-System Comparison](plots/cross_system_comparison.png)")
        w("")
        w("![Parameter Efficiency](plots/parameter_efficiency.png)")
        w("")

    w("## Raw Results (JSON)")
    w("")
    w("See `classical_results.json` for machine-readable data.")
    w("")

    return "\n".join(lines)


# ── Plot generation ───────────────────────────────────────────────────────

COLORS = {
    "classical": "#3B82F6",
    "quantum": "#8B5CF6",
    "control": "#94A3B8",
    "highlight": "#F59E0B",
    "bg": "#0F172A",
    "card": "#1E293B",
    "text": "#E2E8F0",
    "grid": "#334155",
}


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(COLORS["card"])
    ax.tick_params(colors=COLORS["text"], labelsize=9)
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.title.set_color(COLORS["text"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
    ax.grid(axis="y", color=COLORS["grid"], alpha=0.4, linewidth=0.5)


def plot_data_size(doc: dict, out_dir: Path) -> None:
    entries = doc.get("data_size_experiment", [])
    if not entries:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(COLORS["bg"])
    _style_ax(ax)

    xs = [e["trajectories"] for e in entries]
    ys = [e["val_cr"] for e in entries]
    best_idx = ys.index(min(ys))
    colors = [COLORS["highlight"] if i == best_idx else COLORS["classical"] for i in range(len(ys))]

    ax.bar([str(x) for x in xs], ys, color=colors, width=0.6, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Training Trajectories")
    ax.set_ylabel("Competitive Ratio (lower = better)")
    ax.set_title("RLSTCcode — Data Size vs. Clustering Quality")

    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.text(i, y + 0.01, f"{y:.4f}", ha="center", va="bottom",
                fontsize=8, color=COLORS["text"], fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_dir / "data_size_cr.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_kfold(doc: dict, out_dir: Path) -> None:
    cv = doc.get("kfold_cv")
    if not cv:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(COLORS["bg"])
    _style_ax(ax)

    folds = cv["folds"]
    xs = [f"Fold {f['fold']}" for f in folds]
    ys = [f["val_cr"] for f in folds]
    mean_cr = cv["mean_cr"]
    best_idx = ys.index(min(ys))
    colors = [COLORS["highlight"] if i == best_idx else COLORS["classical"] for i in range(len(ys))]

    ax.bar(xs, ys, color=colors, width=0.6, edgecolor="white", linewidth=0.5)
    ax.axhline(mean_cr, color=COLORS["text"], linestyle="--", linewidth=1, alpha=0.7,
               label=f"Mean = {mean_cr:.4f}")

    for i, y in enumerate(ys):
        ax.text(i, y + 0.003, f"{y:.4f}", ha="center", va="bottom",
                fontsize=8, color=COLORS["text"], fontweight="bold")

    ax.set_ylabel("Competitive Ratio (lower = better)")
    ax.set_title("RLSTCcode — 5-Fold Cross-Validation")
    ax.legend(facecolor=COLORS["card"], edgecolor=COLORS["grid"],
              labelcolor=COLORS["text"], fontsize=9)

    plt.tight_layout()
    fig.savefig(out_dir / "kfold_cv.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_cluster_count(doc: dict, out_dir: Path) -> None:
    entries = doc.get("cluster_count_experiment", [])
    if not entries:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(COLORS["bg"])
    _style_ax(ax)

    xs = [f"k={e['k']}" for e in entries]
    ys = [e["val_cr"] for e in entries]
    best_idx = ys.index(min(ys))
    colors = [COLORS["highlight"] if i == best_idx else COLORS["classical"] for i in range(len(ys))]

    ax.bar(xs, ys, color=colors, width=0.6, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Competitive Ratio (lower = better)")
    ax.set_title("RLSTCcode — Cluster Count (k) vs. Clustering Quality")

    for i, y in enumerate(ys):
        ax.text(i, y + 0.005, f"{y:.4f}", ha="center", va="bottom",
                fontsize=8, color=COLORS["text"], fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_dir / "cluster_count_cr.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_cross_system(doc: dict, quantum_results: dict, out_dir: Path) -> None:
    """Bar chart comparing all systems side by side."""
    if not quantum_results or "E1" not in quantum_results:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(COLORS["bg"])
    _style_ax(ax)

    labels, crs, bar_colors = [], [], []

    # Classical best
    if "data_size_experiment" in doc:
        best_ds = min(doc["data_size_experiment"], key=lambda x: x["val_cr"])
        labels.append(f"RLSTCcode\n({best_ds['trajectories']//1000}k trajs)")
        crs.append(best_ds["val_cr"])
        bar_colors.append(COLORS["classical"])

    if "kfold_cv" in doc:
        labels.append(f"RLSTCcode\n(5-fold CV)")
        crs.append(doc["kfold_cv"]["mean_cr"])
        bar_colors.append(COLORS["classical"])

    # Quantum + controls
    for entry in quantum_results["E1"]:
        model = entry["model"]
        kind = entry["kind"]
        labels.append(f"{model}\n({entry['params']} params)")
        crs.append(entry["val_cr"])
        bar_colors.append(COLORS["quantum"] if kind == "quantum" else COLORS["control"])

    xs = range(len(labels))
    bars = ax.bar(xs, crs, color=bar_colors, width=0.65, edgecolor="white", linewidth=0.5)

    for i, (bar, cr) in enumerate(zip(bars, crs)):
        y_pos = min(cr + 0.15, ax.get_ylim()[1] * 0.95) if cr < 5 else cr + 0.15
        ax.text(i, y_pos, f"{cr:.2f}", ha="center", va="bottom",
                fontsize=8, color=COLORS["text"], fontweight="bold")

    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Competitive Ratio (lower = better)")
    ax.set_title("Cross-System Comparison: RLSTCcode vs. Q-RLSTC")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["classical"], label="Classical (RLSTCcode)"),
        Patch(facecolor=COLORS["quantum"], label="Quantum (VQ-DQN)"),
        Patch(facecolor=COLORS["control"], label="Classical Controls (Q-RLSTC)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              facecolor=COLORS["card"], edgecolor=COLORS["grid"],
              labelcolor=COLORS["text"], fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "cross_system_comparison.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_param_efficiency(doc: dict, quantum_results: dict, out_dir: Path) -> None:
    """Scatter plot: params vs CR, showing parameter efficiency."""
    if not quantum_results or "E1" not in quantum_results:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(COLORS["bg"])
    _style_ax(ax)

    # Classical RLSTC (514 params)
    classical_crs = []
    if "data_size_experiment" in doc:
        best_ds = min(doc["data_size_experiment"], key=lambda x: x["val_cr"])
        classical_crs.append(best_ds["val_cr"])
    if "kfold_cv" in doc:
        classical_crs.append(doc["kfold_cv"]["mean_cr"])

    for cr in classical_crs:
        ax.scatter(514, cr, s=120, color=COLORS["classical"], zorder=5,
                   edgecolors="white", linewidth=1)
        ax.annotate(f"RLSTCcode\nCR={cr:.2f}", (514, cr),
                    textcoords="offset points", xytext=(12, -5),
                    fontsize=7.5, color=COLORS["text"])

    # Quantum + controls
    for entry in quantum_results["E1"]:
        params = entry["params"]
        cr = entry["val_cr"]
        kind = entry["kind"]
        color = COLORS["quantum"] if kind == "quantum" else COLORS["control"]
        marker = "D" if kind == "quantum" else "o"
        ax.scatter(params, cr, s=120 if kind == "quantum" else 80,
                   color=color, zorder=5, marker=marker,
                   edgecolors="white", linewidth=1)
        offset = (12, -5) if params > 100 else (12, 5)
        ax.annotate(f"{entry['model']}\nCR={cr:.2f}", (params, cr),
                    textcoords="offset points", xytext=offset,
                    fontsize=7, color=COLORS["text"])

    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters (log scale)")
    ax.set_ylabel("Competitive Ratio (lower = better)")
    ax.set_title("Parameter Efficiency: Params vs. Clustering Quality")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    plt.tight_layout()
    fig.savefig(out_dir / "parameter_efficiency.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Q-RLSTC-format reports from RLSTCcode savemodels"
    )
    parser.add_argument(
        "--savemodels", required=True,
        help="Path to savemodels directory with .h5 files",
    )
    parser.add_argument(
        "--quantum-results", default=None,
        help="Path to Q-RLSTC thesis_results JSON for cross-comparison",
    )
    parser.add_argument(
        "--output", default="results/classical_baseline",
        help="Output directory for generated reports",
    )
    args = parser.parse_args()

    # Scan
    print(f"Scanning {args.savemodels} ...")
    raw = scan_savemodels(args.savemodels)
    print(f"  Found {len(raw)} model directories")
    for name, cr in sorted(raw.items()):
        print(f"    {name}: CR = {cr:.4f}")

    categories = classify_experiments(raw)

    # Build JSON
    doc = build_json(categories, args.savemodels)

    # Load quantum results if provided
    quantum_results = None
    if args.quantum_results and os.path.exists(args.quantum_results):
        print(f"\nLoading quantum results from {args.quantum_results}")
        with open(args.quantum_results) as f:
            quantum_results = json.load(f)

    # Output directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Write JSON
    json_path = out_dir / "classical_results.json"
    with open(json_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"\n✓ {json_path}")

    # Write Markdown
    md = build_markdown(doc, quantum_results)
    md_path = out_dir / "classical_report.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"✓ {md_path}")

    # Generate plots
    print(f"\nGenerating plots → {plots_dir}")
    plot_data_size(doc, plots_dir)
    print("  ✓ data_size_cr.png")
    plot_kfold(doc, plots_dir)
    print("  ✓ kfold_cv.png")
    plot_cluster_count(doc, plots_dir)
    print("  ✓ cluster_count_cr.png")

    if quantum_results:
        plot_cross_system(doc, quantum_results, plots_dir)
        print("  ✓ cross_system_comparison.png")
        plot_param_efficiency(doc, quantum_results, plots_dir)
        print("  ✓ parameter_efficiency.png")

    print(f"\nDone. All outputs in {out_dir}")


if __name__ == "__main__":
    main()
