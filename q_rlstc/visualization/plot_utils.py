"""
Publication-quality plotting utilities for Q-RLSTC experiments.

Adapted from patterns in TheFinalQRLSTC/visualization/plot_utils.py,
New_QRLSTC/QRLSTCcode-theoretical/enhanced_plots.py, and
QRLSTC/plot_utils.py.

All plots are 150 DPI, bbox_inches='tight', with consistent style.
"""

import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

try:
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False


# ── Publication-quality style config ──────────────────────────────────
STYLE_CONFIG = {
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "lines.linewidth": 2,
    "lines.markersize": 8,
}

# Colour palette
COLORS = {
    "version_a": "#4363d8",   # blue
    "version_b": "#e6194B",   # red
    "ideal": "#3cb44b",       # green
    "eagle": "#f58231",       # orange
    "heron": "#911eb4",       # purple
    "classical": "#aaaaaa",   # grey
}


def _require_mpl():
    if not MPL_AVAILABLE:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: uv pip install matplotlib"
        )


def _apply_style():
    """Apply publication-quality rcParams."""
    plt.rcParams.update(STYLE_CONFIG)


def _smooth(values: List[float], window: int = 10) -> np.ndarray:
    """Moving-average smoothing."""
    arr = np.asarray(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def _add_info_box(ax: "plt.Axes", text: str, loc: str = "upper right"):
    """Add a semi-transparent info box to the axes."""
    props = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85,
                 edgecolor="#cccccc")
    anchors = {
        "upper right": (0.98, 0.98, "right", "top"),
        "upper left": (0.02, 0.98, "left", "top"),
        "lower right": (0.98, 0.02, "right", "bottom"),
    }
    x, y, ha, va = anchors.get(loc, anchors["upper right"])
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8,
            verticalalignment=va, horizontalalignment=ha, bbox=props,
            family="monospace")


# ─────────────────────────────────────────────────────────────────────
# Plot Functions
# ─────────────────────────────────────────────────────────────────────

def plot_learning_curves(
    rewards_a: List[float],
    rewards_b: Optional[List[float]] = None,
    out_path: Union[str, Path] = "learning_curves.png",
    losses_a: Optional[List[float]] = None,
    losses_b: Optional[List[float]] = None,
    smooth_window: int = 10,
    title: str = "Learning Curves",
):
    """Dual-axis learning curve: reward (left) and optional loss (right).

    Supports overlay of Version A vs B.
    """
    _require_mpl()
    _apply_style()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # --- Rewards (left axis) ---
    episodes_a = list(range(1, len(rewards_a) + 1))
    ax1.plot(episodes_a, rewards_a, alpha=0.25, color=COLORS["version_a"])
    if len(rewards_a) >= smooth_window:
        smoothed_a = _smooth(rewards_a, smooth_window)
        x_sm = list(range(smooth_window, len(rewards_a) + 1))
        ax1.plot(x_sm, smoothed_a, color=COLORS["version_a"],
                 label=f"Version A (smoothed, w={smooth_window})")
    else:
        ax1.plot(episodes_a, rewards_a, color=COLORS["version_a"],
                 label="Version A")

    if rewards_b is not None:
        episodes_b = list(range(1, len(rewards_b) + 1))
        ax1.plot(episodes_b, rewards_b, alpha=0.25, color=COLORS["version_b"])
        if len(rewards_b) >= smooth_window:
            smoothed_b = _smooth(rewards_b, smooth_window)
            x_sm = list(range(smooth_window, len(rewards_b) + 1))
            ax1.plot(x_sm, smoothed_b, color=COLORS["version_b"],
                     label=f"Version B (smoothed, w={smooth_window})")
        else:
            ax1.plot(episodes_b, rewards_b, color=COLORS["version_b"],
                     label="Version B")

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode Reward")
    ax1.legend(loc="upper left")

    # --- Losses (right axis) ---
    if losses_a is not None:
        ax2 = ax1.twinx()
        ax2.plot(range(1, len(losses_a) + 1), losses_a,
                 alpha=0.4, color=COLORS["version_a"], linestyle="--",
                 label="Loss A")
        if losses_b is not None:
            ax2.plot(range(1, len(losses_b) + 1), losses_b,
                     alpha=0.4, color=COLORS["version_b"], linestyle="--",
                     label="Loss B")
        ax2.set_ylabel("TD Loss")
        ax2.legend(loc="upper right")

    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_od_convergence(
    od_a: List[float],
    od_b: Optional[List[float]] = None,
    out_path: Union[str, Path] = "od_convergence.png",
    title: str = "Overall Distance Convergence",
):
    """Plot OD vs epoch for Version A (and optionally B)."""
    _require_mpl()
    _apply_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    epochs_a = list(range(1, len(od_a) + 1))
    ax.plot(epochs_a, od_a, "o-", color=COLORS["version_a"],
            label="Version A (5q)")
    if len(od_a) > 1:
        delta = od_a[0] - od_a[-1]
        pct = (delta / od_a[0] * 100) if od_a[0] != 0 else 0
        _add_info_box(ax, f"ΔOD_A = {delta:.4f} ({pct:.1f}%)")

    if od_b is not None:
        epochs_b = list(range(1, len(od_b) + 1))
        ax.plot(epochs_b, od_b, "s-", color=COLORS["version_b"],
                label="Version B (8q)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Overall Distance (lower is better)")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(
    metrics: Dict[str, Dict[str, float]],
    out_path: Union[str, Path] = "metric_comparison.png",
    title: str = "Version A vs B — Key Metrics",
):
    """Grouped bar chart comparing multiple metrics across versions.

    Args:
        metrics: {"Version A": {"F1": 0.8, "ΔOD": 2.1, ...}, "Version B": {...}}
    """
    _require_mpl()
    _apply_style()

    versions = list(metrics.keys())
    metric_names = list(metrics[versions[0]].keys())
    n_metrics = len(metric_names)
    n_versions = len(versions)

    x = np.arange(n_metrics)
    width = 0.35

    colors = [COLORS["version_a"], COLORS["version_b"]] + \
             [COLORS.get("classical", "#999")] * 5

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, version in enumerate(versions):
        vals = [metrics[version].get(m, 0) for m in metric_names]
        offset = (i - (n_versions - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=version,
                      color=colors[i % len(colors)], alpha=0.85)
        # Value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=15, ha="right")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_noise_impact(
    results: Dict[str, List[float]],
    out_path: Union[str, Path] = "noise_impact.png",
    smooth_window: int = 10,
    title: str = "NISQ Noise Impact on Training",
):
    """Reward curves under different noise models with resilience ratios.

    Args:
        results: {"ideal": [r1, r2, ...], "eagle": [...], "heron": [...]}
    """
    _require_mpl()
    _apply_style()

    color_map = {
        "ideal": COLORS["ideal"],
        "eagle": COLORS["eagle"],
        "heron": COLORS["heron"],
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    ideal_final = None
    info_lines = []

    for name, rewards in results.items():
        color = color_map.get(name, "#999999")
        episodes = list(range(1, len(rewards) + 1))
        ax.plot(episodes, rewards, alpha=0.2, color=color)

        if len(rewards) >= smooth_window:
            smoothed = _smooth(rewards, smooth_window)
            x_sm = list(range(smooth_window, len(rewards) + 1))
            ax.plot(x_sm, smoothed, color=color, label=name.capitalize())
        else:
            ax.plot(episodes, rewards, color=color, label=name.capitalize())

        avg_final = float(np.mean(rewards[-10:])) if len(rewards) >= 10 \
            else float(np.mean(rewards)) if rewards else 0
        if name == "ideal":
            ideal_final = avg_final
        info_lines.append(f"{name:6}: avg_final={avg_final:.4f}")

    # Resilience ratios
    if ideal_final and ideal_final > 0:
        info_lines.append("─" * 28)
        for name, rewards in results.items():
            if name != "ideal":
                avg = float(np.mean(rewards[-10:])) if len(rewards) >= 10 \
                    else float(np.mean(rewards))
                ratio = avg / ideal_final
                status = "✓" if ratio > 0.8 else "⚠"
                info_lines.append(f"{name:6}: R/R_ideal={ratio:.3f} {status}")

    _add_info_box(ax, "\n".join(info_lines))
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_epsilon_schedule(
    n_episodes: int,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.1,
    epsilon_decay: float = 0.99,
    out_path: Union[str, Path] = "epsilon_schedule.png",
    title: str = "Exploration Schedule (ε-Greedy)",
):
    """Visualise epsilon decay over episodes."""
    _require_mpl()
    _apply_style()

    epsilons = []
    eps = epsilon_start
    for _ in range(n_episodes):
        epsilons.append(eps)
        eps = max(epsilon_min, eps * epsilon_decay)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, n_episodes + 1), epsilons, color=COLORS["version_a"])
    ax.axhline(y=epsilon_min, color="red", linestyle="--", alpha=0.5,
               label=f"ε_min = {epsilon_min}")

    # Mark when epsilon hits minimum
    for i, e in enumerate(epsilons):
        if abs(e - epsilon_min) < 1e-6:
            ax.axvline(x=i + 1, color="green", linestyle=":", alpha=0.4)
            _add_info_box(ax, f"ε reaches minimum\nat episode {i + 1}")
            break

    ax.set_xlabel("Episode")
    ax.set_ylabel("ε (exploration rate)")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_timing_breakdown(
    timing_data: Dict[str, float],
    out_path: Union[str, Path] = "timing_breakdown.png",
    title: str = "Runtime Breakdown",
):
    """Stacked horizontal bar chart of timing components.

    Args:
        timing_data: {"Circuit Eval": 12.3, "SPSA Step": 5.1, ...}
    """
    _require_mpl()
    _apply_style()

    labels = list(timing_data.keys())
    values = list(timing_data.values())
    total = sum(values)

    palette = ["#4363d8", "#e6194B", "#3cb44b", "#f58231",
               "#911eb4", "#42d4f4", "#f032e6"]

    fig, ax = plt.subplots(figsize=(10, 4))
    left = 0
    for i, (label, val) in enumerate(zip(labels, values)):
        pct = (val / total * 100) if total > 0 else 0
        bar = ax.barh("Runtime", val, left=left,
                       color=palette[i % len(palette)], alpha=0.85,
                       label=f"{label} ({val:.1f}s, {pct:.0f}%)")
        if val / total > 0.08:  # only label if wide enough
            ax.text(left + val / 2, 0, f"{val:.1f}s",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white")
        left += val

    ax.set_xlabel("Time (seconds)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{title} — Total: {total:.1f}s")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_circuit_summary(
    info_a: Dict[str, Any],
    info_b: Optional[Dict[str, Any]] = None,
    out_path: Union[str, Path] = "circuit_summary.png",
    title: str = "VQ-DQN Circuit Summary",
):
    """Table-figure showing circuit properties for Version A and B.

    Args:
        info_a: {"n_qubits": 5, "n_params": 20, "depth": 11, ...}
        info_b: Same for Version B (optional).
    """
    _require_mpl()
    _apply_style()

    rows = ["Qubits", "Variational Layers", "Parameters", "Circuit Depth",
            "Feature Dimensions", "Readout Mode"]
    keys = ["n_qubits", "n_layers", "n_params", "depth",
            "feature_dim", "readout_mode"]

    col_labels = ["Property", "Version A"]
    data = [[row, str(info_a.get(k, "—"))] for row, k in zip(rows, keys)]

    if info_b is not None:
        col_labels.append("Version B")
        for i, k in enumerate(keys):
            data[i].append(str(info_b.get(k, "—")))

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    table = ax.table(
        cellText=data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4363d8")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row shading
    for i in range(1, len(data) + 1):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[i, j].set_facecolor("#f0f4ff")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# JSON Serialisation
# ─────────────────────────────────────────────────────────────────────

def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_results_json(
    results: Dict[str, Any],
    out_path: Union[str, Path],
):
    """Save benchmark results to JSON with numpy-safe conversion."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(_make_serializable(results), f, indent=2)


# ─────────────────────────────────────────────────────────────────────
# Cluster & Segmentation Plots
# ─────────────────────────────────────────────────────────────────────

def plot_cluster_assignments(
    points: np.ndarray,
    labels: np.ndarray,
    centroids: Optional[np.ndarray] = None,
    out_path: Union[str, Path] = "cluster_assignments.png",
    title: str = "Cluster Assignments",
    version_label: str = "",
):
    """Scatter plot of trajectory points coloured by cluster assignment.

    Args:
        points: (N, 2) array of 2D coordinates (lon, lat or x, y).
        labels: (N,) integer cluster labels.
        centroids: (K, 2) cluster centroids (optional).
        out_path: Output file path.
        title: Plot title.
        version_label: E.g. "Classical Parity (5q)".
    """
    _require_mpl()
    _apply_style()

    unique_labels = np.unique(labels[labels >= 0])  # skip noise (-1)
    n_clusters = len(unique_labels)

    cmap = plt.cm.get_cmap("tab20", max(n_clusters, 2))

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, k in enumerate(unique_labels):
        mask = labels == k
        ax.scatter(
            points[mask, 0], points[mask, 1],
            c=[cmap(i)], s=8, alpha=0.5, label=f"C{k}",
        )

    # Noise points in grey
    noise_mask = labels < 0
    if noise_mask.any():
        ax.scatter(
            points[noise_mask, 0], points[noise_mask, 1],
            c="grey", s=4, alpha=0.3, label="Noise",
        )

    # Centroids
    if centroids is not None:
        ax.scatter(
            centroids[:, 0], centroids[:, 1],
            c="black", marker="X", s=120, edgecolors="white",
            linewidths=1.5, zorder=10, label="Centroids",
        )

    info = f"K = {n_clusters}  |  N = {len(points)}"
    if version_label:
        info = f"{version_label}\n{info}"
    _add_info_box(ax, info)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(
        loc="upper left", fontsize=7, ncol=max(1, n_clusters // 8),
        markerscale=2,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_segmentation_boundaries(
    trajectory: np.ndarray,
    predicted_boundaries: List[int],
    ground_truth_boundaries: Optional[List[int]] = None,
    out_path: Union[str, Path] = "segmentation_boundaries.png",
    title: str = "Segmentation Boundaries",
    version_label: str = "",
):
    """Timeline plot showing where segments are cut.

    Plots the trajectory as a 1-D signal (distance from start or
    cumulative displacement) with vertical lines at predicted boundaries
    and optional ground-truth comparison.

    Args:
        trajectory: (T, 2+) array of trajectory points.
        predicted_boundaries: List of point indices where segments break.
        ground_truth_boundaries: True segment breaks (optional).
        out_path: Output file path.
        title: Plot title.
        version_label: E.g. "Quantum Enhanced (8q)".
    """
    _require_mpl()
    _apply_style()

    # Compute cumulative displacement as 1-D signal
    if trajectory.shape[1] >= 2:
        diffs = np.diff(trajectory[:, :2], axis=0)
        displacements = np.sqrt(np.sum(diffs ** 2, axis=1))
        cum_disp = np.concatenate([[0], np.cumsum(displacements)])
    else:
        cum_disp = np.arange(len(trajectory), dtype=float)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Plot cumulative displacement
    time_idx = np.arange(len(cum_disp))
    ax.plot(time_idx, cum_disp, "-", color=COLORS["version_a"],
            alpha=0.7, label="Cumulative displacement")

    # Ground truth boundaries
    if ground_truth_boundaries:
        for bi, b in enumerate(ground_truth_boundaries):
            ax.axvline(
                x=b, color=COLORS["ideal"], linestyle="--", alpha=0.7,
                label="Ground truth" if bi == 0 else None,
            )

    # Predicted boundaries
    for bi, b in enumerate(predicted_boundaries):
        ax.axvline(
            x=b, color=COLORS["version_b"], linestyle="-", alpha=0.8,
            label="Predicted" if bi == 0 else None,
        )

    # Shade segments alternately
    all_bounds = sorted([0] + list(predicted_boundaries) + [len(trajectory) - 1])
    for i in range(len(all_bounds) - 1):
        if i % 2 == 0:
            ax.axvspan(all_bounds[i], all_bounds[i + 1],
                       alpha=0.06, color=COLORS["version_a"])

    # Info
    n_pred = len(predicted_boundaries)
    n_gt = len(ground_truth_boundaries) if ground_truth_boundaries else 0
    info = f"Predicted: {n_pred} boundaries\nPoints: {len(trajectory)}"
    if n_gt > 0:
        info += f"\nGround truth: {n_gt} boundaries"
    if version_label:
        info = f"{version_label}\n{info}"
    _add_info_box(ax, info)

    ax.set_xlabel("Point Index")
    ax.set_ylabel("Cumulative Displacement")
    ax.legend(loc="upper left")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_backend_comparison(
    backend_timings: Dict[str, Dict[str, float]],
    out_path: Union[str, Path] = "backend_comparison.png",
    title: str = "Compute Backend Performance",
):
    """Grouped bar chart comparing timing across backends.

    Args:
        backend_timings: {"A_ideal": {"cpu": 12.3, "mlx": 8.1}, ...}
            Keys are run names, values map backend → runtime seconds.
        out_path: Output file path.
        title: Plot title.
    """
    _require_mpl()
    _apply_style()

    runs = list(backend_timings.keys())
    backends = sorted(
        {be for timings in backend_timings.values() for be in timings}
    )
    n_runs = len(runs)
    n_be = len(backends)

    backend_colors = {
        "cpu": "#aaaaaa",
    }

    x = np.arange(n_runs)
    width = 0.7 / max(n_be, 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, be in enumerate(backends):
        vals = [backend_timings[r].get(be, 0) for r in runs]
        offset = (i - (n_be - 1) / 2) * width
        bars = ax.bar(
            x + offset, vals, width,
            label=be.upper(),
            color=backend_colors.get(be, "#999"),
            alpha=0.85,
        )
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.1f}s",
                    ha="center", va="bottom", fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=15, ha="right")
    ax.set_ylabel("Runtime (seconds)")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Thesis-Specific Plots
# ─────────────────────────────────────────────────────────────────────

def plot_pareto_frontier(
    results: List[Dict[str, Any]],
    out_path: Union[str, Path] = "pareto_frontier.png",
    title: str = "Efficiency Frontier: ValCR vs CUT%",
):
    """Scatter plot of ValCR vs CUT% with marker size proportional to param count.

    Highlights the trade-off between clustering quality and segmentation
    aggressiveness. Quantum models should cluster toward low-CUT, low-ValCR.

    Args:
        results: List of benchmark result dicts with keys:
            model, val_cr, cut_pct, params.
        out_path: Output file path.
        title: Plot title.
    """
    _require_mpl()
    _apply_style()

    fig, ax = plt.subplots(figsize=(10, 7))

    for r in results:
        is_quantum = "VQ-DQN" in r.get("model", "")
        color = COLORS["version_a"] if is_quantum else COLORS["classical"]
        marker = "D" if is_quantum else "o"
        size = max(20, r.get("params", 30) * 1.5)

        ax.scatter(
            r["cut_pct"], r["val_cr"],
            s=size, c=color, marker=marker, alpha=0.8,
            edgecolors="white", linewidths=0.8, zorder=5,
        )
        ax.annotate(
            r["model"], (r["cut_pct"], r["val_cr"]),
            textcoords="offset points", xytext=(8, 4),
            fontsize=7, alpha=0.85,
        )

    ax.set_xlabel("CUT% (segmentation aggressiveness)")
    ax.set_ylabel("ValCR (lower is better)")
    ax.set_title(title)

    # Legend for marker types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor=COLORS["version_a"],
               markersize=8, label="Quantum (VQ-DQN)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["classical"],
               markersize=8, label="Classical (SPSA MLP)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    _add_info_box(ax, "Marker size ∝ parameter count", loc="lower right")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_shot_sensitivity(
    shot_counts: List[int],
    val_crs: List[float],
    noise_ratios: Optional[List[float]] = None,
    out_path: Union[str, Path] = "shot_sensitivity.png",
    title: str = "E3: Shot Count Sensitivity",
):
    """Line chart of shots vs ValCR and optional NR for E3 analysis.

    Args:
        shot_counts: List of shot budgets (e.g. [128, 256, 512, 1024, 4096]).
        val_crs: ValCR at each shot count.
        noise_ratios: Optional NR values at each shot count.
        out_path: Output file path.
        title: Plot title.
    """
    _require_mpl()
    _apply_style()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(shot_counts, val_crs, "o-", color=COLORS["version_a"],
             linewidth=2, markersize=8, label="ValCR")
    ax1.set_xlabel("Shots per circuit evaluation")
    ax1.set_ylabel("ValCR (lower is better)", color=COLORS["version_a"])
    ax1.set_xscale("log", base=2)
    ax1.tick_params(axis="y", labelcolor=COLORS["version_a"])

    if noise_ratios is not None:
        ax2 = ax1.twinx()
        ax2.plot(shot_counts, noise_ratios, "s--", color=COLORS["eagle"],
                 linewidth=2, markersize=8, label="Noise Ratio")
        ax2.set_ylabel("Noise Ratio (NR)", color=COLORS["eagle"])
        ax2.tick_params(axis="y", labelcolor=COLORS["eagle"])
        ax2.axhline(y=1.0, color="grey", linestyle=":", alpha=0.5)
        ax2.legend(loc="upper right")

    ax1.legend(loc="upper left")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_q_value_evolution(
    q_values_per_epoch: List[Tuple[float, float]],
    out_path: Union[str, Path] = "q_value_evolution.png",
    title: str = "Q-Value Evolution (Stuckness Diagnostic)",
):
    """Per-epoch Q-value evolution for a fixed probe state.

    If Q-values don't change across epochs, the model's policy is stuck.

    Args:
        q_values_per_epoch: List of (Q_extend, Q_cut) tuples per epoch.
        out_path: Output file path.
        title: Plot title.
    """
    _require_mpl()
    _apply_style()

    epochs = list(range(1, len(q_values_per_epoch) + 1))
    q_ext = [q[0] for q in q_values_per_epoch]
    q_cut = [q[1] for q in q_values_per_epoch]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, q_ext, "o-", color=COLORS["version_a"],
            label="Q(EXTEND)", linewidth=2, markersize=8)
    ax.plot(epochs, q_cut, "s-", color=COLORS["version_b"],
            label="Q(CUT)", linewidth=2, markersize=8)

    # Shade "stuck" region if Q-values barely change
    q_ext_range = max(q_ext) - min(q_ext) if q_ext else 0
    q_cut_range = max(q_cut) - min(q_cut) if q_cut else 0
    if max(q_ext_range, q_cut_range) < 0.01:
        ax.axhspan(min(q_ext + q_cut) - 0.1, max(q_ext + q_cut) + 0.1,
                   alpha=0.15, color="red")
        _add_info_box(ax, "⚠ Q-values stuck\n(policy not updating)")
    else:
        delta = abs(q_cut[-1] - q_ext[-1])
        _add_info_box(ax, f"ΔQ(final) = {delta:.4f}\nQ evolving ✓")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Q-Value (fixed probe state)")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Trajectory Cluster Plots (ported from QRLSTC/plot_utils.py &
#   TheFinalQRLSTC/subtrajcluster/visualization/plot_utils.py)
# ─────────────────────────────────────────────────────────────────────

def _get_distinct_colors(n: int) -> List[Tuple[float, ...]]:
    """Generate *n* maximally distinct colours."""
    PALETTE = [
        '#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
        '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
        '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3',
        '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#000000',
    ]
    if n <= len(PALETTE):
        import matplotlib.colors as mcolors
        return [mcolors.to_rgb(c) for c in PALETTE[:n]]
    import colorsys
    colors = []
    golden = 0.618033988749895
    h = 0.0
    for _ in range(n):
        colors.append(colorsys.hsv_to_rgb(h, 0.85, 0.9))
        h = (h + golden) % 1.0
    return colors


def plot_trajectory_clusters(
    cluster_dict: Dict[int, Any],
    out_path: Union[str, Path],
    alpha: float = 0.4,
    center_alpha: float = 0.9,
    point_sample_rate: int = 10,
    trajectory_sample_rate: int = 5,
    method_name: str = "Q-RLSTC",
    show_centers: bool = True,
    show_info: bool = True,
) -> None:
    """Cluster scatter plot using actual trajectory points with centre highlights.

    Args:
        cluster_dict: ``{i: [avg_dist, center_traj, dists, subtrajs]}``.
    """
    _require_mpl()
    _apply_style()
    gc.collect()

    fig, ax = plt.subplots(figsize=(10, 8))
    n_clusters = len(cluster_dict)
    colors = _get_distinct_colors(n_clusters)
    cluster_indices = sorted(cluster_dict.keys())

    for ci in cluster_indices:
        subtrajs = cluster_dict[ci][3][::trajectory_sample_rate]
        xs, ys = [], []
        for traj in subtrajs:
            pts = traj.points[::point_sample_rate]
            xs.extend(p.x for p in pts)
            ys.extend(p.y for p in pts)
        ax.scatter(xs, ys, s=3, color=colors[ci],
                   label=f"Cluster {ci + 1}", alpha=alpha)

    if show_centers:
        for ci in cluster_indices:
            center = cluster_dict[ci][1]
            pts = center.points[::point_sample_rate]
            xs = [p.x for p in pts]
            ys = [p.y for p in pts]
            ax.plot(xs, ys, color="black", linewidth=2, marker="*",
                    markersize=10, alpha=center_alpha,
                    markerfacecolor=colors[ci], markeredgecolor="black")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Trajectory Clusters ({method_name})")
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    if show_info:
        total = sum(len(cluster_dict[i][3]) for i in cluster_dict)
        displayed = sum(
            len(cluster_dict[i][3][::trajectory_sample_rate])
            for i in cluster_dict
        )
        _add_info_box(ax, f"Clusters: {n_clusters}\n"
                          f"Sub-trajs: {total}\n"
                          f"Displayed: {displayed}")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    gc.collect()


# ─────────────────────────────────────────────────────────────────────
# Metric computation helpers
# ─────────────────────────────────────────────────────────────────────

def compute_sse(cluster_results: Any) -> Tuple[float, int]:
    """SSE for clustering results.

    Args:
        cluster_results: ``[(overall_sim, overall_sim, cluster_dict)]``.

    Returns:
        ``(sse, n)`` — total SSE and valid assignment count.
    """
    from q_rlstc.data.rlstc_trajdistance import traj2trajIED

    cluster_dict = cluster_results[0][2]
    sse, n = 0.0, 0
    for idx in cluster_dict:
        center = cluster_dict[idx][1]
        for traj in cluster_dict[idx][3]:
            d = traj2trajIED(center.points, traj.points)
            if d < 1e9:
                sse += d ** 2
                n += 1
    return sse, n


def compute_silhouette(cluster_dict: Dict[int, Any]) -> float:
    """Average silhouette coefficient for trajectory clustering.

    Uses centre-distance pruning to avoid O(N^2) IED calls.

    Returns:
        Average silhouette in [-1, 1].
    """
    from q_rlstc.data.rlstc_trajdistance import traj2trajIED

    all_sil: List[float] = []
    for ci in cluster_dict:
        cluster_trajs = cluster_dict[ci][3]
        centre_dists: Dict[int, float] = {}
        for oi in cluster_dict:
            if oi != ci:
                d = traj2trajIED(cluster_dict[ci][1].points,
                                 cluster_dict[oi][1].points)
                if d < 1e9:
                    centre_dists[oi] = d

        for traj in cluster_trajs:
            if len(cluster_trajs) > 1:
                intra = [traj2trajIED(traj.points, o.points)
                         for o in cluster_trajs if o is not traj]
                intra = [d for d in intra if d < 1e9]
                a = float(np.mean(intra)) if intra else 0.0
            else:
                a = 0.0

            b = float("inf")
            for oi, cdist in centre_dists.items():
                if cdist >= b:
                    continue
                dists = [traj2trajIED(traj.points, ot.points)
                         for ot in cluster_dict[oi][3]]
                dists = [d for d in dists if d < 1e9]
                if dists:
                    b = min(b, float(np.mean(dists)))

            if b < float("inf"):
                s = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
                all_sil.append(s)

    return float(np.mean(all_sil)) if all_sil else 0.0


# ─────────────────────────────────────────────────────────────────────
# Elbow / Silhouette Analysis
# ─────────────────────────────────────────────────────────────────────

def plot_elbow(
    k_values: List[int],
    sse_values: List[float],
    out_path: Union[str, Path],
    method_name: str = "",
    n_values: Optional[List[int]] = None,
    quantum_sse: Optional[List[float]] = None,
    normalize: bool = True,
) -> None:
    """Elbow curve for optimal *k* determination."""
    _require_mpl()
    _apply_style()

    if normalize and n_values:
        y_vals = [s / n if n > 0 else 0 for s, n in zip(sse_values, n_values)]
        ylabel = "Avg SSE per assignment"
    else:
        y_vals = list(sse_values)
        ylabel = "SSE"

    fig, ax = plt.subplots()
    ax.plot(k_values, y_vals, "o-", color=COLORS["version_a"],
            label="Classical", linewidth=2)

    if quantum_sse is not None:
        q_y = quantum_sse
        if normalize and n_values:
            q_y = [s / n if n > 0 else 0 for s, n in zip(quantum_sse, n_values)]
        ax.plot(k_values, q_y, "s-", color=COLORS["version_b"],
                label="Quantum", linewidth=2)

    for k, y in zip(k_values, y_vals):
        ax.annotate(f"{y:.1f}", (k, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)

    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel(ylabel)
    title = f"{method_name} Elbow Analysis" if method_name else "Elbow Analysis"
    ax.set_title(title)
    ax.set_xticks(k_values)
    ax.legend()
    _add_info_box(ax, f"k range: {min(k_values)}-{max(k_values)}\n"
                      f"Min SSE at k={k_values[int(np.argmin(y_vals))]}")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_silhouette_analysis(
    k_values: List[int],
    silhouette_values: List[float],
    out_path: Union[str, Path],
    method_name: str = "",
    quantum_silhouette: Optional[List[float]] = None,
) -> None:
    """Silhouette score vs *k* with best-k annotation."""
    _require_mpl()
    _apply_style()

    fig, ax = plt.subplots()
    ax.plot(k_values, silhouette_values, "s-", color=COLORS["ideal"],
            label="Classical", linewidth=2)

    if quantum_silhouette is not None:
        ax.plot(k_values, quantum_silhouette, "o-", color=COLORS["version_b"],
                label="Quantum", linewidth=2)

    best_idx = int(np.argmax(silhouette_values))
    best_k = k_values[best_idx]
    best_score = silhouette_values[best_idx]
    ax.axhline(y=best_score, color="green", linestyle="--", alpha=0.5)

    for k, s in zip(k_values, silhouette_values):
        ax.annotate(f"{s:.3f}", (k, s), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)

    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    title = f"{method_name} Silhouette" if method_name else "Silhouette Analysis"
    ax.set_title(title)
    ax.set_xticks(k_values)
    ax.legend()
    _add_info_box(ax, f"Best: {best_score:.3f} (k={best_k})\n"
                      f"Mean: {np.mean(silhouette_values):.3f}")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Combined Classical-vs-Quantum Comparison
# ─────────────────────────────────────────────────────────────────────

def plot_combined_comparison(
    classical_results: Dict[str, Any],
    quantum_results: Dict[str, Any],
    out_dir: Union[str, Path],
    prefix: str = "comparison",
) -> None:
    """SSE, silhouette, and timing comparison panels.

    Args:
        classical_results: ``{"k_values", "sse", "silhouette", "times"}``.
        quantum_results: Same structure.
    """
    _require_mpl()
    _apply_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k_c = classical_results.get("k_values", [])
    k_q = quantum_results.get("k_values", [])

    if "sse" in classical_results and "sse" in quantum_results:
        fig, ax = plt.subplots()
        ax.plot(k_c, classical_results["sse"], "o-",
                color=COLORS["version_a"], label="Classical", linewidth=2)
        ax.plot(k_q, quantum_results["sse"], "s-",
                color=COLORS["version_b"], label="Quantum", linewidth=2)
        ax.set_xlabel("k"); ax.set_ylabel("SSE")
        ax.set_title("SSE: Classical vs Quantum")
        ax.legend(); ax.set_xticks(sorted(set(k_c + k_q)))
        fig.tight_layout()
        fig.savefig(str(out_dir / f"{prefix}_sse.png"), dpi=150, facecolor="white")
        plt.close(fig)

    if "silhouette" in classical_results and "silhouette" in quantum_results:
        fig, ax = plt.subplots()
        ax.plot(k_c, classical_results["silhouette"], "o-",
                color=COLORS["version_a"], label="Classical", linewidth=2)
        ax.plot(k_q, quantum_results["silhouette"], "s-",
                color=COLORS["version_b"], label="Quantum", linewidth=2)
        ax.set_xlabel("k"); ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette: Classical vs Quantum")
        ax.legend(); ax.set_xticks(sorted(set(k_c + k_q)))
        fig.tight_layout()
        fig.savefig(str(out_dir / f"{prefix}_silhouette.png"), dpi=150,
                    facecolor="white")
        plt.close(fig)

    if "times" in classical_results and "times" in quantum_results:
        fig, ax = plt.subplots()
        t_c, t_q = classical_results["times"], quantum_results["times"]
        ax.plot(k_c, t_c, "o-", color=COLORS["version_a"],
                label="Classical", linewidth=2)
        ax.plot(k_q, t_q, "s-", color=COLORS["version_b"],
                label="Quantum", linewidth=2)
        ax.set_xlabel("k"); ax.set_ylabel("Time (s)")
        ax.set_title("Timing: Classical vs Quantum")
        ax.legend()
        ratio = sum(t_c) / sum(t_q) if sum(t_q) > 0 else 0
        _add_info_box(ax, f"Classical: {sum(t_c):.1f}s\n"
                          f"Quantum: {sum(t_q):.1f}s\n"
                          f"Ratio: {ratio:.2f}x")
        fig.tight_layout()
        fig.savefig(str(out_dir / f"{prefix}_timing.png"), dpi=150,
                    facecolor="white")
        plt.close(fig)

    gc.collect()


# ─────────────────────────────────────────────────────────────────────
# Timing per-k
# ─────────────────────────────────────────────────────────────────────

def plot_timing_per_k(
    k_values: List[int],
    times: List[float],
    out_path: Union[str, Path],
    method_name: str = "",
    breakdown: Optional[Dict[str, List[float]]] = None,
) -> None:
    """Execution time vs k with optional sub-component breakdown."""
    _require_mpl()
    _apply_style()

    fig, ax = plt.subplots()
    ax.plot(k_values, times, "o-", color=COLORS["version_b"],
            label="Total", linewidth=2)
    if breakdown:
        sub_colors = [COLORS["version_a"], COLORS["ideal"], COLORS["eagle"]]
        for (name, vals), c in zip(breakdown.items(), sub_colors):
            ax.plot(k_values, vals, "--", color=c, alpha=0.7, label=name)
    avg_t = float(np.mean(times))
    ax.axhline(y=avg_t, color="gray", linestyle=":", alpha=0.5)
    for k, t in zip(k_values, times):
        ax.annotate(f"{t:.1f}s", (k, t), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7)
    ax.set_xlabel("k")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"{method_name} Timing" if method_name else "Timing Analysis")
    ax.set_xticks(k_values)
    ax.legend(loc="upper left")
    _add_info_box(ax, f"Total: {sum(times):.1f}s\nAvg: {avg_t:.1f}s/k")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Interactive Folium Map Overlay
# ─────────────────────────────────────────────────────────────────────

def plot_clusters_on_map(
    cluster_dict: Dict[int, Any],
    out_path: Union[str, Path],
    center: Tuple[float, float] = (39.9, 116.4),
    zoom: int = 12,
    sample_rate: int = 5,
) -> None:
    """Interactive HTML map with coloured cluster polylines.

    Args:
        cluster_dict: ``{i: [avg_dist, center_traj, dists, subtrajs]}``.
        out_path: Output ``.html`` path.
        center: ``(lat, lon)`` for initial map centre.
        zoom: Initial zoom level.
        sample_rate: Draw every Nth sub-trajectory.
    """
    if not FOLIUM_AVAILABLE:
        print("folium not installed - skipping map plot")
        return

    HEX = [
        '#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
        '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
    ]
    m = folium.Map(location=list(center), zoom_start=zoom,
                   tiles="CartoDB positron")

    for ci in sorted(cluster_dict.keys()):
        color = HEX[ci % len(HEX)]
        ctr = cluster_dict[ci][1]
        folium.PolyLine([[p.y, p.x] for p in ctr.points],
                        color=color, weight=5, opacity=1.0,
                        tooltip=f"Cluster {ci} Centre").add_to(m)
        for traj in cluster_dict[ci][3][::sample_rate]:
            folium.PolyLine([[p.y, p.x] for p in traj.points],
                            color=color, weight=2, opacity=0.4,
                            tooltip=f"Cluster {ci}").add_to(m)
    m.save(str(out_path))
    print(f"Map saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────
# Paper-Style Cluster Visualization (Fig 16)
# ─────────────────────────────────────────────────────────────────────

def plot_paper_style_clusters(
    trajectories: List[Any],
    centroids: List[Any],
    out_path: Union[str, Path],
    title: str = "Clustering Results on T-Drive",
    lon_range: Tuple[float, float] = (116.1, 116.7),
    lat_range: Tuple[float, float] = (39.7, 40.15),
    figsize: Tuple[int, int] = (12, 10),
) -> None:
    """Thin blue trajectories + thick red centroids (paper Fig 16 style)."""
    _require_mpl()
    _apply_style()
    gc.collect()

    fig, ax = plt.subplots(figsize=figsize)
    for traj in trajectories:
        if not traj:
            continue
        pts = traj.points if hasattr(traj, "points") else traj
        if not pts:
            continue
        first = pts[0]
        if not (lon_range[0] <= first.x <= lon_range[1]):
            continue
        ax.plot([p.x for p in pts], [p.y for p in pts],
                color="blue", linewidth=0.3, alpha=0.4)

    for centroid in centroids:
        if not centroid:
            continue
        pts = centroid.points if hasattr(centroid, "points") else centroid
        ax.plot([p.x for p in pts], [p.y for p in pts],
                color="red", linewidth=3.0, alpha=0.9, solid_capstyle="round")

    ax.set_xlim(*lon_range)
    ax.set_ylim(*lat_range)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.legend(handles=[
        plt.Line2D([0], [0], color="blue", lw=1, label="Trajectories"),
        plt.Line2D([0], [0], color="red", lw=3, label="Representatives"),
    ], loc="lower right")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    gc.collect()


# ─────────────────────────────────────────────────────────────────────
# QBAI Ansatz Analysis
# ─────────────────────────────────────────────────────────────────────

def plot_qbai_analysis(
    selection_history: List[Dict[str, Any]],
    out_path: Union[str, Path],
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """Circuit selection / best-arm identification analysis.

    Args:
        selection_history: List of dicts with ``"round"``, ``"arms"``,
            ``"scores"``, and ``"selected"`` keys.
    """
    _require_mpl()
    _apply_style()

    if not selection_history:
        return

    rounds = [h["round"] for h in selection_history]
    arms = selection_history[0].get("arms", [])
    n_arms = len(arms)
    scores = np.array([h["scores"] for h in selection_history])

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    colors = _get_distinct_colors(n_arms)
    for j in range(n_arms):
        ax.plot(rounds, scores[:, j], "o-", color=colors[j],
                label=arms[j], markersize=5)
    ax.set_xlabel("Elimination Round")
    ax.set_ylabel("Score")
    ax.set_title("QBAI — Arm Score Evolution")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    final = scores[-1]
    bar_colors = [COLORS["ideal"]
                  if arms[j] == selection_history[-1]["selected"]
                  else COLORS["classical"] for j in range(n_arms)]
    ax2.barh(arms, final, color=bar_colors)
    ax2.set_xlabel("Final Score")
    ax2.set_title("QBAI — Final Selection")
    _add_info_box(ax2, f"Selected: {selection_history[-1]['selected']}\n"
                       f"Rounds: {len(rounds)}")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
