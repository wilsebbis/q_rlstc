"""
Publication-quality plotting utilities for Q-RLSTC experiments.

Adapted from patterns in TheFinalQRLSTC/visualization/plot_utils.py,
New_QRLSTC/QRLSTCcode-theoretical/enhanced_plots.py, and
QRLSTC/plot_utils.py.

All plots are 150 DPI, bbox_inches='tight', with consistent style.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


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


def _add_info_box(ax: plt.Axes, text: str, loc: str = "upper right"):
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
        "mlx": "#4363d8",
        "cuda": "#3cb44b",
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

