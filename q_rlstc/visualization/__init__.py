"""
Visualization module for Q-RLSTC trajectory clustering.

Provides plotting utilities for:
- Learning curves (reward + loss)
- OD convergence tracking
- Metric comparison bar charts
- Noise impact analysis
- Timing breakdowns
- Circuit summary tables
- Cluster assignment scatter plots
- Segmentation boundary timelines
- Backend performance comparison

And a BenchmarkRunner for reproducible, tiered experiments.
"""

from .plot_utils import (
    plot_learning_curves,
    plot_od_convergence,
    plot_metric_comparison,
    plot_noise_impact,
    plot_epsilon_schedule,
    plot_timing_breakdown,
    plot_circuit_summary,
    plot_cluster_assignments,
    plot_segmentation_boundaries,
    plot_backend_comparison,
    save_results_json,
    STYLE_CONFIG,
)

from .benchmark import (
    BenchmarkRunner,
    BenchmarkResults,
)

__all__ = [
    "plot_learning_curves",
    "plot_od_convergence",
    "plot_metric_comparison",
    "plot_noise_impact",
    "plot_epsilon_schedule",
    "plot_timing_breakdown",
    "plot_circuit_summary",
    "plot_cluster_assignments",
    "plot_segmentation_boundaries",
    "plot_backend_comparison",
    "save_results_json",
    "STYLE_CONFIG",
    "BenchmarkRunner",
    "BenchmarkResults",
]
