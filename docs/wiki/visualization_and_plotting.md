# Visualization & Plotting

The `q_rlstc.visualization` module provides **13 publication-quality plotting functions** and a `BenchmarkRunner` for reproducible experiments.

## Quick Start

```python
from q_rlstc.visualization import (
    plot_learning_curves,
    plot_cluster_assignments,
    plot_segmentation_boundaries,
    BenchmarkRunner,
)
```

Or from the CLI:

```bash
python experiments/run_full_benchmark.py --tier small
```

## Plot Functions

### Training & Performance

| Function | Description |
|---|---|
| `plot_learning_curves()` | Dual-axis reward + loss, A vs B overlay, moving-average smoothing |
| `plot_od_convergence()` | OD vs epoch with ΔOD annotation and improvement percentage |
| `plot_metric_comparison()` | Grouped bar chart: F1, ΔOD, param efficiency, avg reward |
| `plot_noise_impact()` | Reward curves under ideal/Eagle/Heron with resilience ratios |
| `plot_epsilon_schedule()` | ε-greedy decay visualization |

### Clustering & Segmentation

| Function | Description |
|---|---|
| `plot_cluster_assignments()` | Scatter plot of trajectory points coloured by cluster, with centroids |
| `plot_segmentation_boundaries()` | Timeline showing where segments are cut vs ground truth |

### Infrastructure

| Function | Description |
|---|---|
| `plot_timing_breakdown()` | Stacked horizontal bar of runtime components |
| `plot_circuit_summary()` | Table-figure with gate counts, depth, params for Version A/B |
| `plot_backend_comparison()` | Grouped bar chart comparing CPU/MLX/CUDA timing |

### Utilities

| Function | Description |
|---|---|
| `save_results_json()` | NumPy-safe JSON serialisation |

## Using Cluster Plots

The cluster assignment plot requires a `(N, 2)` array of point coordinates and an `(N,)` array of integer cluster labels:

```python
import numpy as np
from q_rlstc.visualization import plot_cluster_assignments

points = np.random.randn(500, 2)
labels = np.random.randint(0, 5, size=500)
centroids = np.array([[0, 0], [1, 1], [-1, 1], [1, -1], [-1, -1]])

plot_cluster_assignments(
    points, labels, centroids,
    out_path="clusters.png",
    version_label="Classical Parity (5q)",
)
```

## Using Segmentation Plots

The segmentation boundary plot shows predicted vs ground-truth segment breaks:

```python
from q_rlstc.visualization import plot_segmentation_boundaries

# trajectory: (T, 2) array of coordinates
# predicted: list of point indices where segments break
plot_segmentation_boundaries(
    trajectory,
    predicted_boundaries=[15, 42, 78],
    ground_truth_boundaries=[14, 40, 80],
    out_path="segments.png",
    version_label="Quantum Enhanced (8q)",
)
```

## Style Configuration

All plots use a consistent publication-quality style via `STYLE_CONFIG`:

- 150 DPI output
- White background with subtle grid
- 12pt axis labels, 14pt titles
- Consistent colour palette:
  - **Blue** (`#4363d8`) — Version A / Classical Parity
  - **Red** (`#e6194B`) — Version B / Quantum Enhanced
  - **Green** (`#3cb44b`) — Ideal/noiseless
  - **Orange** (`#f58231`) — Eagle noise
  - **Purple** (`#911eb4`) — Heron noise

## Dependencies

Plots require `matplotlib`:

```bash
uv pip install matplotlib
# or
pip install 'q_rlstc[viz]'
```

---

← [Benchmarking](benchmarking.md) | [Compute Backends](compute_backends.md) →
