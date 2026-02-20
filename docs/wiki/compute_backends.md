# Compute Backends (MLX / CUDA / CPU)

Q-RLSTC supports hardware-accelerated computation for distance calculations, array operations, and optimizer steps via three backends.

## Backend Priority

When `--backend auto` is used (default), the system auto-detects:

1. **Apple MLX** — M1/M2/M3/M4 Apple Silicon via `mlx.core`
2. **NVIDIA CUDA** — via PyTorch (`torch.cuda`)
3. **CPU** — NumPy fallback (always available)

## Module: `q_rlstc.accelerator`

| Component | Purpose |
|---|---|
| `ComputeBackend` (Enum) | `CPU`, `MLX`, `CUDA`, `AUTO` |
| `detect_best_backend()` | Auto-detect best available backend |
| `resolve_backend(name)` | Resolve `"auto"/"cpu"/"mlx"/"cuda"` to `ComputeBackend` |
| `AcceleratedArray` | Unified array wrapper (NumPy/MLX/Torch) |
| `pairwise_distances(X, Y)` | Accelerated Euclidean distance matrix |
| `get_device_info()` | Dict of available hardware and versions |

## CLI Usage

```bash
# Auto-detect (recommended)
python experiments/run_full_benchmark.py

# Force specific backend
python experiments/run_full_benchmark.py --backend mlx
python experiments/run_full_benchmark.py --backend cuda
python experiments/run_full_benchmark.py --backend cpu

# Compare all available backends
python experiments/run_full_benchmark.py --compare-backends
```

## Python API

```python
from q_rlstc.accelerator import (
    resolve_backend,
    pairwise_distances,
    get_device_info,
)

# Check what's available
info = get_device_info()
print(info)
# {'numpy_version': '1.26.4', 'backends': {'cpu': True, 'mlx': True, 'cuda': False}, 'best': 'mlx'}

# Use accelerated distances
backend = resolve_backend("mlx")
distances = pairwise_distances(X, Y, backend=backend)
```

## Configuration

The `QRLSTCConfig` master config includes a `compute_backend` field:

```python
from q_rlstc.config import QRLSTCConfig

config = QRLSTCConfig(
    version="A",
    compute_backend="mlx",  # "auto", "cpu", "mlx", "cuda"
)
```

## Installing Backend Dependencies

```bash
# Apple MLX (macOS Apple Silicon only)
pip install mlx

# CUDA (requires NVIDIA GPU + CUDA toolkit)
pip install torch  # with CUDA support

# CPU — no extra dependencies needed
```

## Backend Comparison Plots

When running `--compare-backends`, the CLI generates a timing comparison chart in the output directory:

```
outputs/benchmark_20260220/plots/backend_comparison.png
```

This shows grouped bars of runtime per run per backend, making it easy to see where MLX or CUDA provide speedup.

## Important Notes

- **Quantum circuit simulation** always uses Qiskit Aer (CPU-based) regardless of compute backend. The compute backend only accelerates array operations (distance matrices, clustering, feature extraction).
- **MLX** uses `float32` precision for best GPU utilisation on Apple Silicon.
- **CUDA** uses `float64` to match NumPy CPU precision exactly.
- Lazy imports: `mlx.core` and `torch` are only imported when their backend is selected.

---

← [Visualization & Plotting](visualization_and_plotting.md) | [Benchmarking](benchmarking.md) →
