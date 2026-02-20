"""
Hardware acceleration module for Q-RLSTC.

Auto-detects and uses available compute backends:
- Apple MLX  (M1/M2/M3/M4 Apple Silicon)
- NVIDIA CUDA  (via PyTorch)
- CPU  (NumPy fallback — always available)

Ported from New_QRLSTC/QRLSTCcode-main/subtrajcluster/accelerator.py.
"""

from enum import Enum
from typing import Any, Dict, Optional

import numpy as np


# ── Lazy availability flags ───────────────────────────────────────────

_BACKENDS: Dict[str, bool] = {
    "mlx": False,
    "cuda": False,
    "numpy": True,
}


class ComputeBackend(str, Enum):
    """Available computation backends."""

    CPU = "cpu"
    MLX = "mlx"
    CUDA = "cuda"
    AUTO = "auto"


# ── Detection ─────────────────────────────────────────────────────────

def _detect_mlx() -> bool:
    """Detect if Apple MLX is available."""
    try:
        import mlx.core as mx  # noqa: F401

        # Quick smoke test
        _ = mx.array([1.0, 2.0])
        _BACKENDS["mlx"] = True
        return True
    except Exception:
        return False


def _detect_cuda() -> bool:
    """Detect if CUDA is available via PyTorch."""
    try:
        import torch

        if torch.cuda.is_available():
            _BACKENDS["cuda"] = True
            return True
    except Exception:
        pass
    return False


def detect_best_backend() -> ComputeBackend:
    """Auto-detect the best available backend.

    Priority: MLX → CUDA → CPU.
    """
    if _detect_mlx():
        return ComputeBackend.MLX
    if _detect_cuda():
        return ComputeBackend.CUDA
    return ComputeBackend.CPU


def resolve_backend(requested: str = "auto") -> ComputeBackend:
    """Resolve a user-requested backend string to a concrete backend.

    Args:
        requested: One of "auto", "cpu", "mlx", "cuda".

    Returns:
        Resolved ComputeBackend.

    Raises:
        RuntimeError: If the requested backend is not available.
    """
    requested = requested.lower().strip()

    if requested == "auto":
        return detect_best_backend()

    if requested == "cpu":
        return ComputeBackend.CPU

    if requested == "mlx":
        if _detect_mlx():
            return ComputeBackend.MLX
        raise RuntimeError(
            "MLX requested but not available. "
            "Install with: pip install mlx  (Apple Silicon only)"
        )

    if requested == "cuda":
        if _detect_cuda():
            return ComputeBackend.CUDA
        raise RuntimeError(
            "CUDA requested but not available. "
            "Install PyTorch with CUDA support."
        )

    raise ValueError(f"Unknown backend: {requested!r}. Use auto/cpu/mlx/cuda.")


# ── Accelerated Array ─────────────────────────────────────────────────

class AcceleratedArray:
    """Unified array interface that works with any backend.

    Wraps NumPy, MLX, or PyTorch tensors behind a common API.
    """

    def __init__(self, data: Any, backend: Optional[ComputeBackend] = None):
        self.backend = backend or detect_best_backend()
        self._data = self._convert(data)

    def _convert(self, data: Any) -> Any:
        """Convert input to the selected backend format."""
        arr = np.asarray(data, dtype=np.float64) if not isinstance(data, np.ndarray) else data

        if self.backend == ComputeBackend.MLX:
            import mlx.core as mx
            return mx.array(arr.astype(np.float32))

        if self.backend == ComputeBackend.CUDA:
            import torch
            return torch.tensor(arr, dtype=torch.float64, device="cuda")

        return arr

    def to_numpy(self) -> np.ndarray:
        """Convert back to NumPy."""
        if self.backend == ComputeBackend.MLX:
            return np.array(self._data)
        if self.backend == ComputeBackend.CUDA:
            return self._data.cpu().numpy()
        return self._data

    @property
    def shape(self):
        return self._data.shape


# ── Accelerated Distance ──────────────────────────────────────────────

def pairwise_distances(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    backend: Optional[ComputeBackend] = None,
) -> np.ndarray:
    """Compute pairwise Euclidean distances with hardware acceleration.

    Args:
        X: Array of shape (n, d).
        Y: Array of shape (m, d). If None, computes X vs X.
        backend: Force specific backend (auto-detect if None).

    Returns:
        Distance matrix of shape (n, m).
    """
    backend = backend or detect_best_backend()

    if Y is None:
        Y = X

    if backend == ComputeBackend.MLX:
        import mlx.core as mx

        x = mx.array(X.astype(np.float32))
        y = mx.array(Y.astype(np.float32))
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
        xx = mx.sum(x * x, axis=1, keepdims=True)
        yy = mx.sum(y * y, axis=1, keepdims=True)
        dist_sq = xx + mx.transpose(yy) - 2.0 * (x @ mx.transpose(y))
        dist_sq = mx.maximum(dist_sq, mx.array(0.0))
        return np.array(mx.sqrt(dist_sq))

    if backend == ComputeBackend.CUDA:
        import torch

        x = torch.tensor(X, dtype=torch.float64, device="cuda")
        y = torch.tensor(Y, dtype=torch.float64, device="cuda")
        return torch.cdist(x, y).cpu().numpy()

    # CPU fallback
    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


# ── Device Info ───────────────────────────────────────────────────────

def get_device_info() -> Dict[str, Any]:
    """Get information about available compute devices.

    Returns:
        Dict with detected backends, best backend, and version info.
    """
    info: Dict[str, Any] = {
        "numpy_version": np.__version__,
        "backends": {},
        "best": str(detect_best_backend().value),
    }

    info["backends"]["cpu"] = True

    if _detect_mlx():
        try:
            import mlx.core as mx
            info["backends"]["mlx"] = True
            info["mlx_version"] = mx.__version__ if hasattr(mx, "__version__") else "unknown"
        except Exception:
            info["backends"]["mlx"] = False
    else:
        info["backends"]["mlx"] = False

    if _detect_cuda():
        try:
            import torch
            info["backends"]["cuda"] = True
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["torch_version"] = torch.__version__
        except Exception:
            info["backends"]["cuda"] = False
    else:
        info["backends"]["cuda"] = False

    return info
