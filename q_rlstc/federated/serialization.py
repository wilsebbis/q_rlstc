"""Serialization utilities for Q-FL model exchange.

Q-RLSTC models are extraordinarily small:
- Version A: 20 params × 4 bytes = 80 bytes
- Version B: 32 params × 4 bytes = 128 bytes
- Version C: 24 params × 4 bytes = 96 bytes (EQC is more efficient)

This makes them ideal for federated learning — model updates
fit in a single UDP packet.
"""

import numpy as np
import struct
from typing import Tuple


def serialize_params(params: np.ndarray) -> bytes:
    """Serialize model parameters to compact binary format.
    
    Uses float32 for minimal bandwidth. Includes a 4-byte header
    with the parameter count.
    
    Args:
        params: Parameter vector.
    
    Returns:
        Compact binary representation.
    
    Example:
        >>> params = np.array([0.1, -0.2, 0.3])
        >>> data = serialize_params(params)
        >>> len(data)  # 4 (header) + 3 * 4 (floats) = 16
        16
    """
    params_f32 = np.asarray(params, dtype=np.float32)
    header = struct.pack('<I', len(params_f32))
    return header + params_f32.tobytes()


def deserialize_params(data: bytes) -> np.ndarray:
    """Deserialize binary back to parameter vector.
    
    Args:
        data: Binary data from serialize_params.
    
    Returns:
        Parameter vector as float64 numpy array.
    """
    n_params = struct.unpack('<I', data[:4])[0]
    params = np.frombuffer(data[4:4 + n_params * 4], dtype=np.float32)
    return params.astype(np.float64)


def serialize_gradient(gradient: np.ndarray, scale: float = 1.0) -> bytes:
    """Serialize a gradient update for federated transmission.
    
    Optionally scales the gradient for differential privacy.
    
    Args:
        gradient: Gradient vector.
        scale: Scaling factor (for gradient clipping/DP noise).
    
    Returns:
        Compact binary gradient.
    """
    scaled = np.asarray(gradient * scale, dtype=np.float32)
    header = struct.pack('<If', len(scaled), scale)
    return header + scaled.tobytes()


def deserialize_gradient(data: bytes) -> Tuple[np.ndarray, float]:
    """Deserialize gradient from binary.
    
    Args:
        data: Binary gradient data.
    
    Returns:
        Tuple of (gradient vector, scale factor).
    """
    n_params, scale = struct.unpack('<If', data[:8])
    gradient = np.frombuffer(data[8:8 + n_params * 4], dtype=np.float32)
    return gradient.astype(np.float64), scale


def model_size_bytes(n_params: int) -> int:
    """Calculate model transmission size in bytes.
    
    Args:
        n_params: Number of parameters.
    
    Returns:
        Total bytes (including header).
    """
    return 4 + n_params * 4
