"""Hilbert Space-Filling Curve for spatial locality-preserving projection.

Projects 2D spatial coordinates (longitude, latitude) to a 1D scalar
using a Hilbert curve. This mathematically preserves spatial locality —
points close on the 2D map remain close in the 1D projection.

This enables encoding 2D start/end coordinates as single qubit
rotation angles without losing geographic meaning.

Used by Version D (VLDB Aligned) to compress the VLDB paper's
Starting/Ending Point features from 4D (2×2) into 2D (2×1),
keeping the total state vector at exactly 5 dimensions for 5 qubits.

References:
    - Hilbert, D. (1891). "Über die stetige Abbildung einer Linie auf ein Flächenstück"
    - The implementation maps a bounded spatial region into [0, 1] using
      an order-P Hilbert curve, where P controls the resolution.
"""

import numpy as np
from typing import Tuple, Optional


def _rotate_quadrant(
    n: int,
    x: int,
    y: int,
    rx: int,
    ry: int,
) -> Tuple[int, int]:
    """Rotate/flip a quadrant in the Hilbert curve.
    
    Args:
        n: Grid size (power of 2).
        x: Current x coordinate.
        y: Current y coordinate.
        rx: Rotation flag x.
        ry: Rotation flag y.
    
    Returns:
        Rotated (x, y) coordinates.
    """
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        # Swap x and y
        x, y = y, x
    return x, y


def xy_to_hilbert(x: int, y: int, order: int = 16) -> int:
    """Convert 2D grid coordinates to Hilbert curve index.
    
    Args:
        x: Grid x coordinate [0, 2^order - 1].
        y: Grid y coordinate [0, 2^order - 1].
        order: Hilbert curve order (resolution = 2^order).
    
    Returns:
        Hilbert curve index [0, 4^order - 1].
    """
    n = 1 << order  # 2^order
    d = 0
    s = n // 2
    
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = _rotate_quadrant(s, x, y, rx, ry)
        s //= 2
    
    return d


def spatial_to_hilbert(
    x: float,
    y: float,
    x_min: float = -500.0,
    x_max: float = 500.0,
    y_min: float = -500.0,
    y_max: float = 500.0,
    order: int = 12,
) -> float:
    """Project 2D spatial coordinates to a 1D Hilbert scalar in [0, 1].
    
    Normalises the input coordinates to a grid, computes the Hilbert
    curve index, and returns it normalised to [0, 1].
    
    Args:
        x: X-coordinate (e.g., longitude or meters).
        y: Y-coordinate (e.g., latitude or meters).
        x_min: Minimum X bound of the study area.
        x_max: Maximum X bound of the study area.
        y_min: Minimum Y bound of the study area.
        y_max: Maximum Y bound of the study area.
        order: Hilbert curve order (12 = 4096×4096 grid = ~meter resolution).
    
    Returns:
        Normalised Hilbert index in [0, 1].
    
    Example:
        >>> spatial_to_hilbert(0.0, 0.0)  # Center of default grid
        0.5  # (approximately — Hilbert curve maps center to mid-range)
    """
    n = 1 << order  # Grid resolution: 2^order
    max_d = (1 << (2 * order))  # Maximum Hilbert index: 4^order
    
    # Clamp and normalise to [0, n-1] grid coordinates
    x_norm = np.clip((x - x_min) / (x_max - x_min), 0.0, 1.0)
    y_norm = np.clip((y - y_min) / (y_max - y_min), 0.0, 1.0)
    
    gx = int(x_norm * (n - 1))
    gy = int(y_norm * (n - 1))
    
    # Compute Hilbert index and normalise to [0, 1]
    d = xy_to_hilbert(gx, gy, order)
    return d / max_d


def compute_spatial_bounds(
    trajectories,
    padding: float = 0.1,
) -> Tuple[float, float, float, float]:
    """Compute bounding box from a set of trajectories.
    
    Args:
        trajectories: List of Trajectory objects.
        padding: Fractional padding around the bounding box.
    
    Returns:
        Tuple of (x_min, x_max, y_min, y_max).
    """
    all_x = []
    all_y = []
    
    for traj in trajectories:
        for p in traj.points:
            all_x.append(p.x)
            all_y.append(p.y)
    
    if not all_x:
        return -500.0, 500.0, -500.0, 500.0
    
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    # Add padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Ensure non-zero range
    x_range = max(x_range, 1.0)
    y_range = max(y_range, 1.0)
    
    x_min -= padding * x_range
    x_max += padding * x_range
    y_min -= padding * y_range
    y_max += padding * y_range
    
    return x_min, x_max, y_min, y_max
