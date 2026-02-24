"""Quantum module for VQ-DQN circuits.

Contains:
- VQ-DQN circuit builder with angle encoding
- Backend factory for Aer simulators
"""

from .vqdqn_circuit import (
    build_vqdqn_circuit,
    evaluate_q_values,
    angle_encode,
    VQDQNCircuitBuilder,
)
from .backends import (
    get_backend,
    BackendFactory,
)

__all__ = [
    "build_vqdqn_circuit",
    "evaluate_q_values",
    "angle_encode",
    "VQDQNCircuitBuilder",
    "get_backend",
    "BackendFactory",
]
