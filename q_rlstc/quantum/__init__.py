"""Quantum module for VQ-DQN circuits.

Contains:
- VQ-DQN circuit builder with angle encoding
- Backend factory for Aer simulators
- Readout error mitigation

Note: Swap test was removed. All distance computation is classical.
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
from .mitigation import (
    ReadoutMitigator,
    apply_mitigation,
)

__all__ = [
    "build_vqdqn_circuit",
    "evaluate_q_values",
    "angle_encode",
    "VQDQNCircuitBuilder",
    "get_backend",
    "BackendFactory",
    "ReadoutMitigator",
    "apply_mitigation",
]
