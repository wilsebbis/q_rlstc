"""Quantum module for VQ-DQN circuits and swap test distance."""

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
from .swaptest_distance import (
    swaptest_distance,
    SwapTestDistanceEstimator,
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
    "swaptest_distance",
    "SwapTestDistanceEstimator",
    "ReadoutMitigator",
    "apply_mitigation",
]
