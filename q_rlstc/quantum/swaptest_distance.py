"""Swap test for quantum distance estimation.

Implements amplitude encoding and swap test circuits to compute
distance between trajectory vectors using quantum interference.

For normalized vectors û and v̂:
    |⟨û|v̂⟩|² = 2*P(0) - 1  (swap test)
    d²(u,v) = ||u||² + ||v||² - 2||u|| ||v|| ⟨û|v̂⟩
"""

import numpy as np
from typing import Dict, Optional, Tuple, Callable
from dataclasses import dataclass
from functools import lru_cache

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator


def normalize_and_pad(vector: np.ndarray) -> Tuple[np.ndarray, float, int]:
    """Normalize vector and pad to power of 2.
    
    Args:
        vector: Input vector.
    
    Returns:
        Tuple of (normalized padded vector, original norm, n_qubits needed).
    """
    vector = np.asarray(vector).flatten()
    norm = np.linalg.norm(vector)
    
    if norm < 1e-12:
        norm = 1.0
        normalized = np.zeros_like(vector)
    else:
        normalized = vector / norm
    
    # Pad to power of 2
    n = len(normalized)
    n_qubits = int(np.ceil(np.log2(max(n, 2))))
    padded_len = 2 ** n_qubits
    
    padded = np.zeros(padded_len)
    padded[:n] = normalized
    
    # Re-normalize after padding
    pad_norm = np.linalg.norm(padded)
    if pad_norm > 1e-12:
        padded = padded / pad_norm
    
    return padded, norm, n_qubits


def amplitude_encode_circuit(vector: np.ndarray, name: str = "prep") -> QuantumCircuit:
    """Create circuit that prepares amplitude-encoded state.
    
    Uses Qiskit's initialize method for exact state preparation.
    
    Args:
        vector: Normalized vector to encode (length must be power of 2).
        name: Name for the circuit.
    
    Returns:
        QuantumCircuit that prepares the state.
    """
    n = len(vector)
    n_qubits = int(np.log2(n))
    
    circuit = QuantumCircuit(n_qubits, name=name)
    
    # Normalize if not already
    norm = np.linalg.norm(vector)
    if norm > 1e-12:
        state = vector / norm
    else:
        state = vector
    
    # Use initialize (will be decomposed by Qiskit)
    circuit.initialize(state, range(n_qubits))
    
    return circuit


@dataclass 
class SwapTestResult:
    """Result from a swap test.
    
    Attributes:
        overlap_squared: |⟨ψ|φ⟩|² estimate.
        distance: Euclidean distance estimate.
        p_zero: Probability of measuring 0 on ancilla.
        shots: Number of shots used.
    """
    overlap_squared: float
    distance: float
    p_zero: float
    shots: int


class SwapTestDistanceEstimator:
    """Estimator for quantum distance using swap test.
    
    Uses amplitude encoding and swap test to estimate Euclidean distance
    between vectors. Includes caching for repeated state preparations.
    """
    
    def __init__(
        self,
        backend: Optional[AerSimulator] = None,
        default_shots: int = 1024,
        cache_size: int = 128,
    ):
        """Initialize estimator.
        
        Args:
            backend: Qiskit backend for execution.
            default_shots: Default measurement shots.
            cache_size: LRU cache size for state prep circuits.
        """
        self.backend = backend or AerSimulator()
        self.default_shots = default_shots
        self._circuit_cache: Dict[str, QuantumCircuit] = {}
        self._cache_size = cache_size
    
    def _get_cache_key(self, vector: np.ndarray) -> str:
        """Get cache key for vector."""
        return hash(vector.tobytes())
    
    def _build_swap_test_circuit(
        self,
        vector_u: np.ndarray,
        vector_v: np.ndarray,
    ) -> Tuple[QuantumCircuit, float, float]:
        """Build swap test circuit for two vectors.
        
        Args:
            vector_u: First vector.
            vector_v: Second vector.
        
        Returns:
            Tuple of (circuit, norm_u, norm_v).
        """
        # Normalize and pad both vectors
        u_prep, norm_u, n_qubits_u = normalize_and_pad(vector_u)
        v_prep, norm_v, n_qubits_v = normalize_and_pad(vector_v)
        
        # Use larger qubit count
        n_qubits = max(n_qubits_u, n_qubits_v)
        
        # Re-pad to same size if needed
        target_len = 2 ** n_qubits
        if len(u_prep) < target_len:
            u_new = np.zeros(target_len)
            u_new[:len(u_prep)] = u_prep
            u_prep = u_new / np.linalg.norm(u_new) if np.linalg.norm(u_new) > 0 else u_new
        if len(v_prep) < target_len:
            v_new = np.zeros(target_len)
            v_new[:len(v_prep)] = v_prep
            v_prep = v_new / np.linalg.norm(v_new) if np.linalg.norm(v_new) > 0 else v_new
        
        # Create registers
        ancilla = QuantumRegister(1, 'anc')
        reg_a = QuantumRegister(n_qubits, 'a')
        reg_b = QuantumRegister(n_qubits, 'b')
        cr = ClassicalRegister(1, 'c')
        
        circuit = QuantumCircuit(ancilla, reg_a, reg_b, cr)
        
        # Step 1: Hadamard on ancilla
        circuit.h(ancilla[0])
        
        # Step 2: Prepare |ψ_u⟩ on register A
        prep_u = amplitude_encode_circuit(u_prep, 'u')
        circuit.compose(prep_u, reg_a, inplace=True)
        
        # Step 3: Prepare |ψ_v⟩ on register B
        prep_v = amplitude_encode_circuit(v_prep, 'v')
        circuit.compose(prep_v, reg_b, inplace=True)
        
        # Step 4: Controlled-SWAP
        for i in range(n_qubits):
            circuit.cswap(ancilla[0], reg_a[i], reg_b[i])
        
        # Step 5: Hadamard on ancilla
        circuit.h(ancilla[0])
        
        # Step 6: Measure ancilla
        circuit.measure(ancilla[0], cr[0])
        
        return circuit, norm_u, norm_v
    
    def estimate_overlap(
        self,
        vector_u: np.ndarray,
        vector_v: np.ndarray,
        shots: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Estimate overlap between two vectors.
        
        Args:
            vector_u: First vector.
            vector_v: Second vector.
            shots: Measurement shots (uses default if None).
        
        Returns:
            Tuple of (overlap_squared, p_zero).
        """
        shots = shots or self.default_shots
        
        circuit, _, _ = self._build_swap_test_circuit(vector_u, vector_v)
        
        # Execute
        transpiled = transpile(circuit, self.backend)
        job = self.backend.run(transpiled, shots=shots)
        counts = job.result().get_counts()
        
        # P(0) = count of '0' / shots
        count_0 = counts.get('0', 0)
        p_zero = count_0 / shots
        
        # |⟨ψ|φ⟩|² = 2*P(0) - 1
        overlap_squared = max(0.0, 2 * p_zero - 1)
        
        return overlap_squared, p_zero
    
    def estimate_distance(
        self,
        vector_u: np.ndarray,
        vector_v: np.ndarray,
        shots: Optional[int] = None,
    ) -> SwapTestResult:
        """Estimate Euclidean distance using swap test.
        
        For vectors u and v with norms ||u|| and ||v||:
            d²(u,v) = ||u||² + ||v||² - 2||u|| ||v|| ⟨û|v̂⟩
        
        Args:
            vector_u: First vector.
            vector_v: Second vector.
            shots: Measurement shots.
        
        Returns:
            SwapTestResult with distance estimate.
        """
        shots = shots or self.default_shots
        
        # Get norms
        u = np.asarray(vector_u).flatten()
        v = np.asarray(vector_v).flatten()
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        
        # Estimate overlap
        overlap_squared, p_zero = self.estimate_overlap(u, v, shots)
        
        # For real vectors, overlap = sqrt(overlap_squared)
        overlap = np.sqrt(overlap_squared)
        
        # Distance formula
        dist_squared = norm_u**2 + norm_v**2 - 2 * norm_u * norm_v * overlap
        distance = np.sqrt(max(0.0, dist_squared))
        
        return SwapTestResult(
            overlap_squared=overlap_squared,
            distance=distance,
            p_zero=p_zero,
            shots=shots,
        )


def swaptest_distance(
    x: np.ndarray,
    y: np.ndarray,
    backend: Optional[AerSimulator] = None,
    shots: int = 1024,
) -> float:
    """Compute distance between vectors using swap test.
    
    Convenience function for single distance calculation.
    
    Args:
        x: First vector.
        y: Second vector.
        backend: Qiskit backend.
        shots: Measurement shots.
    
    Returns:
        Estimated Euclidean distance.
    """
    estimator = SwapTestDistanceEstimator(backend, shots)
    result = estimator.estimate_distance(x, y, shots)
    return result.distance


def classical_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Classical Euclidean distance for comparison.
    
    Args:
        x: First vector.
        y: Second vector.
    
    Returns:
        Euclidean distance.
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    # Pad to same length
    max_len = max(len(x), len(y))
    x_pad = np.zeros(max_len)
    y_pad = np.zeros(max_len)
    x_pad[:len(x)] = x
    y_pad[:len(y)] = y
    
    return np.linalg.norm(x_pad - y_pad)
