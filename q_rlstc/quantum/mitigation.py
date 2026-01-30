"""Readout error mitigation for NISQ circuits.

Provides basic readout error mitigation using calibration matrices
when available. Falls back to no-op if mitigation is not possible.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MitigationResult:
    """Result of applying mitigation.
    
    Attributes:
        mitigated_counts: Counts after mitigation.
        raw_counts: Original counts before mitigation.
        mitigation_applied: Whether mitigation was actually applied.
    """
    mitigated_counts: Dict[str, float]
    raw_counts: Dict[str, int]
    mitigation_applied: bool


class ReadoutMitigator:
    """Readout error mitigator using calibration matrix.
    
    Implements simple matrix-based mitigation for readout errors.
    If no calibration is available, acts as a pass-through.
    """
    
    def __init__(
        self,
        n_qubits: int,
        calibration_matrix: Optional[np.ndarray] = None,
    ):
        """Initialize mitigator.
        
        Args:
            n_qubits: Number of qubits.
            calibration_matrix: Pre-computed calibration matrix.
                Shape: (2^n, 2^n) mapping ideal to noisy probabilities.
                If None, no mitigation is applied.
        """
        self.n_qubits = n_qubits
        self.n_states = 2 ** n_qubits
        self.calibration_matrix = calibration_matrix
        self._inverse_matrix: Optional[np.ndarray] = None
        
        if calibration_matrix is not None:
            try:
                self._inverse_matrix = np.linalg.pinv(calibration_matrix)
            except np.linalg.LinAlgError:
                self._inverse_matrix = None
    
    @classmethod
    def from_calibration_circuits(
        cls,
        backend,
        n_qubits: int,
        shots: int = 8192,
    ) -> "ReadoutMitigator":
        """Create mitigator by running calibration circuits.
        
        Runs 2^n circuits preparing each computational basis state
        and measures to build calibration matrix.
        
        Args:
            backend: Qiskit backend for calibration.
            n_qubits: Number of qubits.
            shots: Shots for each calibration circuit.
        
        Returns:
            Configured ReadoutMitigator.
        """
        from qiskit import QuantumCircuit, transpile
        
        n_states = 2 ** n_qubits
        calibration_matrix = np.zeros((n_states, n_states))
        
        for state_idx in range(n_states):
            # Prepare basis state
            circuit = QuantumCircuit(n_qubits, n_qubits)
            state_binary = format(state_idx, f'0{n_qubits}b')
            
            for qubit, bit in enumerate(reversed(state_binary)):
                if bit == '1':
                    circuit.x(qubit)
            
            circuit.measure(range(n_qubits), range(n_qubits))
            
            # Run and collect counts
            transpiled = transpile(circuit, backend)
            job = backend.run(transpiled, shots=shots)
            counts = job.result().get_counts()
            
            # Build column of calibration matrix
            for measured_state, count in counts.items():
                measured_idx = int(measured_state, 2)
                calibration_matrix[measured_idx, state_idx] = count / shots
        
        return cls(n_qubits, calibration_matrix)
    
    def mitigate(self, counts: Dict[str, int], shots: int) -> MitigationResult:
        """Apply readout error mitigation to measurement counts.
        
        Args:
            counts: Raw measurement counts.
            shots: Total number of shots.
        
        Returns:
            MitigationResult with mitigated counts.
        """
        if self._inverse_matrix is None:
            # No mitigation available - pass through
            return MitigationResult(
                mitigated_counts={k: float(v) for k, v in counts.items()},
                raw_counts=counts,
                mitigation_applied=False,
            )
        
        # Convert counts to probability vector
        prob_vector = np.zeros(self.n_states)
        for bitstring, count in counts.items():
            idx = int(bitstring.zfill(self.n_qubits), 2)
            prob_vector[idx] = count / shots
        
        # Apply inverse calibration matrix
        mitigated_probs = self._inverse_matrix @ prob_vector
        
        # Clip negative probabilities (can occur due to noise)
        mitigated_probs = np.clip(mitigated_probs, 0, None)
        
        # Renormalize
        if mitigated_probs.sum() > 0:
            mitigated_probs /= mitigated_probs.sum()
        
        # Convert back to counts format
        mitigated_counts = {}
        for idx in range(self.n_states):
            if mitigated_probs[idx] > 1e-10:
                bitstring = format(idx, f'0{self.n_qubits}b')
                mitigated_counts[bitstring] = mitigated_probs[idx] * shots
        
        return MitigationResult(
            mitigated_counts=mitigated_counts,
            raw_counts=counts,
            mitigation_applied=True,
        )
    
    def mitigate_expectation(
        self,
        counts: Dict[str, int],
        shots: int,
        qubit_idx: int,
    ) -> Tuple[float, bool]:
        """Mitigate and compute Z-expectation for a qubit.
        
        Args:
            counts: Raw measurement counts.
            shots: Total shots.
            qubit_idx: Qubit to compute expectation for.
        
        Returns:
            Tuple of (expectation value, whether mitigation was applied).
        """
        result = self.mitigate(counts, shots)
        
        expectation = 0.0
        for bitstring, count in result.mitigated_counts.items():
            padded = bitstring.zfill(self.n_qubits)
            bit = padded[-(qubit_idx + 1)]
            sign = 1 if bit == '0' else -1
            expectation += sign * count
        
        return expectation / shots, result.mitigation_applied


def apply_mitigation(
    counts: Dict[str, int],
    shots: int,
    n_qubits: int,
    mitigator: Optional[ReadoutMitigator] = None,
) -> Dict[str, float]:
    """Apply readout mitigation to counts.
    
    Convenience function that handles None mitigator.
    
    Args:
        counts: Raw measurement counts.
        shots: Total shots.
        n_qubits: Number of qubits.
        mitigator: Optional ReadoutMitigator.
    
    Returns:
        Mitigated counts (or original if no mitigator).
    """
    if mitigator is None:
        return {k: float(v) for k, v in counts.items()}
    
    result = mitigator.mitigate(counts, shots)
    return result.mitigated_counts
