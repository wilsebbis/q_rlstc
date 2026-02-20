"""Shadow Qubit Register for Quantum Recurrent Memory (Q-RNN).

Implements a temporal memory mechanism by preserving one qubit's state
across time steps. Instead of measuring and resetting all qubits at
every step, the shadow qubit's quantum state is carried over and
entangled with new spatial features, creating a Q-RNN effect.

This grants the agent latent temporal memory — allowing it to track
the sequential "rhythm" of a trajectory — without heavy classical
RNN/LSTM layers that would destroy parameter efficiency.

Usage:
    The shadow qubit is reserved as qubit 0. Data qubits shift to 1..N.
    Between steps, the shadow qubit's statevector is preserved and
    re-initialised at the start of the next circuit evaluation.
"""

import numpy as np
from typing import Optional, Tuple

from qiskit import QuantumCircuit, QuantumRegister


class ShadowQubitRegister:
    """Manages carry-over state for the shadow qubit (Q-RNN memory).
    
    The shadow qubit creates temporal entanglement by:
    1. Initialising qubit 0 with the previous step's statevector
    2. Entangling it with fresh data qubits via CNOT
    3. After measurement, extracting the shadow qubit's state for next step
    
    Attributes:
        n_data_qubits: Number of data qubits (excluding shadow qubit).
        shadow_state: 2D statevector [α, β] for the shadow qubit.
    """
    
    def __init__(self, n_data_qubits: int):
        """Initialize shadow qubit register.
        
        Args:
            n_data_qubits: Number of data qubits (the shadow qubit
                is additional, so total qubits = n_data_qubits + 1).
        """
        self.n_data_qubits = n_data_qubits
        # Start in |0⟩
        self.shadow_state = np.array([1.0, 0.0], dtype=complex)
    
    @property
    def total_qubits(self) -> int:
        """Total qubits including shadow qubit."""
        return self.n_data_qubits + 1
    
    def prepare_shadow_qubit(self, circuit: QuantumCircuit) -> None:
        """Initialise qubit 0 with the carried-over shadow state.
        
        This should be called BEFORE angle encoding on data qubits.
        
        Args:
            circuit: Quantum circuit to modify in-place.
        """
        # Normalise the shadow state (numerical safety)
        norm = np.linalg.norm(self.shadow_state)
        if norm > 1e-10:
            state = self.shadow_state / norm
        else:
            state = np.array([1.0, 0.0], dtype=complex)
        
        # Initialize qubit 0 with the shadow state
        circuit.initialize(state.tolist(), [0])
    
    def add_entanglement(self, circuit: QuantumCircuit) -> None:
        """Entangle shadow qubit with all data qubits.
        
        Applies CNOT(shadow → data_i) for each data qubit,
        creating temporal-spatial interference.
        
        Args:
            circuit: Quantum circuit to modify in-place.
        """
        for i in range(1, self.total_qubits):
            circuit.cx(0, i)
    
    def update_from_counts(
        self,
        counts: dict,
        n_qubits: int,
    ) -> None:
        """Update shadow state from measurement counts.
        
        Estimates the shadow qubit's marginal state from the
        measurement distribution. The probability of measuring |1⟩
        on qubit 0 gives us the new α and β.
        
        Args:
            counts: Measurement counts dictionary.
            n_qubits: Total number of qubits in the circuit.
        """
        total_shots = sum(counts.values())
        if total_shots == 0:
            return
        
        # Count how often shadow qubit (qubit 0) was measured as |1⟩
        # Qiskit bit ordering: rightmost bit = qubit 0
        p1 = 0
        for bitstring, count in counts.items():
            if len(bitstring) >= 1 and bitstring[-1] == '1':
                p1 += count
        
        p1 /= total_shots
        p0 = 1 - p1
        
        # Reconstruct approximate state (phase information is lost,
        # but the amplitude carries temporal signal)
        alpha = np.sqrt(max(p0, 0))
        beta = np.sqrt(max(p1, 0))
        self.shadow_state = np.array([alpha, beta], dtype=complex)
    
    def reset(self) -> None:
        """Reset shadow qubit to |0⟩ (start of new episode)."""
        self.shadow_state = np.array([1.0, 0.0], dtype=complex)
    
    def get_memory_signal(self) -> float:
        """Get scalar memory signal from shadow qubit.
        
        Returns the probability of measuring |1⟩, which encodes
        the temporal history. Range: [0, 1].
        
        Returns:
            P(|1⟩) for the shadow qubit.
        """
        return float(np.abs(self.shadow_state[1]) ** 2)
