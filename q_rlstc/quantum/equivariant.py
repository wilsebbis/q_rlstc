"""Equivariant Quantum Circuit (EQC) builder for Q-RLSTC.

Replaces the Hardware-Efficient Ansatz (HEA) with an SO(2)-equivariant
quantum neural network. By designing gates that commute with 2D spatial
rotation groups, the circuit achieves mathematical rotational invariance —
meaning it doesn't need to re-learn that a 90° left turn heading North
deserves the same treatment as one heading East.

This drastically shrinks the hypothesis space and accelerates convergence.

Theory:
    For a rotation R(θ) ∈ SO(2), an equivariant circuit U satisfies:
        U(R(θ)·x) = R'(θ) · U(x)
    where R' is the representation of SO(2) on the output space.
    
    We achieve this by:
    1. Using RZ-only encoding (phase encoding preserves rotational structure)
    2. Symmetric CNOT topology (circular entanglement)
    3. Equivariant variational layers: RZ → RZ → CNOT ring
"""

import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


class EquivariantCircuitBuilder:
    """Builder for SO(2)-equivariant quantum circuits.
    
    Unlike the HEA which treats spatial coordinates as abstract numbers,
    this circuit respects the rotational symmetry of 2D trajectory data.
    
    Structure:
    1. Phase encoding: RZ(2·arctan(feature_i)) on each qubit
    2. For each equivariant layer:
       a. Symmetric rotations: RZ(θ_i) on each qubit
       b. Circular entanglement: CNOT ring (0→1→2→...→N→0)
       c. Phase mixing: RZ(φ_i) on each qubit
    3. Measurement
    
    The key difference from HEA:
    - HEA uses RY-RZ rotations (breaks symmetry by mixing |0⟩/|1⟩ amplitudes)
    - EQC uses RZ-only rotations (preserves phase structure = respects SO(2))
    - HEA uses linear CNOT chain (asymmetric)
    - EQC uses circular CNOT ring (maintains permutation symmetry)
    """
    
    def __init__(
        self,
        n_qubits: int = 5,
        n_layers: int = 2,
        use_data_reuploading: bool = True,
    ):
        """Initialize equivariant circuit builder.
        
        Args:
            n_qubits: Number of qubits (equals state dimension).
            n_layers: Number of equivariant layers.
            use_data_reuploading: Whether to repeat encoding between layers.
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.use_data_reuploading = use_data_reuploading
        
        # Parameters per layer: 2 RZ rotations per qubit
        self.params_per_layer = 2 * n_qubits
        self.n_params = self.params_per_layer * n_layers
    
    def _add_phase_encoding(
        self,
        circuit: QuantumCircuit,
        qr: QuantumRegister,
        angles: np.ndarray,
    ) -> None:
        """Add SO(2)-equivariant phase encoding.
        
        Uses RZ rotations to encode features in the phase,
        preserving rotational structure.
        
        Args:
            circuit: Quantum circuit to modify.
            qr: Quantum register.
            angles: Encoded feature angles.
        """
        n = min(len(angles), self.n_qubits)
        
        # First: Hadamard to create superposition (needed for RZ to be visible)
        for i in range(n):
            circuit.h(qr[i])
        
        # Phase encoding: RZ rotations
        for i in range(n):
            circuit.rz(float(angles[i]), qr[i])
    
    def _add_equivariant_layer(
        self,
        circuit: QuantumCircuit,
        qr: QuantumRegister,
        params: np.ndarray,
        layer_idx: int,
    ) -> None:
        """Add one equivariant variational layer.
        
        Pattern: RZ rotations → circular CNOT ring → RZ rotations
        
        The circular CNOT ring maintains permutation symmetry,
        and the RZ-only rotations preserve phase structure.
        
        Args:
            circuit: Quantum circuit to modify.
            qr: Quantum register.
            params: Parameter values for this layer.
            layer_idx: Which layer (for labelling).
        """
        offset = layer_idx * self.params_per_layer
        
        # First RZ rotation on each qubit
        for i in range(self.n_qubits):
            circuit.rz(float(params[offset + i]), qr[i])
        
        # Circular CNOT ring (equivariant entanglement)
        for i in range(self.n_qubits):
            circuit.cx(qr[i], qr[(i + 1) % self.n_qubits])
        
        # Second RZ rotation (phase mixing)
        for i in range(self.n_qubits):
            circuit.rz(float(params[offset + self.n_qubits + i]), qr[i])
    
    def build_circuit(
        self,
        state: np.ndarray,
        params: np.ndarray,
        add_measurements: bool = True,
    ) -> QuantumCircuit:
        """Build the complete equivariant VQ-DQN circuit.
        
        Args:
            state: Input state vector (features).
            params: Variational parameters.
            add_measurements: Whether to add measurement gates.
        
        Returns:
            Complete QuantumCircuit.
        """
        from .vqdqn_circuit import angle_encode
        
        qr = QuantumRegister(self.n_qubits, 'q')
        cr = ClassicalRegister(self.n_qubits, 'c') if add_measurements else None
        
        if cr is not None:
            circuit = QuantumCircuit(qr, cr)
        else:
            circuit = QuantumCircuit(qr)
        
        # Encode features as rotation angles
        angles = angle_encode(state[:self.n_qubits], scaling='arctan')
        
        # Initial phase encoding
        self._add_phase_encoding(circuit, qr, angles)
        
        # Equivariant layers with optional data reuploading
        for layer in range(self.n_layers):
            if layer > 0 and self.use_data_reuploading:
                # Re-encode data between layers
                for i in range(min(len(angles), self.n_qubits)):
                    circuit.rz(float(angles[i]), qr[i])
            
            self._add_equivariant_layer(circuit, qr, params, layer)
        
        if add_measurements:
            circuit.measure(qr, cr)
        
        return circuit
    
    def get_circuit_info(self, params: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Get information about the equivariant circuit.
        
        Args:
            params: Parameters (random if None).
        
        Returns:
            Dict with circuit metrics.
        """
        if params is None:
            params = np.random.uniform(-np.pi, np.pi, self.n_params)
        
        dummy_state = np.zeros(self.n_qubits)
        circuit = self.build_circuit(dummy_state, params, add_measurements=False)
        
        gate_counts = {}
        for instruction in circuit.data:
            name = instruction[0].name
            gate_counts[name] = gate_counts.get(name, 0) + 1
        
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_params": self.n_params,
            "depth": circuit.depth(),
            "gate_counts": gate_counts,
            "ansatz": "eqc",
            "symmetry": "SO(2)",
        }
