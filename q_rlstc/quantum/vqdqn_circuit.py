"""VQ-DQN circuit builder with angle encoding and HEA.

Implements the Variational Quantum Deep Q-Network circuit:
- 5 qubits (one per state feature)
- Angle encoding: RY(2*arctan(feature_i)) on qubit i
- Hardware-Efficient Ansatz (HEA) with depth=2
- Linear entanglement: CNOT chain (0→1→2→3→4)
- Data re-uploading: encoding repeated between variational layers
- Output: Z-expectations on qubits 0,1 mapped to Q-values via linear head
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector


def angle_encode(features: np.ndarray, scaling: str = 'arctan') -> np.ndarray:
    """Encode classical features as rotation angles.
    
    Maps feature values to angles in [-π, π] for RY rotations.
    
    Args:
        features: 1D array of feature values.
        scaling: Scaling method.
            'arctan': θ = 2 * arctan(x), maps (-∞, ∞) → (-π, π)
            'linear': θ = π * x (assumes x ∈ [-1, 1])
            'sigmoid': θ = π * (2 * sigmoid(x) - 1)
    
    Returns:
        Array of rotation angles.
    """
    features = np.asarray(features).flatten()
    
    if scaling == 'arctan':
        # Most robust for unbounded features
        angles = 2 * np.arctan(features)
    elif scaling == 'linear':
        # Direct mapping, assumes normalized features
        angles = np.pi * np.clip(features, -1, 1)
    elif scaling == 'sigmoid':
        # Smooth mapping via sigmoid
        sigmoid = 1 / (1 + np.exp(-features))
        angles = np.pi * (2 * sigmoid - 1)
    else:
        raise ValueError(f"Unknown scaling method: {scaling}")
    
    return angles


@dataclass
class CircuitInfo:
    """Information about a VQ-DQN circuit.
    
    Attributes:
        n_qubits: Number of qubits.
        n_layers: Number of variational layers.
        n_params: Total trainable parameters.
        depth: Circuit depth.
        gate_counts: Dictionary of gate type to count.
    """
    n_qubits: int
    n_layers: int
    n_params: int
    depth: int
    gate_counts: Dict[str, int]


class VQDQNCircuitBuilder:
    """Builder for VQ-DQN quantum circuits.
    
    Constructs circuits with:
    - Angle encoding layer (RY rotations from state features)
    - Variational layers with RY-RZ rotations (2 per qubit)
    - Linear CNOT chain for entanglement
    - Optional data re-uploading between layers
    
    Parameter count: n_qubits * 2 rotations * n_layers = 5 * 2 * 2 = 20
    """
    
    def __init__(
        self,
        n_qubits: int = 5,
        n_layers: int = 2,
        use_data_reuploading: bool = True,
        entanglement: str = 'linear',
    ):
        """Initialize circuit builder.
        
        Args:
            n_qubits: Number of qubits (equals state dimension).
            n_layers: Number of variational layers.
            use_data_reuploading: Whether to repeat encoding between layers.
            entanglement: Entanglement pattern ('linear', 'circular', 'full').
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.use_data_reuploading = use_data_reuploading
        self.entanglement = entanglement
        
        # Parameter count: 2 rotations (RY, RZ) per qubit per layer
        self.n_params_per_layer = n_qubits * 2
        self.n_total_params = self.n_params_per_layer * n_layers
    
    def _add_encoding_layer(
        self, 
        circuit: QuantumCircuit, 
        qr: QuantumRegister, 
        angles: np.ndarray
    ) -> None:
        """Add angle encoding layer.
        
        Args:
            circuit: Quantum circuit to modify.
            qr: Quantum register.
            angles: Encoded feature angles.
        """
        for i in range(self.n_qubits):
            circuit.ry(float(angles[i]), qr[i])
    
    def _add_variational_layer(
        self,
        circuit: QuantumCircuit,
        qr: QuantumRegister,
        params: np.ndarray,
        layer_idx: int,
    ) -> None:
        """Add one variational layer.
        
        Pattern: RY-RZ rotations followed by CNOT chain.
        (2 rotations per qubit, not 3)
        
        Args:
            circuit: Quantum circuit to modify.
            qr: Quantum register.
            params: All parameters (will extract relevant slice).
            layer_idx: Which layer (0-indexed).
        """
        param_offset = layer_idx * self.n_params_per_layer
        
        # RY-RZ rotations on each qubit (2 per qubit)
        for i in range(self.n_qubits):
            base_idx = param_offset + i * 2
            circuit.ry(float(params[base_idx]), qr[i])
            circuit.rz(float(params[base_idx + 1]), qr[i])
        
        # Entanglement layer
        if self.entanglement == 'linear':
            # Linear chain: 0→1→2→3→4 (NO ring closure)
            for i in range(self.n_qubits - 1):
                circuit.cx(qr[i], qr[i + 1])
        elif self.entanglement == 'circular':
            # Circular: 0→1→2→3→4→0
            for i in range(self.n_qubits):
                circuit.cx(qr[i], qr[(i + 1) % self.n_qubits])
        elif self.entanglement == 'full':
            # All-to-all
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    circuit.cx(qr[i], qr[j])
    
    def build_circuit(
        self,
        state: np.ndarray,
        params: np.ndarray,
        add_measurements: bool = True,
    ) -> QuantumCircuit:
        """Build the complete VQ-DQN circuit.
        
        Structure:
        1. Angle encoding (RY rotations from state)
        2. For each layer:
           a. Variational rotations (RY-RZ-RY)
           b. Entanglement (CNOT chain)
           c. Data re-uploading (if enabled, except last layer)
        3. Measurement
        
        Args:
            state: Input state vector [n_qubits features].
            params: Variational parameters [n_total_params].
            add_measurements: Whether to add measurement gates.
        
        Returns:
            Complete QuantumCircuit.
        """
        state = np.asarray(state).flatten()
        params = np.asarray(params).flatten()
        
        if len(state) != self.n_qubits:
            raise ValueError(f"State dimension {len(state)} != n_qubits {self.n_qubits}")
        if len(params) != self.n_total_params:
            raise ValueError(f"Param count {len(params)} != n_total_params {self.n_total_params}")
        
        # Create circuit
        qr = QuantumRegister(self.n_qubits, 'q')
        circuit = QuantumCircuit(qr)
        
        if add_measurements:
            cr = ClassicalRegister(self.n_qubits, 'c')
            circuit.add_register(cr)
        
        # Encode features
        angles = angle_encode(state, scaling='arctan')
        self._add_encoding_layer(circuit, qr, angles)
        
        # Variational layers
        for layer in range(self.n_layers):
            self._add_variational_layer(circuit, qr, params, layer)
            
            # Data re-uploading: repeat encoding between layers (not after last)
            if self.use_data_reuploading and layer < self.n_layers - 1:
                self._add_encoding_layer(circuit, qr, angles)
        
        # Measurement
        if add_measurements:
            circuit.measure(qr, cr)
        
        return circuit
    
    def get_circuit_info(self, params: Optional[np.ndarray] = None) -> CircuitInfo:
        """Get information about the circuit structure.
        
        Args:
            params: Parameters to use (random if None).
        
        Returns:
            CircuitInfo with metrics.
        """
        if params is None:
            params = np.random.uniform(-np.pi, np.pi, self.n_total_params)
        
        state = np.zeros(self.n_qubits)
        circuit = self.build_circuit(state, params, add_measurements=False)
        
        return CircuitInfo(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            n_params=self.n_total_params,
            depth=circuit.depth(),
            gate_counts=dict(circuit.count_ops()),
        )


def build_vqdqn_circuit(
    state: np.ndarray,
    params: np.ndarray,
    n_qubits: int = 5,
    n_layers: int = 2,
    use_data_reuploading: bool = True,
    add_measurements: bool = True,
) -> QuantumCircuit:
    """Build a VQ-DQN circuit.
    
    Convenience function that creates a builder and builds a circuit.
    
    Args:
        state: Input state vector.
        params: Variational parameters.
        n_qubits: Number of qubits.
        n_layers: Number of variational layers.
        use_data_reuploading: Whether to repeat encoding.
        add_measurements: Whether to add measurements.
    
    Returns:
        Complete QuantumCircuit.
    """
    builder = VQDQNCircuitBuilder(
        n_qubits=n_qubits,
        n_layers=n_layers,
        use_data_reuploading=use_data_reuploading,
    )
    return builder.build_circuit(state, params, add_measurements)


def compute_expectation_from_counts(
    counts: Dict[str, int],
    shots: int,
    qubit_idx: int,
    n_qubits: int,
) -> float:
    """Compute Pauli-Z expectation from measurement counts.
    
    Args:
        counts: Measurement counts dict (e.g., {'00000': 512, '00001': 512}).
        shots: Total number of shots.
        qubit_idx: Which qubit's expectation to compute.
        n_qubits: Total number of qubits.
    
    Returns:
        Expectation value in [-1, 1].
    """
    expectation = 0.0
    
    for bitstring, count in counts.items():
        # Qiskit uses little-endian ordering
        # Pad bitstring to correct length
        padded = bitstring.zfill(n_qubits)
        bit = padded[-(qubit_idx + 1)]
        sign = 1 if bit == '0' else -1
        expectation += sign * count
    
    return expectation / shots


def compute_parity_expectation(
    counts: Dict[str, int],
    shots: int,
    qubit_a: int,
    qubit_b: int,
    n_qubits: int,
) -> float:
    """Compute two-qubit parity expectation ⟨ZₐZ_b⟩ from measurement counts.
    
    Parity is +1 when both qubits have same value, -1 when different.
    This captures correlations between qubit measurements that individual
    ⟨Z⟩ cannot — a uniquely quantum-useful observable.
    
    Args:
        counts: Measurement counts dict.
        shots: Total number of shots.
        qubit_a: First qubit index.
        qubit_b: Second qubit index.
        n_qubits: Total number of qubits.
    
    Returns:
        Parity expectation in [-1, 1].
    """
    expectation = 0.0
    
    for bitstring, count in counts.items():
        padded = bitstring.zfill(n_qubits)
        bit_a = padded[-(qubit_a + 1)]
        bit_b = padded[-(qubit_b + 1)]
        # Same value → parity +1, different → parity -1
        sign = 1 if bit_a == bit_b else -1
        expectation += sign * count
    
    return expectation / shots


def evaluate_q_values(
    state: np.ndarray,
    params: np.ndarray,
    backend,
    shots: int = 1024,
    n_qubits: int = 5,
    n_layers: int = 2,
    use_data_reuploading: bool = True,
    output_scale: np.ndarray = None,
    output_bias: np.ndarray = None,
    readout_mode: str = "standard",
) -> np.ndarray:
    """Evaluate Q-values by executing the VQ-DQN circuit.
    
    Supports two readout modes:
    - "standard" (Version A): Q-values from ⟨Z₀⟩ and ⟨Z₁⟩
    - "multi_observable" (Version B): Q-values from linear combination
      of single-qubit and parity expectations:
        Q(extend) = w₀·⟨Z₀⟩ + w₂·⟨Z₂Z₃⟩ + b₀
        Q(cut)    = w₁·⟨Z₁⟩ + w₃·⟨Z₄Z₅⟩ + b₁
    
    Args:
        state: Input state vector.
        params: Variational parameters.
        backend: Qiskit backend for execution.
        shots: Number of measurement shots.
        n_qubits: Number of qubits.
        n_layers: Number of layers.
        use_data_reuploading: Whether to use re-uploading.
        output_scale: Scale factors.
            Standard: shape (2,) — one per action.
            Multi-observable: shape (4,) — [w₀, w₁, w₂, w₃].
        output_bias: Bias terms, shape (2,).
        readout_mode: "standard" or "multi_observable".
    
    Returns:
        Array of Q-values [Q(s, extend), Q(s, cut)].
    """
    from qiskit import transpile
    
    if output_scale is None:
        n_scale = 4 if readout_mode == "multi_observable" else 2
        output_scale = np.ones(n_scale)
    if output_bias is None:
        output_bias = np.zeros(2)
    
    # Build circuit
    circuit = build_vqdqn_circuit(
        state, params, n_qubits, n_layers, use_data_reuploading, add_measurements=True
    )
    
    # Execute
    transpiled = transpile(circuit, backend)
    job = backend.run(transpiled, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    if readout_mode == "multi_observable":
        # Version B: combine single-qubit and parity expectations
        z0 = compute_expectation_from_counts(counts, shots, 0, n_qubits)
        z1 = compute_expectation_from_counts(counts, shots, 1, n_qubits)
        z2z3 = compute_parity_expectation(counts, shots, 2, 3, n_qubits)
        z4z5 = compute_parity_expectation(counts, shots, 4, 5, n_qubits)
        
        q_values = np.array([
            output_scale[0] * z0 + output_scale[2] * z2z3 + output_bias[0],
            output_scale[1] * z1 + output_scale[3] * z4z5 + output_bias[1],
        ])
    else:
        # Version A: standard readout from qubits 0 and 1
        q_values = np.zeros(2)
        for action in range(2):
            qubit_idx = action
            exp_val = compute_expectation_from_counts(counts, shots, qubit_idx, n_qubits)
            q_values[action] = exp_val * output_scale[action] + output_bias[action]
    
    return q_values
