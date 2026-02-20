"""Tests for angle encoding in VQ-DQN circuit."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from q_rlstc.quantum.vqdqn_circuit import (
    angle_encode,
    VQDQNCircuitBuilder,
    build_vqdqn_circuit,
)


class TestAngleEncode:
    """Tests for angle encoding function."""
    
    def test_arctan_scaling_zero(self):
        """arctan(0) = 0"""
        features = np.array([0.0])
        angles = angle_encode(features, scaling='arctan')
        assert abs(angles[0]) < 1e-10
    
    def test_arctan_scaling_positive(self):
        """arctan(positive) > 0"""
        features = np.array([1.0, 10.0, 100.0])
        angles = angle_encode(features, scaling='arctan')
        assert all(a > 0 for a in angles)
    
    def test_arctan_scaling_negative(self):
        """arctan(negative) < 0"""
        features = np.array([-1.0, -10.0, -100.0])
        angles = angle_encode(features, scaling='arctan')
        assert all(a < 0 for a in angles)
    
    def test_arctan_scaling_bounded(self):
        """arctan output bounded in [-π, π]"""
        features = np.array([-1e6, -100, 0, 100, 1e6])
        angles = angle_encode(features, scaling='arctan')
        assert all(abs(a) < np.pi for a in angles)
    
    def test_linear_scaling(self):
        """Linear scaling maps [-1, 1] to [-π, π]"""
        features = np.array([0, 0.5, -0.5, 1.0, -1.0])
        angles = angle_encode(features, scaling='linear')
        
        assert abs(angles[0]) < 1e-10
        assert abs(angles[1] - np.pi * 0.5) < 1e-10
        assert abs(angles[2] + np.pi * 0.5) < 1e-10
    
    def test_output_shape(self):
        """Output has same shape as input"""
        for n in [1, 3, 5, 10]:
            features = np.random.randn(n)
            angles = angle_encode(features)
            assert len(angles) == n


class TestVQDQNCircuit:
    """Tests for VQ-DQN circuit builder."""
    
    def test_circuit_uses_5_qubits(self):
        """Default circuit uses 5 qubits"""
        state = np.zeros(5)
        params = np.random.randn(20)  # 5 * 2 * 2
        
        circuit = build_vqdqn_circuit(state, params)
        
        assert circuit.num_qubits == 5
    
    def test_circuit_has_ry_gates(self):
        """Circuit contains RY gates for encoding"""
        builder = VQDQNCircuitBuilder(n_qubits=5, n_layers=2)
        state = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        params = np.random.randn(builder.n_total_params)
        
        circuit = builder.build_circuit(state, params, add_measurements=False)
        ops = circuit.count_ops()
        
        assert 'ry' in ops
        assert ops['ry'] > 0
    
    def test_circuit_has_measurements(self):
        """Circuit with measurements has correct count"""
        state = np.zeros(5)
        params = np.random.randn(20)
        
        circuit = build_vqdqn_circuit(state, params, add_measurements=True)
        ops = circuit.count_ops()
        
        assert 'measure' in ops
        assert ops['measure'] == 5
    
    def test_parameter_count(self):
        """Correct number of parameters for layers"""
        for n_layers in [1, 2, 3]:
            builder = VQDQNCircuitBuilder(n_qubits=5, n_layers=n_layers)
            expected = 5 * 2 * n_layers  # 2 rotations per qubit per layer
            assert builder.n_total_params == expected


class TestCircuitInfo:
    """Tests for circuit info extraction."""
    
    def test_circuit_info_values(self):
        """Circuit info has expected values"""
        builder = VQDQNCircuitBuilder(n_qubits=5, n_layers=2)
        info = builder.get_circuit_info()
        
        assert info.n_qubits == 5
        assert info.n_layers == 2
        assert info.n_params == 20
        assert info.depth > 0
        assert 'ry' in info.gate_counts


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
