"""Tests for HEA circuit structure and depth."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from q_rlstc.quantum.vqdqn_circuit import (
    VQDQNCircuitBuilder,
    build_vqdqn_circuit,
)


class TestHEADepth:
    """Tests for Hardware-Efficient Ansatz depth."""
    
    def test_two_layers_depth(self):
        """Depth=2 HEA has expected structure"""
        builder = VQDQNCircuitBuilder(n_qubits=5, n_layers=2)
        state = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        params = np.random.randn(builder.n_total_params)
        
        circuit = builder.build_circuit(state, params, add_measurements=False)
        
        # Should have reasonable depth for 2 layers
        assert circuit.depth() > 0
        assert builder.n_layers == 2
    
    def test_cnot_chain_exists(self):
        """Circuit has CNOT gates for entanglement"""
        builder = VQDQNCircuitBuilder(n_qubits=5, n_layers=2)
        state = np.zeros(5)
        params = np.random.randn(builder.n_total_params)
        
        circuit = builder.build_circuit(state, params, add_measurements=False)
        ops = circuit.count_ops()
        
        assert 'cx' in ops
        # Linear entanglement: 4 CNOTs per layer (0->1, 1->2, 2->3, 3->4)
        assert ops['cx'] >= 4  # At least from first layer
    
    def test_data_reuploading_layers(self):
        """Data re-uploading creates additional encoding layers"""
        builder_with = VQDQNCircuitBuilder(
            n_qubits=5, n_layers=2, use_data_reuploading=True
        )
        builder_without = VQDQNCircuitBuilder(
            n_qubits=5, n_layers=2, use_data_reuploading=False
        )
        
        state = np.zeros(5)
        params_with = np.random.randn(builder_with.n_total_params)
        params_without = np.random.randn(builder_without.n_total_params)
        
        circuit_with = builder_with.build_circuit(state, params_with, add_measurements=False)
        circuit_without = builder_without.build_circuit(state, params_without, add_measurements=False)
        
        # With re-uploading should have more RY gates
        ops_with = circuit_with.count_ops()
        ops_without = circuit_without.count_ops()
        
        # With data re-uploading adds 5 extra RY gates between layers
        assert ops_with.get('ry', 0) > ops_without.get('ry', 0)
    
    def test_variational_rotation_types(self):
        """Circuit uses RY-RZ-RY rotation sequence"""
        builder = VQDQNCircuitBuilder(n_qubits=5, n_layers=1)
        state = np.zeros(5)
        params = np.random.randn(builder.n_total_params)
        
        circuit = builder.build_circuit(state, params, add_measurements=False)
        ops = circuit.count_ops()
        
        assert 'ry' in ops
        assert 'rz' in ops
    
    def test_layer_scaling(self):
        """More layers = more gates"""
        state = np.zeros(5)
        
        builder_1 = VQDQNCircuitBuilder(n_qubits=5, n_layers=1, use_data_reuploading=False)
        builder_3 = VQDQNCircuitBuilder(n_qubits=5, n_layers=3, use_data_reuploading=False)
        
        circuit_1 = builder_1.build_circuit(
            state, np.random.randn(builder_1.n_total_params), add_measurements=False
        )
        circuit_3 = builder_3.build_circuit(
            state, np.random.randn(builder_3.n_total_params), add_measurements=False
        )
        
        # More layers should have more depth
        assert circuit_3.depth() > circuit_1.depth()
        
        # More parameters
        assert builder_3.n_total_params == 3 * builder_1.n_total_params


class TestCircularEntanglement:
    """Test circular entanglement option."""
    
    def test_circular_entanglement(self):
        """Circular entanglement creates ring of CNOTs"""
        builder = VQDQNCircuitBuilder(
            n_qubits=5, n_layers=1, entanglement='circular'
        )
        state = np.zeros(5)
        params = np.random.randn(builder.n_total_params)
        
        circuit = builder.build_circuit(state, params, add_measurements=False)
        ops = circuit.count_ops()
        
        # Circular should have 5 CNOTs (includes 4->0)
        assert ops.get('cx', 0) >= 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
