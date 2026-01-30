"""Tests for swap test distance estimation."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from q_rlstc.quantum.swaptest_distance import (
    swaptest_distance,
    classical_distance,
    normalize_and_pad,
    SwapTestDistanceEstimator,
)


class TestNormalizePad:
    """Tests for normalization and padding."""
    
    def test_normalize_preserves_direction(self):
        """Normalization preserves vector direction"""
        vector = np.array([3.0, 4.0])
        padded, norm, n_qubits = normalize_and_pad(vector)
        
        # Original norm should be 5
        assert abs(norm - 5.0) < 1e-10
        
        # Should pad to power of 2
        assert len(padded) == 2 ** n_qubits
        assert n_qubits >= 1
    
    def test_zero_vector(self):
        """Zero vector handled gracefully"""
        vector = np.zeros(4)
        padded, norm, n_qubits = normalize_and_pad(vector)
        
        assert norm == 1.0  # Default norm for zero
        assert len(padded) == 4
    
    def test_padding_to_power_of_2(self):
        """Vectors padded to power of 2 length"""
        for n in [3, 5, 7, 9]:
            vector = np.random.randn(n)
            padded, _, n_qubits = normalize_and_pad(vector)
            assert len(padded) == 2 ** n_qubits
            assert 2 ** n_qubits >= n


class TestClassicalDistance:
    """Tests for classical distance baseline."""
    
    def test_same_vector_zero_distance(self):
        """Distance from vector to itself is 0"""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        dist = classical_distance(x, x)
        assert abs(dist) < 1e-10
    
    def test_orthogonal_vectors(self):
        """Known distance between orthogonal vectors"""
        x = np.array([1.0, 0.0])
        y = np.array([0.0, 1.0])
        dist = classical_distance(x, y)
        expected = np.sqrt(2)
        assert abs(dist - expected) < 1e-10
    
    def test_different_lengths(self):
        """Handles vectors of different lengths"""
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0, 3.0])
        dist = classical_distance(x, y)
        # y is padded, distance is |3| = 3
        assert dist > 0


class TestSwapTestDistance:
    """Tests for quantum swap test distance."""
    
    @pytest.fixture
    def estimator(self):
        """Create estimator with high shot count for accuracy."""
        return SwapTestDistanceEstimator(default_shots=2048)
    
    def test_same_vector_small_distance(self, estimator):
        """Distance from vector to itself should be near 0"""
        x = np.array([1.0, 0.0, 0.0, 0.0])
        result = estimator.estimate_distance(x, x)
        
        # Should be very small (not exactly 0 due to shot noise)
        assert result.distance < 0.3
    
    def test_orthogonal_larger_distance(self, estimator):
        """Orthogonal vectors should have larger distance"""
        x = np.array([1.0, 0.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0, 0.0])
        
        result_same = estimator.estimate_distance(x, x)
        result_ortho = estimator.estimate_distance(x, y)
        
        # Orthogonal should be farther
        assert result_ortho.distance > result_same.distance
    
    def test_convenience_function(self):
        """swaptest_distance convenience function works"""
        x = np.array([1.0, 2.0, 0.0, 0.0])
        y = np.array([1.0, 2.0, 0.0, 0.0])
        
        dist = swaptest_distance(x, y, shots=1024)
        
        # Should be close-ish to 0 for same vectors
        assert dist < 0.5


class TestSwapTestVsClassical:
    """Compare quantum and classical distance estimates."""
    
    def test_approximate_agreement(self):
        """Quantum distance roughly agrees with classical"""
        # Use simple test case
        x = np.array([1.0, 0.5, 0.0, 0.0])
        y = np.array([0.5, 1.0, 0.0, 0.0])
        
        classical_dist = classical_distance(x, y)
        quantum_dist = swaptest_distance(x, y, shots=4096)
        
        # Should be in the same ballpark (within 50% due to shot noise)
        ratio = quantum_dist / max(classical_dist, 0.01)
        assert 0.2 < ratio < 5.0, f"Ratio {ratio} out of expected range"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
