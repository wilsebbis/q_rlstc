"""Backend factory for quantum circuit execution.

Provides backends for:
- Ideal (noiseless) simulation
- Noisy simulation with IBM device-like noise models
- IBM Runtime (placeholder for real hardware)
"""

from typing import Literal, Optional
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)


class BackendFactory:
    """Factory for creating quantum backends.
    
    Supports ideal simulation, noisy simulation with various noise models,
    and placeholder for IBM Runtime integration.
    """
    
    @staticmethod
    def get_ideal_backend() -> AerSimulator:
        """Get a noiseless quantum simulator backend."""
        return AerSimulator(method='statevector')
    
    @staticmethod
    def get_simple_noise_model(
        single_qubit_error: float = 0.001,
        two_qubit_error: float = 0.01,
        readout_error: float = 0.02,
    ) -> NoiseModel:
        """Create simple depolarizing noise model.
        
        Args:
            single_qubit_error: Error rate for 1Q gates.
            two_qubit_error: Error rate for 2Q gates.
            readout_error: Measurement bit-flip probability.
        
        Returns:
            Configured NoiseModel.
        """
        noise_model = NoiseModel()
        
        # Single-qubit gate errors
        error_1q = depolarizing_error(single_qubit_error, 1)
        noise_model.add_all_qubit_quantum_error(
            error_1q, ['ry', 'rz', 'rx', 'h', 'x', 'y', 'z', 'u']
        )
        
        # Two-qubit gate errors
        error_2q = depolarizing_error(two_qubit_error, 2)
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz', 'swap'])
        
        # Readout errors
        if readout_error > 0:
            read_error = ReadoutError([
                [1 - readout_error, readout_error],
                [readout_error, 1 - readout_error]
            ])
            noise_model.add_all_qubit_readout_error(read_error)
        
        return noise_model
    
    @staticmethod
    def get_thermal_noise_model(
        t1: float = 50e-6,
        t2: float = 70e-6,
        gate_time_1q: float = 50e-9,
        gate_time_2q: float = 300e-9,
        single_qubit_error: float = 0.001,
        two_qubit_error: float = 0.01,
        readout_error: float = 0.02,
    ) -> NoiseModel:
        """Create thermal relaxation noise model.
        
        Combines depolarizing errors with T1/T2 relaxation.
        
        Args:
            t1: T1 relaxation time (seconds).
            t2: T2 dephasing time (seconds).
            gate_time_1q: Single-qubit gate duration.
            gate_time_2q: Two-qubit gate duration.
            single_qubit_error: Additional depolarizing for 1Q.
            two_qubit_error: Additional depolarizing for 2Q.
            readout_error: Measurement error probability.
        
        Returns:
            NoiseModel with thermal relaxation.
        """
        noise_model = NoiseModel()
        
        # Thermal relaxation for 1Q gates
        thermal_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
        depol_1q = depolarizing_error(single_qubit_error, 1)
        combined_1q = thermal_1q.compose(depol_1q)
        noise_model.add_all_qubit_quantum_error(combined_1q, ['ry', 'rz', 'rx', 'h', 'u'])
        
        # Thermal relaxation for 2Q gates
        thermal_2q = thermal_relaxation_error(t1, t2, gate_time_2q).tensor(
            thermal_relaxation_error(t1, t2, gate_time_2q)
        )
        depol_2q = depolarizing_error(two_qubit_error, 2)
        combined_2q = thermal_2q.compose(depol_2q)
        noise_model.add_all_qubit_quantum_error(combined_2q, ['cx', 'cz'])
        
        # Readout errors
        if readout_error > 0:
            read_error = ReadoutError([
                [1 - readout_error, readout_error],
                [readout_error, 1 - readout_error]
            ])
            noise_model.add_all_qubit_readout_error(read_error)
        
        return noise_model
    
    @staticmethod
    def get_ibm_eagle_noise_model() -> NoiseModel:
        """Get noise model approximating IBM Eagle processor.
        
        Based on typical IBM Eagle (127 qubit) error rates.
        """
        return BackendFactory.get_thermal_noise_model(
            t1=300e-6,
            t2=150e-6,
            gate_time_1q=56e-9,
            gate_time_2q=400e-9,
            single_qubit_error=0.0005,
            two_qubit_error=0.008,
            readout_error=0.01,
        )
    
    @staticmethod
    def get_ibm_heron_noise_model() -> NoiseModel:
        """Get noise model approximating IBM Heron processor.
        
        Based on IBM Heron specifications (lower error rates).
        """
        return BackendFactory.get_thermal_noise_model(
            t1=400e-6,
            t2=200e-6,
            gate_time_1q=40e-9,
            gate_time_2q=100e-9,
            single_qubit_error=0.0002,
            two_qubit_error=0.002,
            readout_error=0.005,
        )
    
    @staticmethod
    def get_noisy_backend(noise_model: NoiseModel) -> AerSimulator:
        """Get a noisy simulator backend.
        
        Args:
            noise_model: NoiseModel to apply.
        
        Returns:
            AerSimulator with noise.
        """
        return AerSimulator(noise_model=noise_model)
    
    @classmethod
    def get_noise_model_by_name(
        cls,
        name: Literal["ideal", "simple", "thermal", "eagle", "heron"],
    ) -> Optional[NoiseModel]:
        """Get noise model by name.
        
        Args:
            name: Noise model name.
        
        Returns:
            NoiseModel or None for ideal.
        """
        models = {
            'ideal': None,
            'simple': cls.get_simple_noise_model,
            'thermal': cls.get_thermal_noise_model,
            'eagle': cls.get_ibm_eagle_noise_model,
            'heron': cls.get_ibm_heron_noise_model,
        }
        
        if name not in models:
            raise ValueError(f"Unknown noise model: {name}")
        
        factory = models[name]
        return factory() if factory else None


def get_backend(
    mode: Literal["ideal", "noisy_sim", "ibm_runtime"] = "ideal",
    noise_model_name: Literal["simple", "thermal", "eagle", "heron"] = "eagle",
    device_name: Optional[str] = None,
) -> AerSimulator:
    """Get a quantum backend for circuit execution.
    
    Args:
        mode: Backend mode.
            'ideal': Noiseless simulation.
            'noisy_sim': Noisy simulation with specified model.
            'ibm_runtime': Placeholder for real IBM hardware.
        noise_model_name: Noise model for noisy_sim mode.
        device_name: Device name for IBM runtime (not yet implemented).
    
    Returns:
        Configured backend.
    
    Raises:
        NotImplementedError: For ibm_runtime mode.
    """
    factory = BackendFactory()
    
    if mode == "ideal":
        return factory.get_ideal_backend()
    
    elif mode == "noisy_sim":
        noise_model = factory.get_noise_model_by_name(noise_model_name)
        if noise_model is None:
            return factory.get_ideal_backend()
        return factory.get_noisy_backend(noise_model)
    
    elif mode == "ibm_runtime":
        raise NotImplementedError(
            "IBM Runtime integration not yet implemented. "
            "To use real hardware, integrate with qiskit-ibm-runtime."
        )
    
    else:
        raise ValueError(f"Unknown backend mode: {mode}")
