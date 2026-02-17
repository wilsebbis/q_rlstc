"""Configuration dataclasses for Q-RLSTC."""

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional, Dict, Any


@dataclass
class VQDQNConfig:
    """Configuration for the VQ-DQN quantum circuit.
    
    Attributes:
        n_qubits: Number of qubits (equals state dimension).
        n_layers: Number of variational layers in HEA.
        use_data_reuploading: Whether to repeat encoding between layers.
        shots_train: Measurement shots during training.
        shots_eval: Measurement shots during evaluation.
    """
    n_qubits: int = 5
    n_layers: int = 2
    use_data_reuploading: bool = True
    shots_train: int = 512   # Low precision, high speed for training
    shots_eval: int = 4096   # High precision for final metrics


@dataclass
class NoiseConfig:
    """Configuration for NISQ noise simulation.
    
    Attributes:
        use_noise: Whether to simulate noise.
        noise_model: Type of noise model to use.
        single_qubit_error: Depolarizing error rate for 1Q gates.
        two_qubit_error: Depolarizing error rate for 2Q gates.
        readout_error: Measurement error probability.
    """
    use_noise: bool = False
    noise_model: Literal["ideal", "simple", "thermal", "eagle", "heron"] = "ideal"
    single_qubit_error: float = 0.001
    two_qubit_error: float = 0.01
    readout_error: float = 0.02


@dataclass
class SPSAConfig:
    """Configuration for SPSA optimizer.
    
    Attributes:
        A: Stability constant for learning rate schedule.
        a: Initial learning rate scale.
        c: Initial perturbation size.
        alpha: Learning rate decay exponent.
        gamma_spsa: Perturbation decay exponent.
        seed: Random seed for reproducibility.
    """
    A: int = 20
    a: float = 0.12
    c: float = 0.10  # Larger perturbation to overcome shot noise variance
    alpha: float = 0.602
    gamma_spsa: float = 0.101
    seed: int = 42


@dataclass
class RLConfig:
    """Configuration for reinforcement learning.
    
    Attributes:
        gamma: Discount factor.
        epsilon_start: Initial exploration rate.
        epsilon_min: Minimum exploration rate.
        epsilon_decay: Decay rate per episode.
        batch_size: Minibatch size for replay.
        memory_size: Maximum replay buffer size.
        target_update_freq: Episodes between target network updates.
    """
    gamma: float = 0.90  # Shorter horizon; NISQ noise makes long-term credit unreliable
    epsilon_start: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay: float = 0.99
    batch_size: int = 32
    memory_size: int = 5000
    target_update_freq: int = 10


@dataclass
class ClusteringConfig:
    """Configuration for hybrid k-means clustering.
    
    Attributes:
        n_clusters: Number of clusters.
        max_iter: Maximum iterations for k-means.
        convergence_threshold: Centroid shift threshold for convergence.
        distance_mode: Distance computation mode.
        vector_dim: Dimension for amplitude encoding (must be power of 2).
    """
    n_clusters: int = 10
    max_iter: int = 50
    convergence_threshold: float = 0.01
    distance_mode: Literal["quantum", "classical"] = "quantum"
    vector_dim: int = 32


@dataclass
class TrainingConfig:
    """Configuration for the training loop.
    
    Attributes:
        n_epochs: Number of training epochs.
        n_trajectories: Number of trajectories to use.
        validation_split: Fraction of data for validation.
        checkpoint_freq: Episodes between checkpoint saves.
        output_dir: Directory for saving outputs.
    """
    n_epochs: int = 2
    n_trajectories: int = 100
    validation_split: float = 0.2
    checkpoint_freq: int = 50
    output_dir: str = "./outputs"


@dataclass
class QRLSTCConfig:
    """Master configuration for Q-RLSTC.
    
    Combines all sub-configurations into a single object.
    
    Attributes:
        version: "A" (direct comparison, 5 qubits) or "B" (quantum-optimized, 8 qubits).
    """
    version: str = "A"
    vqdqn: VQDQNConfig = field(default_factory=VQDQNConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    spsa: SPSAConfig = field(default_factory=SPSAConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def __post_init__(self):
        """Auto-adjust settings based on version."""
        if self.version.upper() == "B":
            self.vqdqn.n_qubits = 8
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to nested dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QRLSTCConfig":
        """Create config from dictionary."""
        return cls(
            vqdqn=VQDQNConfig(**d.get("vqdqn", {})),
            noise=NoiseConfig(**d.get("noise", {})),
            spsa=SPSAConfig(**d.get("spsa", {})),
            rl=RLConfig(**d.get("rl", {})),
            clustering=ClusteringConfig(**d.get("clustering", {})),
            training=TrainingConfig(**d.get("training", {})),
        )
