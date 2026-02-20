"""Configuration dataclasses for Q-RLSTC."""

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional, Dict, Any


# Human-readable version labels
VERSION_LABELS = {
    "A": "Classical Parity (5q)",
    "B": "Quantum Enhanced (8q)",
    "C": "Next-Gen Q-RNN (6q)",
    "D": "VLDB Aligned (5q)",
}


@dataclass
class VQDQNConfig:
    """Configuration for the VQ-DQN quantum circuit.
    
    Attributes:
        n_qubits: Number of qubits (equals state dimension).
        n_layers: Number of variational layers in HEA.
        use_data_reuploading: Whether to repeat encoding between layers.
        shots_train: Measurement shots during training.
        shots_eval: Measurement shots during evaluation.
        ansatz: Circuit ansatz — "hea" (default) or "eqc" (equivariant).
        use_shadow_qubit: Reserve qubit 0 as temporal memory (Q-RNN).
        adaptive_shots: Use adaptive shot allocation (burst → deep read).
        shots_burst: Low-cost burst shot count for confident decisions.
        confidence_threshold: Margin threshold to skip deep read.
    """
    n_qubits: int = 5
    n_layers: int = 2
    use_data_reuploading: bool = True
    shots_train: int = 512
    shots_eval: int = 4096
    ansatz: str = "hea"
    use_shadow_qubit: bool = False
    adaptive_shots: bool = False
    shots_burst: int = 32
    confidence_threshold: float = 0.3


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
        use_momentum: Enable momentum-SPSA (m-SPSA).
        momentum: Momentum coefficient for gradient averaging.
    """
    A: int = 20
    a: float = 0.12
    c: float = 0.10
    alpha: float = 0.602
    gamma_spsa: float = 0.101
    seed: int = 42
    use_momentum: bool = True
    momentum: float = 0.9


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
        n_actions: Action space size — 2=[EXTEND,CUT], 3=[EXTEND,CUT,DROP].
        drop_penalty: Micro-penalty for DROP action.
        separability_weight: Weight for cluster separability bonus.
        degeneracy_penalty: Penalty for single-cluster collapse.
        empty_cluster_penalty: Penalty for too few effective clusters.
        agent_type: "dqn" (default) or "sac" (Soft Actor-Critic).
        entropy_alpha: Entropy bonus coefficient for SAC.
        critic_lr: Learning rate for classical critic (SAC only).
        critic_hidden: Hidden layer size for critic MLP (SAC only).
    """
    gamma: float = 0.90
    epsilon_start: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay: float = 0.995
    batch_size: int = 32
    memory_size: int = 5000
    target_update_freq: int = 10
    n_actions: int = 2
    drop_penalty: float = 0.05
    separability_weight: float = 0.3
    degeneracy_penalty: float = 2.0
    empty_cluster_penalty: float = 1.0
    agent_type: str = "dqn"
    entropy_alpha: float = 0.2
    critic_lr: float = 0.001
    critic_hidden: int = 64
    skip_distance: int = 5


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
        version: "A" (Classical Parity, 5 qubits) or "B" (Quantum Enhanced, 8 qubits).
        compute_backend: Hardware acceleration — "auto", "cpu", "mlx", or "cuda".
    """
    version: str = "A"
    compute_backend: str = "auto"
    vqdqn: VQDQNConfig = field(default_factory=VQDQNConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    spsa: SPSAConfig = field(default_factory=SPSAConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def __post_init__(self):
        """Auto-adjust settings based on version."""
        version = self.version.upper()
        if version == "B":
            self.vqdqn.n_qubits = 8
        elif version == "C":
            # Version C: Next-Gen Q-RNN (6q)
            # 5 data qubits + 1 shadow qubit = 6 total
            self.vqdqn.n_qubits = 6
            self.vqdqn.ansatz = "eqc"
            self.vqdqn.use_shadow_qubit = True
            self.vqdqn.adaptive_shots = True
            self.rl.agent_type = "sac"
            self.rl.n_actions = 3  # EXTEND, CUT, DROP
            self.spsa.use_momentum = True
        elif version == "D":
            # Version D: VLDB 2024 Aligned (5q)
            # Exact paper state vector (OD_s, OD_n, OD_b, L_b, L_f)
            # Binary actions {EXTEND, CUT} — paper's exact MDP
            # Q-SKIP is an opt-in extension (set n_actions=3 + skip_distance)
            self.vqdqn.n_qubits = 5
            self.vqdqn.n_layers = 3  # 30 params vs paper's ~514 classical
            self.rl.n_actions = 2  # Binary baseline: EXTEND, CUT
    
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
