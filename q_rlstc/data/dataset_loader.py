"""Abstract dataset loader for Q-RLSTC experiments.

Provides a common interface for loading trajectory datasets from
different sources (GeoLife, Porto, T-Drive, custom) with optional
preprocessing.

Usage:
    loader = DatasetLoader("geolife", preprocess="RLSTC_MDL")
    trajectories = loader.load(fold=0, split="train")
"""

import os
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class TrajectoryDataset:
    """Container for a loaded trajectory dataset.

    Attributes:
        name: Dataset name.
        trajectories: List of trajectories (each is a list of (lat, lon) tuples).
        n_trajectories: Number of trajectories.
        metadata: Optional metadata dict (source, preprocessing, etc.).
    """
    name: str
    trajectories: list
    n_trajectories: int
    metadata: Dict[str, Any]


class DatasetLoader(ABC):
    """Abstract base class for trajectory dataset loaders."""

    @abstractmethod
    def load(
        self,
        fold: int = 0,
        split: str = "train",
    ) -> TrajectoryDataset:
        """Load trajectories for a given fold and split.

        Args:
            fold: Cross-validation fold index.
            split: "train" or "val".

        Returns:
            TrajectoryDataset with loaded trajectories.
        """
        ...

    @abstractmethod
    def n_folds(self) -> int:
        """Number of available folds."""
        ...


@dataclass
class DatasetConfig:
    """Configuration for dataset loading.

    Attributes:
        name: Dataset identifier ("geolife", "porto", "tdrive", "custom").
        data_path: Path to dataset files.
        preprocess: Preprocessing mode:
            "RLSTC_MDL" — apply RLSTC paper's MDL-based segmentation.
            "none" — raw trajectories, no preprocessing.
            "current" — use the project's default preprocessing.
        k_clusters: Number of clusters for evaluation (k-sensitivity).
        n_folds: Number of cross-validation folds.
        min_trajectory_length: Minimum trajectory length (in points).
    """
    name: str = "custom"
    data_path: str = ""
    preprocess: str = "current"
    k_clusters: int = 10
    n_folds: int = 5
    min_trajectory_length: int = 5


class CustomDatasetLoader(DatasetLoader):
    """Loader for custom trajectory data (e.g., Deer dataset).

    Wraps the existing RLSTCCluster data loading for backward
    compatibility while providing the abstract DatasetLoader interface.
    """

    def __init__(self, config: DatasetConfig):
        self.config = config

    def load(self, fold: int = 0, split: str = "train") -> TrajectoryDataset:
        """Load custom dataset using existing infrastructure.

        Note: Actual loading delegated to RLSTCCluster — this wrapper
        provides the abstract interface for future multi-dataset support.
        """
        return TrajectoryDataset(
            name=self.config.name,
            trajectories=[],  # Populated by RLSTCCluster
            n_trajectories=0,
            metadata={
                "fold": fold,
                "split": split,
                "preprocess": self.config.preprocess,
                "k_clusters": self.config.k_clusters,
            },
        )

    def n_folds(self) -> int:
        return self.config.n_folds


def get_loader(config: DatasetConfig) -> DatasetLoader:
    """Factory function: get the appropriate loader for a dataset.

    Args:
        config: Dataset configuration.

    Returns:
        DatasetLoader instance.
    """
    # All datasets currently use the custom loader
    # Future: add GeoLifeLoader, PortoLoader, TDriveLoader
    return CustomDatasetLoader(config)
