"""Traversal engine configuration."""

from dataclasses import dataclass


@dataclass
class TraversalConfig:
    """Configuration for the traversal engine."""

    depth: int = 8
    """Maximum number of steps before stopping."""

    epsilon: float = 0.1
    """Probability of random exploration vs. exploitation (0.0 = pure exploit, 1.0 = pure explore)."""

    top_k: int = 5
    """Number of nearest candidates to consider at each step."""

    collection_name: str = "modernity-v1"
    """Qdrant collection to traverse."""

    @classmethod
    def from_dict(cls, data: dict) -> "TraversalConfig":
        """Create config from a dictionary (e.g., loaded from YAML)."""
        return cls(
            depth=data.get("depth", 8),
            epsilon=data.get("epsilon", 0.1),
            top_k=data.get("top_k", 5),
            collection_name=data.get("collection_name", "modernity-v1"),
        )
