"""Traversal engine configuration."""

from dataclasses import dataclass


@dataclass
class TraversalConfig:
    """Configuration for the traversal engine."""

    depth: int = 8
    """Maximum number of steps before stopping."""

    epsilon: float = 0.1
    """Probability of random exploration vs. exploitation (0.0 = pure exploit, 1.0 = pure explore)."""

    top_k: int = 20
    """Number of nearest candidates to consider at each step."""

    collection_name: str = "modernity-v1"
    """Qdrant collection to traverse."""

    temperature: float = 1.0
    """Softmax temperature for exploit-path sampling. 0.0=greedy, 1.0=natural, 2.0+=flat/exploratory."""

    max_same_article_consecutive: int = 2
    """Hard block: rolling window size for same-article consecutive steps. 0=disabled."""

    @classmethod
    def from_dict(cls, data: dict) -> "TraversalConfig":
        """Create config from a dictionary (e.g., loaded from YAML)."""
        return cls(
            depth=data.get("depth", 8),
            epsilon=data.get("epsilon", 0.1),
            top_k=data.get("top_k", 20),
            collection_name=data.get("collection_name", "modernity-v1"),
            temperature=data.get("temperature", 1.0),
            max_same_article_consecutive=data.get("max_same_article_consecutive", 2),
        )
