"""API config — re-exports the shared RhizomeConfig for convenience.

Import from here rather than directly from rhizome.config to keep
API-specific imports grouped.
"""

from rhizome.config import get_config, RhizomeConfig

__all__ = ["get_config", "RhizomeConfig"]
