"""Test isolation for the process environment.

`rhizome.cli.main` calls `load_dotenv()` at import time, and pytest imports
every test module during collection — before any test runs. A developer's local
`.env` therefore lands in `os.environ` for the whole session and silently
overrides config defaults, making tests that assert defaults depend on whether
the repo happens to have a `.env`.

This fixture clears every environment variable that maps to a RhizomeConfig
field before each test and restores the original environment afterwards. Alias
names are derived from the model so new config fields are covered automatically.
"""

import os

import pytest

from rhizome.config import RhizomeConfig


def _config_env_names() -> set[str]:
    """Environment variable names that feed RhizomeConfig.

    Pydantic-settings resolves both the field name and its alias, and the
    project sets case_sensitive=False, so every casing of both is cleared.
    """
    names: set[str] = set()
    for field_name, field in RhizomeConfig.model_fields.items():
        names.add(field_name)
        alias = getattr(field, "alias", None)
        if isinstance(alias, str):
            names.add(alias)
        for name in list(names):
            names.add(name.upper())
            names.add(name.lower())
    return names


@pytest.fixture(autouse=True)
def hermetic_env():
    """Hide local `.env` values from config construction during tests.

    RhizomeConfig reads the repo `.env` directly via pydantic-settings'
    ``env_file``, so clearing ``os.environ`` is not enough — the fixture also
    points ``env_file`` at a nonexistent path for the duration of the test.
    """
    saved = dict(os.environ)
    saved_env_file = RhizomeConfig.model_config.get("env_file")
    try:
        for name in _config_env_names():
            os.environ.pop(name, None)
        RhizomeConfig.model_config["env_file"] = "/nonexistent/.env"
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)
        RhizomeConfig.model_config["env_file"] = saved_env_file
