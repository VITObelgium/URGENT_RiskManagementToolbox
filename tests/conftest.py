import os

import numpy as np
import pytest

from urgent_plugins import get_registry

# Process-global env vars written at runtime by plugin resolution
# (urgent_plugins.utils._resolve_plugin) and run configuration
# (risk_management_service._configure_environment).
_RUNTIME_ENV_PREFIXES = ("URGENT_",)
_RUNTIME_ENV_KEYS = frozenset(
    {
        "RUNNER_MODE",
        "RUN_MODE",
        "WORKER_SIMULATION_TIMEOUT_SECONDS",
        "SERVER_JOB_TIMEOUT_SECONDS",
    }
)


def _is_runtime_env_key(key: str) -> bool:
    return key.startswith(_RUNTIME_ENV_PREFIXES) or key in _RUNTIME_ENV_KEYS


@pytest.fixture(autouse=True)
def mock_default_rng(monkeypatch):
    original_default_rng = np.random.default_rng

    def mocked_default_rng(seed=None):
        if seed is None:
            # Use the legacy random state to generate a seed
            seed = np.random.randint(0, 2**31 - 1)
        return original_default_rng(seed)

    monkeypatch.setattr(np.random, "default_rng", mocked_default_rng)


@pytest.fixture(autouse=True)
def isolate_plugin_state():
    """Reset the plugin registry and runtime env vars after every test.

    Plugin resolution writes os.environ directly and registers descriptors in
    a process-wide registry, so without this one test's plugins and env leak
    into the next (and results depend on test order).
    """
    saved_env = {k: v for k, v in os.environ.items() if _is_runtime_env_key(k)}
    yield
    get_registry().clear()
    for key in [k for k in os.environ if _is_runtime_env_key(k)]:
        if key not in saved_env:
            del os.environ[key]
    os.environ.update(saved_env)
