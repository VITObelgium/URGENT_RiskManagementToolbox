import numpy as np
import pytest


@pytest.fixture(autouse=True)
def mock_default_rng(monkeypatch):
    original_default_rng = np.random.default_rng

    def mocked_default_rng(seed=None):
        if seed is None:
            # Use the legacy random state to generate a seed
            seed = np.random.randint(0, 2**31 - 1)
        return original_default_rng(seed)

    monkeypatch.setattr(np.random, "default_rng", mocked_default_rng)
