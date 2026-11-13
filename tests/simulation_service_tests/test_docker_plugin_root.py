from pathlib import Path

import pytest

from services.simulation_service.core.service import simulation_cluster_manager as scm
from utils import find_repo_root


def test_bundled_plugin_root_is_accepted(monkeypatch):
    monkeypatch.delenv("URGENT_PLUGIN_PATH", raising=False)
    scm.ensure_docker_compatible_plugin_root()


def test_explicit_bundled_plugin_root_is_accepted(monkeypatch):
    monkeypatch.setenv("URGENT_PLUGIN_PATH", str(find_repo_root() / "plugins"))
    scm.ensure_docker_compatible_plugin_root()


def test_external_plugin_root_fails_fast(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("URGENT_PLUGIN_PATH", str(tmp_path))
    with pytest.raises(RuntimeError, match="Docker mode only supports plugins"):
        scm.ensure_docker_compatible_plugin_root()


def test_cluster_context_checks_plugin_root_before_building(monkeypatch, tmp_path):
    monkeypatch.setenv("URGENT_PLUGIN_PATH", str(tmp_path))
    built = []
    monkeypatch.setattr(
        scm.SimulationClusterManager,
        "build_simulation_cluster_images",
        staticmethod(lambda: built.append(True)),
    )
    with pytest.raises(RuntimeError):
        with scm.simulation_cluster_context_manager(worker_count=1):
            pass
    assert built == []
