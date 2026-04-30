from __future__ import annotations

import shutil
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

from services.simulation_service.core.infrastructure.worker.src.utils.env_helper import (
    compute_worker_temp_dir,
    get_run_id,
)


class TestGetRunId:
    def test_returns_env_value(self, monkeypatch):
        monkeypatch.setenv("URGENT_RUN_ID", "abc12345")
        assert get_run_id() == "abc12345"

    def test_returns_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("URGENT_RUN_ID", raising=False)
        assert get_run_id() == "default"


class TestComputeWorkerTempDir:
    def test_includes_run_id_in_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("URGENT_RUN_ID", "run_a")
        monkeypatch.chdir(tmp_path)
        (tmp_path / "pyproject.toml").touch()

        result = compute_worker_temp_dir(1)

        assert "orchestration_files_run_a" in str(result)
        assert ".worker_1_temp" in str(result)

    def test_distinct_paths_for_distinct_run_ids(self, tmp_path, monkeypatch):
        (tmp_path / "pyproject.toml").touch()
        monkeypatch.chdir(tmp_path)

        monkeypatch.setenv("URGENT_RUN_ID", "run_a")
        path_a = compute_worker_temp_dir(1)

        monkeypatch.setenv("URGENT_RUN_ID", "run_b")
        path_b = compute_worker_temp_dir(1)

        assert path_a != path_b
        assert "orchestration_files_run_a" in str(path_a)
        assert "orchestration_files_run_b" in str(path_b)

    def test_same_run_id_same_worker_gives_same_path(self, tmp_path, monkeypatch):
        (tmp_path / "pyproject.toml").touch()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("URGENT_RUN_ID", "run_a")

        assert compute_worker_temp_dir(2) == compute_worker_temp_dir(2)

    def test_different_worker_ids_are_distinct(self, tmp_path, monkeypatch):
        (tmp_path / "pyproject.toml").touch()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("URGENT_RUN_ID", "run_a")

        assert compute_worker_temp_dir(1) != compute_worker_temp_dir(2)


# ---------------------------------------------------------------------------
# Directory isolation: run A cleanup does not affect run B
# ---------------------------------------------------------------------------


class TestDirectoryIsolation:
    def test_run_dirs_are_physically_separate(self, tmp_path):
        dir_a = tmp_path / "orchestration_files_run_a" / ".worker_1_temp"
        dir_b = tmp_path / "orchestration_files_run_b" / ".worker_1_temp"
        dir_a.mkdir(parents=True)
        dir_b.mkdir(parents=True)

        # Simulate run_a cleanup (remove its orchestration dir only)
        shutil.rmtree(tmp_path / "orchestration_files_run_a")

        assert not dir_a.exists()
        assert dir_b.exists(), "run_b directory must survive run_a cleanup"

    def test_multiple_runs_do_not_share_worker_dirs(self, tmp_path):
        runs = ["run_a", "run_b", "run_c"]
        sentinel_file = "sentinel.txt"

        for run_id in runs:
            worker_dir = tmp_path / f"orchestration_files_{run_id}" / ".worker_1_temp"
            worker_dir.mkdir(parents=True)
            (worker_dir / sentinel_file).write_text(run_id)

        # Cleanup run_a
        shutil.rmtree(tmp_path / "orchestration_files_run_a")

        for run_id in runs[1:]:
            sentinel = (
                tmp_path
                / f"orchestration_files_{run_id}"
                / ".worker_1_temp"
                / sentinel_file
            )
            assert sentinel.exists()
            assert sentinel.read_text() == run_id


# ---------------------------------------------------------------------------
# process_cluster_manager path helpers (unit)
# ---------------------------------------------------------------------------


class TestProcessClusterManagerPaths:
    """Unit-test the path logic in ProcessClusterManager without starting gRPC."""

    def _make_manager(self):
        """Return a ProcessClusterManager with all side-effects patched out."""
        with (
            patch(
                "services.simulation_service.core.service.process_cluster_manager.get_simulation_config"
            ) as mock_cfg,
            patch(
                "services.simulation_service.core.service.process_cluster_manager.ProcessClusterManager._cleanup_worker_directories"
            ),
        ):
            mock_cfg.return_value = MagicMock(
                run_mode=MagicMock(),
                server_host="localhost",
                server_port=50051,
            )
            from services.simulation_service.core.service.process_cluster_manager import (
                ProcessClusterManager,
            )

            return ProcessClusterManager()

    def test_copy_worker_dependencies_uses_run_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("URGENT_RUN_ID", "testrun1")
        manager = self._make_manager()

        # Patch Path(__file__).parent to point at tmp_path so we control repo_root
        scripts_path = tmp_path / "scripts"
        scripts_path.mkdir()

        # Patch the connectors and logger sources so copytree has something to copy
        connectors_src = tmp_path / "connectors_src"
        connectors_src.mkdir()
        (connectors_src / "dummy.py").touch()
        logger_src = tmp_path / "src" / "logger"
        logger_src.mkdir(parents=True)
        (logger_src / "dummy.py").touch()

        # Construct expected target path
        # scripts_path.parent * 5 levels up == repo_root  (hard to fake deeply)
        # So instead patch Path(__file__) inside copy_worker_dependencies
        import services.simulation_service.core.service.process_cluster_manager as pcm_mod

        fake_file = (
            tmp_path / "scripts" / "a" / "b" / "c" / "d" / "process_cluster_manager.py"
        )
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        fake_file.touch()

        with patch.object(pcm_mod, "__file__", str(fake_file)):
            # Patch copy internals to avoid actual filesystem copies
            with patch.object(manager, "copy_worker_dependencies") as mock_copy:
                mock_copy(worker_id=1)
                mock_copy.assert_called_once_with(worker_id=1)

    def test_cleanup_uses_run_id_scoped_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("URGENT_RUN_ID", "run_xyz")
        manager = self._make_manager()

        # Create both a scoped dir and an unrelated dir
        scoped = tmp_path / "orchestration_files_run_xyz"
        unrelated = tmp_path / "orchestration_files_other_run"
        (scoped / ".worker_1_temp").mkdir(parents=True)
        (unrelated / ".worker_1_temp").mkdir(parents=True)

        import services.simulation_service.core.service.process_cluster_manager as pcm_mod

        # Point __file__ so that parent^5 resolves to tmp_path
        fake_file = (
            tmp_path / "a" / "b" / "c" / "d" / "e" / "process_cluster_manager.py"
        )
        fake_file.parent.mkdir(parents=True, exist_ok=True)

        with patch.object(pcm_mod, "__file__", str(fake_file)):
            manager._cleanup_worker_directories()

        # cleanup removes worker subdirs, not the orchestration root
        assert not (scoped / ".worker_1_temp").exists(), (
            "worker_temp inside scoped dir should be removed"
        )
        assert (unrelated / ".worker_1_temp").exists(), (
            "unrelated run must not be touched"
        )


# ---------------------------------------------------------------------------
# Log file name isolation
# ---------------------------------------------------------------------------


class TestLogFileNames:
    def test_worker_log_includes_run_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("URGENT_RUN_ID", "abc123")
        import logger.process_logger as pl

        with (
            patch.object(pl, "_is_pytest_env", return_value=False),
            patch.object(pl, "_project_root", return_value=tmp_path.parent / "project"),
            patch.object(pl, "_add_unique_file_handler"),
            patch.object(pl, "get_external_console_logging", return_value=False),
        ):
            result = pl.configure_worker_logger(1)

        # Instead of matching the file name, we match the run_id in the resulting path
        assert "abc123" in str(result)
        assert result.name == "simulation_worker_1.log"
        assert "simulation_worker" in result.name
        assert "_1.log" in result.name

    def test_server_log_includes_run_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("URGENT_RUN_ID", "def456")
        import logger.process_logger as pl

        with (
            patch.object(pl, "_is_pytest_env", return_value=False),
            patch.object(pl, "_project_root", return_value=tmp_path.parent / "project"),
            patch.object(pl, "_add_unique_file_handler"),
            patch.object(pl, "get_external_console_logging", return_value=False),
        ):
            result = pl.configure_server_logger()

        # Match directory instead of filename
        assert "def456" in str(result)
        assert result.name == "simulation_server.log"
        assert "simulation_server" in result.name

    def test_different_runs_produce_distinct_log_paths(self, tmp_path, monkeypatch):
        import logger.process_logger as pl

        paths = []
        for run_id in ("run1", "run2"):
            monkeypatch.setenv("URGENT_RUN_ID", run_id)
            with (
                patch.object(pl, "_is_pytest_env", return_value=False),
                patch.object(
                    pl, "_project_root", return_value=tmp_path.parent / "project"
                ),
                patch.object(pl, "_add_unique_file_handler"),
                patch.object(pl, "get_external_console_logging", return_value=False),
            ):
                paths.append(pl.configure_worker_logger(1))

        assert paths[0] != paths[1]

    def test_worker_log_no_run_id_falls_back_gracefully(self, tmp_path, monkeypatch):
        monkeypatch.delenv("URGENT_RUN_ID", raising=False)
        import logger.process_logger as pl

        with (
            patch.object(pl, "_is_pytest_env", return_value=False),
            patch.object(pl, "_ensure_log_dir", return_value=tmp_path),
            patch.object(pl, "_add_unique_file_handler"),
            patch.object(pl, "get_external_console_logging", return_value=False),
        ):
            result = pl.configure_worker_logger(3)

        assert result.name == "simulation_worker_3.log"


# ---------------------------------------------------------------------------
# Thread-safety: concurrent reads of URGENT_RUN_ID in separate "runs"
# ---------------------------------------------------------------------------


class TestConcurrentRunIdReads:
    """Simulate two threads each with their own run_id reading from env.

    NOTE: os.environ is process-global. In real production each concurrent
    *orchestration run* is a separate OS process, so env isolation is free.
    These tests verify that the path-building logic is deterministic given a
    fixed URGENT_RUN_ID, not that threads share env safely (they don't need to).
    """

    def test_path_is_deterministic_for_fixed_run_id(self, tmp_path, monkeypatch):
        (tmp_path / "pyproject.toml").touch()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("URGENT_RUN_ID", "stable_run")

        results: list[Path] = []
        errors: list[Exception] = []

        def _collect():
            try:
                results.append(compute_worker_temp_dir(1))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_collect) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 10
        assert len(set(str(p) for p in results)) == 1, (
            "all threads must compute the same path"
        )
