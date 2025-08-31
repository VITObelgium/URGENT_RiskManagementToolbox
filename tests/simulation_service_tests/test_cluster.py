import subprocess
from unittest.mock import MagicMock, patch

import pytest

from services.simulation_service.core.service.simulation_cluster_manager import (
    SimulationClusterManager,
    simulation_cluster_context_manager,
)


@pytest.fixture
def mock_subprocess_run():
    with patch("subprocess.run") as mock_run:
        yield mock_run


@pytest.fixture
def mock_subprocess_popen():
    with patch("subprocess.Popen") as mock_popen:
        mock_popen.return_value.stdout = MagicMock()
        mock_popen.return_value.stderr = MagicMock()
        yield mock_popen


@pytest.fixture
def mock_core_directory():
    with patch(
        "services.simulation_service.core.service.simulation_cluster_manager.core_directory"
    ) as mock_cd:
        yield mock_cd


class TestSimulationClusterManager:
    def test_build_simulation_cluster_images_success(
        self, mock_subprocess_run, mock_core_directory
    ):
        mock_subprocess_run.return_value = MagicMock(
            check=True, stdout="success", stderr=""
        )
        SimulationClusterManager.build_simulation_cluster_images()
        mock_subprocess_run.assert_called_once_with(
            ["docker", "compose", "build", "--progress", "plain", "--force-rm"],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_build_simulation_cluster_images_failure(
        self, mock_subprocess_run, mock_core_directory
    ):
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "cmd", stderr="error"
        )
        with pytest.raises(subprocess.CalledProcessError):
            SimulationClusterManager.build_simulation_cluster_images()

    def test_prune_dangling_images(self, mock_subprocess_popen):
        SimulationClusterManager.prune_dangling_images()
        mock_subprocess_popen.assert_called_once_with(
            ["docker", "image", "prune", "-f"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def test_start_simulation_cluster_success(
        self, mock_subprocess_run, mock_core_directory
    ):
        worker_count = 5
        with patch(
            "services.simulation_service.core.service.simulation_cluster_manager.log_docker_logs"
        ) as mock_log_docker_logs:
            SimulationClusterManager.start_simulation_cluster(worker_count)
            mock_subprocess_run.assert_called_once_with(
                [
                    "docker",
                    "compose",
                    "up",
                    "-d",
                    "--scale",
                    f"worker={worker_count}",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            mock_log_docker_logs.assert_called_once()

    def test_start_simulation_cluster_failure(
        self, mock_subprocess_run, mock_core_directory
    ):
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "cmd", stderr="error"
        )
        with pytest.raises(subprocess.CalledProcessError):
            SimulationClusterManager.start_simulation_cluster()

    def test_shutdown_simulation_cluster_success(
        self, mock_subprocess_run, mock_core_directory
    ):
        with patch(
            "services.simulation_service.core.service.grpc_stub_manager.GrpcStubManager.close"
        ) as mock_close:
            SimulationClusterManager.shutdown_simulation_cluster()
            mock_close.assert_called_once()
            mock_subprocess_run.assert_called_once_with(
                ["docker", "compose", "down"],
                check=True,
                capture_output=True,
                text=True,
            )

    def test_shutdown_simulation_cluster_failure(
        self, mock_subprocess_run, mock_core_directory
    ):
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "cmd", stderr="error"
        )
        with pytest.raises(subprocess.CalledProcessError):
            SimulationClusterManager.shutdown_simulation_cluster()


@patch(
    "services.simulation_service.core.service.simulation_cluster_manager.SimulationClusterManager.build_simulation_cluster_images"
)
@patch(
    "services.simulation_service.core.service.simulation_cluster_manager.SimulationClusterManager.prune_dangling_images"
)
@patch(
    "services.simulation_service.core.service.simulation_cluster_manager.SimulationClusterManager.start_simulation_cluster"
)
@patch(
    "services.simulation_service.core.service.simulation_cluster_manager.SimulationClusterManager.shutdown_simulation_cluster"
)
def test_simulation_cluster_context_manager(
    mock_shutdown, mock_start, mock_prune, mock_build
):
    worker_count = 4
    with simulation_cluster_context_manager(worker_count):
        pass

    mock_build.assert_called_once()
    mock_prune.assert_called_once()
    mock_start.assert_called_once_with(worker_count)
    mock_shutdown.assert_called_once()
