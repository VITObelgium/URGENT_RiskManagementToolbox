import io
import subprocess
import sys
from subprocess import Popen as OriginalPopen
from typing import Any, Generator
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from services.simulation_service.core.connectors.common import (
    GridCell,
    SimulationStatus,
    WellManagementServiceResultSchema,
    WellName,
)
from services.simulation_service.core.connectors.open_darts import (
    OpenDartsConnector,
    _GlobalData,
    _StructDiscretizerProtocol,
    _StructReservoirProtocol,
    open_darts_input_configuration_injector,
)


@pytest.fixture
def mock_discretizer() -> Mock:
    """Creates a mock of the reservoir discretizer with appropriate test data."""
    discretizer_mock = Mock()

    # Create cell lengths that will position centroids predictably
    # The test expects the well trajectory at [0, 0, z] to connect with cells at (1, 1, z)

    # Create 10x10x10 arrays for cell lengths in each direction
    len_cell_xdir = np.ones((10, 10, 10)) * 50.0
    len_cell_ydir = np.ones((10, 10, 10)) * 50.0
    len_cell_zdir = np.ones((10, 10, 10)) * 50.0

    # Configure the mock's attributes
    discretizer_mock.len_cell_xdir = len_cell_xdir
    discretizer_mock.len_cell_ydir = len_cell_ydir
    discretizer_mock.len_cell_zdir = len_cell_zdir

    return discretizer_mock


@pytest.fixture
def global_data_mock() -> Mock:
    """Creates a mock of the global data with appropriate test values."""
    mock = Mock()

    # Create a dictionary-like behavior for the global_data
    global_data = {
        "dx": np.ones((10, 10, 10)) * 50.0,
        "dy": np.ones((10, 10, 10)) * 50.0,
        "dz": np.ones((10, 10, 10)) * 50.0,
        "start_z": 0.0,  # Starting at z=0
    }

    # Configure the mock to return values from the dictionary
    mock.__getitem__ = lambda self, key: global_data[key]

    return mock


@pytest.fixture
def mock_struct_reservoir(
    mock_discretizer: _StructDiscretizerProtocol, global_data_mock: _GlobalData
) -> _StructReservoirProtocol:
    class MockStructReservoir:
        nx: int = 10
        ny: int = 10
        nz: int = 10
        discretizer: _StructDiscretizerProtocol = mock_discretizer
        global_data: _GlobalData = global_data_mock

    return MockStructReservoir()


@pytest.mark.parametrize(
    "wells_result, expected_output",
    [
        (
            {
                "wells": [
                    {
                        "name": "Test IWell",
                        "trajectory": [
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 10.0],
                            [0.0, 0.0, 50.0],
                            [0.0, 0.0, 60.0],
                            [0.0, 0.0, 70.0],
                            [0.0, 0.0, 80.0],
                            [0.0, 0.0, 100.0],
                        ],
                        "completion": {
                            "perforations": [
                                {
                                    "range": [10.0, 60.0],
                                    "points": [
                                        [0.0, 0.0, 10.0],
                                        [0.0, 0.0, 50.0],
                                        [0.0, 0.0, 60.0],
                                    ],
                                },
                                {
                                    "range": [70.0, 80.0],
                                    "points": [[0.0, 0.0, 70.0], [0.0, 0.0, 80.0]],
                                },
                            ]
                        },
                    }
                ]
            },
            {"Test IWell": ((1, 1, 1), (1, 1, 2))},
        )
    ],
)
def test_get_well_connection_cells(
    mock_struct_reservoir: Mock,
    wells_result: WellManagementServiceResultSchema,
    expected_output: dict[WellName, tuple[GridCell, ...]],
) -> None:
    actual_result = OpenDartsConnector.get_well_connection_cells(
        wells_result, mock_struct_reservoir
    )
    assert actual_result == expected_output


def test_get_well_connection_cells_empty_wells(mock_struct_reservoir):
    wells_result = {"wells": []}
    result = OpenDartsConnector.get_well_connection_cells(
        wells_result, mock_struct_reservoir
    )
    assert result == {}


def test_get_well_connection_cells_missing_completion(mock_struct_reservoir):
    wells_result = {"wells": [{"name": "NoComp", "trajectory": [[0, 0, 0]]}]}
    import pytest

    with pytest.raises(KeyError, match="Well does not have a field: 'completion'"):
        OpenDartsConnector.get_well_connection_cells(
            wells_result, mock_struct_reservoir
        )


@pytest.fixture
def mock_subprocess_run() -> Generator[Mock, None, None]:
    with patch("subprocess.run") as mock_run:
        yield mock_run


@pytest.fixture
def mock_popen() -> Generator[Mock, None, None]:
    # Patch Popen in the module where ManagedSubprocess uses it
    with patch(
        "services.simulation_service.core.connectors.conn_utils.managed_subprocess.subprocess.Popen"
    ) as mock_popen_constructor:
        mock_popen_instance = MagicMock(
            spec=OriginalPopen
        )  # Use OriginalPopen for spec
        mock_popen_instance.pid = 12345  # Dummy PID

        # These will be configured per test
        mock_popen_instance.stdout = io.StringIO("")
        mock_popen_instance.stderr = io.StringIO("")

        def _mock_wait(timeout: Any = None) -> int | None:
            # Simulate process completion. The actual returncode should be set
            # on mock_popen_instance by the test setup.
            # Popen.wait() returns the exit status.
            return getattr(mock_popen_instance, "returncode", None)

        mock_popen_instance.wait = MagicMock(side_effect=_mock_wait)

        def _mock_poll() -> int | None:
            # Return the returncode if wait() has "completed" and set it
            # or if the process has terminated and set its returncode.
            return getattr(mock_popen_instance, "returncode", None)

        mock_popen_instance.poll = MagicMock(side_effect=_mock_poll)
        mock_popen_instance.terminate = MagicMock()
        mock_popen_instance.kill = MagicMock()

        mock_popen_constructor.return_value = mock_popen_instance
        yield mock_popen_constructor


@pytest.fixture(autouse=True)
def patch_stream_reader(monkeypatch):
    """
    Patch the stream_reader used by SubprocessRunner so it appends lines from the mock's
    stdout/stderr to manager.stdout_lines/manager.stderr_lines without using real logging.
    """
    from services.simulation_service.core.connectors import open_darts, runner

    def fake_stream_reader(stream, lines_list, logger_func):
        for line in stream:
            lines_list.append(line.rstrip("\n"))

    monkeypatch.setattr(open_darts, "stream_reader", fake_stream_reader, raising=False)
    monkeypatch.setattr(runner, "stream_reader", fake_stream_reader, raising=False)
    yield


def test_run_success(mock_popen: Mock) -> None:
    mock_popen_instance = mock_popen.return_value

    mock_popen_instance.stdout = io.StringIO(
        "Building connection list...\n"
        "OpenDartsConnector: Type:heat, Value:2312.12\n"
        "OpenDartsConnector: Type:heat, Value:[1.23, 4.56, 7.89]\n"
        "# 1 \tT = 0.0001\tDT = 0.0001\tNI = 2\tLI=2\n"
        "# 2 \tT = 0.0003\tDT = 0.0002\tNI = 1\tLI=1\n"
        "# 3 \tT = 0.0007\tDT = 0.0004\tNI = 1\tLI=1\n"
        "# 4 \tT = 0.0015\tDT = 0.0008\tNI = 1\tLI=1\n"
    )
    mock_popen_instance.stderr = io.StringIO("")
    mock_popen_instance.returncode = 0

    simulation_status, results = OpenDartsConnector.run("test_config")

    assert results == {"heat": [2312.12, [1.23, 4.56, 7.89]]}
    assert simulation_status == SimulationStatus.SUCCESS


def test_run_handles_no_output_lines(mock_popen: Mock) -> None:
    mock_popen_instance = mock_popen.return_value
    mock_popen_instance.stdout = io.StringIO("")
    mock_popen_instance.stderr = io.StringIO("")
    mock_popen_instance.returncode = 0
    simulation_status, results = OpenDartsConnector.run("test_config")
    assert simulation_status == SimulationStatus.SUCCESS
    assert results == {}


def test_run_success_with_single_broadcasted_value(mock_popen: Mock) -> None:
    mock_popen_instance = mock_popen.return_value

    mock_popen_instance.stdout = io.StringIO(
        "Building connection list...\n"
        "OpenDartsConnector: Type:heat, Value:2312.12\n"
        "# 1 \tT = 0.0001\tDT = 0.0001\tNI = 2\tLI=2\n"
    )
    mock_popen_instance.stderr = io.StringIO("")
    mock_popen_instance.returncode = 0

    simulation_status, results = OpenDartsConnector.run("test_config")

    assert simulation_status == SimulationStatus.SUCCESS
    assert results == {"heat": 2312.12}
    mock_popen.assert_called_once()
    assert mock_popen_instance.wait.called


def test_run_failure(mock_popen: Mock) -> None:
    mock_popen_instance = mock_popen.return_value

    mock_popen_instance.stdout = io.StringIO("Some error output if any")
    mock_popen_instance.stderr = io.StringIO("Error details from stderr")
    mock_popen_instance.returncode = 1

    simulation_status, _ = OpenDartsConnector.run("test_config")

    assert simulation_status == SimulationStatus.FAILED

    mock_popen.assert_called_once()
    assert mock_popen_instance.wait.called


def test_run_handles_exception(mock_popen: Mock) -> None:
    # Simulate exception in Popen
    with patch(
        "services.simulation_service.core.connectors.open_darts.ManagedSubprocess.__enter__",
        side_effect=Exception("fail"),
    ):
        status, results = OpenDartsConnector.run("test_config")
        assert status == SimulationStatus.FAILED
        # The default_failed_results dict is returned, not an empty dict
        from services.simulation_service.core.connectors.common import (
            SimulationResultType,
        )

        assert results == {k: -1e3 for k in SimulationResultType}


@open_darts_input_configuration_injector
def main(configuration_content: dict[str, Any]) -> None:
    print(f"Received configuration: {configuration_content}")


# Pytest test cases
def test_decorator_with_valid_json(monkeypatch, capsys):  # type: ignore
    # Mock sys.argv to simulate command-line arguments
    valid_json = '{"key": "value", "number": 42}'
    monkeypatch.setattr(sys, "argv", ["main.py", valid_json])

    # Call the decorated function
    main()

    # Capture printed output and verify it
    captured = capsys.readouterr()
    assert "Received configuration: {'key': 'value', 'number': 42}" in captured.out


def test_decorator_with_extra_args(monkeypatch, capsys):
    # Should still parse first arg as JSON
    valid_json = '{"foo": 1}'
    monkeypatch.setattr(sys, "argv", ["main.py", valid_json, "extra"])
    main()
    captured = capsys.readouterr()
    assert "Received configuration: {'foo': 1}" in captured.out


def test_decorator_with_invalid_json(monkeypatch):
    # Mock sys.argv to simulate command-line arguments with invalid JSON
    invalid_json = '{"key": "value", "number": 42'
    monkeypatch.setattr(sys, "argv", ["main.py", invalid_json])

    # Mock sys.exit to prevent the test from stopping
    with patch("sys.exit") as mock_exit:
        mock_func = Mock()
        decorated_func = open_darts_input_configuration_injector(mock_func)
        decorated_func()
        mock_exit.assert_called_once_with(1)
        mock_func.assert_not_called()


def test_decorator_with_non_dict_json(monkeypatch):
    # Should exit if JSON is not a dict
    monkeypatch.setattr(sys, "argv", ["main.py", "123"])

    with patch("sys.exit") as mock_exit:
        mock_func = Mock()
        decorated_func = open_darts_input_configuration_injector(mock_func)
        decorated_func()
        mock_exit.assert_called_once_with(1)
        mock_func.assert_not_called()


def test_decorator_with_missing_argument(monkeypatch):
    # Mock sys.argv to simulate no arguments provided
    monkeypatch.setattr(sys, "argv", ["main.py"])

    with patch("sys.exit") as mock_exit:
        mock_func = Mock()
        decorated_func = open_darts_input_configuration_injector(mock_func)
        decorated_func()
        mock_exit.assert_called_once_with(1)
        mock_func.assert_not_called()


def test_decorator_with_dict_but_not_str_keys(monkeypatch):
    # Should exit if JSON dict has non-str keys

    class Dummy(dict):
        pass

    def fake_loads(s):
        return Dummy({1: "a"})

    monkeypatch.setattr(sys, "argv", ["main.py", '{"1": "a"}'])

    with patch("json.loads", fake_loads):
        with patch("sys.exit") as mock_exit:
            mock_func = Mock()
            decorated_func = open_darts_input_configuration_injector(mock_func)
            decorated_func()
            mock_exit.assert_called_once_with(1)
            mock_func.assert_not_called()


def test_managed_subprocess_exit_terminates_process(mock_popen):
    # Mock the subprocess to simulate a running process
    mock_popen_instance = mock_popen.return_value
    mock_popen_instance.poll.return_value = None  # Simulate process still running

    # Simulate timeout scenario
    mock_popen_instance.wait.side_effect = subprocess.TimeoutExpired(
        cmd="test", timeout=5
    )

    # Create a ManagedSubprocess instance
    from services.simulation_service.core.connectors.conn_utils.managed_subprocess import (
        ManagedSubprocess,
    )

    manager = ManagedSubprocess(
        command_args=["python3", "-u", "main.py"],
        stream_reader_func=lambda *args: None,
        logger_info_func=lambda msg: None,
        logger_error_func=lambda msg: None,
    )

    # Use the context manager and trigger __exit__
    with patch.object(manager, "logger_warning_func", lambda msg: None):
        try:
            with manager:
                pass
        except subprocess.TimeoutExpired:
            # Expected exception, do nothing and then check if process was terminated properly
            pass

    # Verify terminate and kill were called
    mock_popen_instance.terminate.assert_called_once()
    mock_popen_instance.kill.assert_called_once()
