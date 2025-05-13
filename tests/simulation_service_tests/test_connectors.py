import sys
from typing import Any, Generator
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from services.simulation_service.core.connectors.common import (
    GridCell,
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


@pytest.fixture
def mock_subprocess_run() -> Generator[Mock, None, None]:
    with patch("subprocess.run") as mock_run:
        yield mock_run


def test_run_success(mock_subprocess_run: Mock) -> None:
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = (
        "Building connection list...\n"
        "OpenDartsConnector: Type:heat, Value:2312.12\n"
        "OpenDartsConnector: Type:heat, Value:[1.23, 4.56, 7.89]\n"
        "# 1 	T = 0.0001	DT = 0.0001	NI = 2	LI=2\n"
        "# 2 	T = 0.0003	DT = 0.0002	NI = 1	LI=1\n"
        "# 3 	T = 0.0007	DT = 0.0004	NI = 1	LI=1\n"
        "# 4 	T = 0.0015	DT = 0.0008	NI = 1	LI=1\n"
    )
    mock_subprocess_run.return_value = mock_process

    results = OpenDartsConnector.run("test_config")

    assert results == {"heat": [2312.12, [1.23, 4.56, 7.89]]}


def test_run_success_with_single_broadcasted_value(mock_subprocess_run: Mock) -> None:
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = (
        "Building connection list...\n"
        "OpenDartsConnector: Type:heat, Value:2312.12\n"
        "# 1 	T = 0.0001	DT = 0.0001	NI = 2	LI=2\n"
        "# 2 	T = 0.0003	DT = 0.0002	NI = 1	LI=1\n"
        "# 3 	T = 0.0007	DT = 0.0004	NI = 1	LI=1\n"
        "# 4 	T = 0.0015	DT = 0.0008	NI = 1	LI=1\n"
    )
    mock_subprocess_run.return_value = mock_process

    results = OpenDartsConnector.run("test_config")

    assert results == {"heat": 2312.12}


def test_run_failure(mock_subprocess_run: Mock) -> None:
    mock_process = MagicMock()
    mock_process.returncode = 1
    mock_process.stdout = ""
    mock_subprocess_run.return_value = mock_process

    with pytest.raises(RuntimeError):
        _ = OpenDartsConnector.run("test_config")


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


def test_decorator_with_invalid_json(monkeypatch, capsys):  # type: ignore
    # Mock sys.argv to simulate command-line arguments with invalid JSON
    invalid_json = '{"key": "value", "number": 42'
    monkeypatch.setattr(sys, "argv", ["main.py", invalid_json])

    # Mock sys.exit to prevent the test from stopping
    with patch("sys.exit") as mock_exit:
        main()
        mock_exit.assert_called_once_with(1)

    # Capture printed output and verify the error message
    captured = capsys.readouterr()
    assert "Invalid JSON input." in captured.out


def test_decorator_with_missing_argument(monkeypatch, capsys):  # type: ignore
    # Mock sys.argv to simulate no arguments provided
    monkeypatch.setattr(sys, "argv", ["main.py"])

    # Mock sys.exit to raise an exception
    with patch("sys.exit", side_effect=SystemExit) as mock_exit:
        with pytest.raises(SystemExit):
            main()
        mock_exit.assert_called_once_with(1)

    # Capture printed output and verify the usage message
    captured = capsys.readouterr()
    assert "Usage: python main.py 'configuration.json (TBD)'" in captured.out
