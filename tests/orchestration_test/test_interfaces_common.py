"""Exit-status handling of the ``interfaces.common.risk_management`` entry point."""

from unittest.mock import MagicMock, patch

import pytest

import interfaces.common as common

_COMMON = "interfaces.common"


@pytest.fixture
def _patched_entry(monkeypatch):
    monkeypatch.delenv("RUNNER_MODE", raising=False)
    with (
        patch(f"{_COMMON}.configure_logger"),
        patch(f"{_COMMON}.compute_file_hash", return_value="hash"),
        patch(f"{_COMMON}._load_problem_definition", return_value=(MagicMock(), None)),
        patch(f"{_COMMON}._try_zip_results") as mock_zip,
        patch(f"{_COMMON}.run_risk_management") as mock_run,
    ):
        yield mock_run, mock_zip


def test_none_result_completes_successfully(_patched_entry):
    """Evaluation mode and user interrupts return None; that is not a failure."""
    mock_run, mock_zip = _patched_entry
    mock_run.return_value = None

    assert common.risk_management(model_file="m.zip", config_file="c.json") is None
    mock_zip.assert_called_once()
    assert mock_zip.call_args.args[1] == common.RunStatus.COMPLETED


def test_result_is_unpacked(_patched_entry):
    mock_run, mock_zip = _patched_entry
    mock_run.return_value = MagicMock(values=1.5, control_vectors={"x": 1})

    result = common.risk_management(model_file="m.zip", config_file="c.json")

    assert result == (1.5, {"x": 1})
    assert mock_zip.call_args.args[1] == common.RunStatus.COMPLETED
