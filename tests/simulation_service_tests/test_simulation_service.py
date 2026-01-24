from pathlib import Path
from unittest.mock import MagicMock, patch

import grpc
import pytest

from services.simulation_service.core.service.simulation_service import (
    SimulationService,
)


@patch(
    "services.simulation_service.core.service.simulation_service.SimulationService._perform_simulations_on_cluster"
)
@patch(
    "services.simulation_service.core.service.simulation_service.SimulationServiceRequest"
)
@patch(
    "services.simulation_service.core.service.simulation_service.SimulationServiceResponse"
)
def test_process_request_valid(mock_response, mock_request, mock_perform):
    mock_request.return_value.simulation_cases = [MagicMock()]
    mock_perform.return_value = [MagicMock()]
    mock_response.return_value = "response"
    req = {"simulation_cases": [1, 2]}
    result = SimulationService.process_request(req)
    assert result == "response"
    mock_perform.assert_called_once()
    mock_response.assert_called_once()


@patch.object(SimulationService, "_transfer_model_archive")
def test_transfer_simulation_model_bytes(mock_transfer):
    SimulationService.transfer_simulation_model(b"abc")
    mock_transfer.assert_called_once_with(b"abc")


@patch.object(SimulationService, "_transfer_model_archive_from_path")
def test_transfer_simulation_model_str(mock_transfer):
    SimulationService.transfer_simulation_model("/tmp/fake.zip")
    mock_transfer.assert_called_once()
    assert isinstance(mock_transfer.call_args[0][0], Path)


def test_transfer_simulation_model_invalid():
    with pytest.raises(TypeError):
        SimulationService.transfer_simulation_model(123)


# 3. _transfer_model_archive: normal and grpc error
@patch(
    "services.simulation_service.core.service.simulation_service.GrpcStubManager.get_stub"
)
@patch("services.simulation_service.core.service.simulation_service.sm.SimulationModel")
def test__transfer_model_archive_success(mock_sim_model, mock_get_stub):
    stub = MagicMock()
    stub.TransferSimulationModel.return_value = MagicMock(message="ok")
    mock_get_stub.return_value.__enter__.return_value = stub
    SimulationService._transfer_model_archive(b"abc")
    stub.TransferSimulationModel.assert_called_once()


@patch(
    "services.simulation_service.core.service.simulation_service.GrpcStubManager.get_stub"
)
@patch("services.simulation_service.core.service.simulation_service.sm.SimulationModel")
def test__transfer_model_archive_grpc_error(mock_sim_model, mock_get_stub):
    stub = MagicMock()
    stub.TransferSimulationModel.side_effect = grpc.RpcError("fail")
    mock_get_stub.return_value.__enter__.return_value = stub
    with pytest.raises(grpc.RpcError):
        SimulationService._transfer_model_archive(b"abc")


@patch.object(SimulationService, "_transfer_model_archive")
def test__transfer_model_archive_from_path_exists(mock_transfer, tmp_path):
    file = tmp_path / "model.zip"
    file.write_bytes(b"abc")
    SimulationService._transfer_model_archive_from_path(file)
    mock_transfer.assert_called_once_with(b"abc")


def test__transfer_model_archive_from_path_not_exists():
    fake = Path("/tmp/doesnotexist.zip")
    with pytest.raises(FileNotFoundError):
        SimulationService._transfer_model_archive_from_path(fake)


@patch(
    "services.simulation_service.core.service.simulation_service.GrpcStubManager.get_stub"
)
@patch("services.simulation_service.core.service.simulation_service.sm.Simulations")
@patch.object(SimulationService, "_to_grpc")
@patch.object(SimulationService, "_from_grpc")
def test__perform_simulations_on_cluster_success(
    mock_from, mock_to, mock_simulations, mock_get_stub
):
    stub = MagicMock()
    sim_proto = MagicMock()
    mock_to.side_effect = lambda x: sim_proto
    mock_simulations.return_value = "sim_req"
    stub.PerformSimulations.return_value.simulations = [MagicMock(), MagicMock()]
    mock_get_stub.return_value.__enter__.return_value = stub
    mock_from.side_effect = ["case1", "case2"]
    cases = [MagicMock(), MagicMock()]
    result = SimulationService._perform_simulations_on_cluster(cases)
    assert result == ["case1", "case2"]
    stub.PerformSimulations.assert_called_once_with("sim_req")


@patch(
    "services.simulation_service.core.service.simulation_service.GrpcStubManager.get_stub"
)
@patch("services.simulation_service.core.service.simulation_service.sm.Simulations")
@patch.object(SimulationService, "_to_grpc")
def test__perform_simulations_on_cluster_grpc_error(
    mock_to, mock_simulations, mock_get_stub
):
    stub = MagicMock()
    mock_to.side_effect = lambda x: MagicMock()
    mock_simulations.return_value = "sim_req"
    stub.PerformSimulations.side_effect = grpc.RpcError("fail")
    mock_get_stub.return_value.__enter__.return_value = stub
    cases = [MagicMock()]
    with pytest.raises(grpc.RpcError):
        SimulationService._perform_simulations_on_cluster(cases)


@patch("services.simulation_service.core.service.simulation_service.sm.Simulation")
@patch("services.simulation_service.core.service.simulation_service.sm.SimulationInput")
@patch(
    "services.simulation_service.core.service.simulation_service.sm.SimulationControlVector"
)
@patch(
    "services.simulation_service.core.service.simulation_service.json_to_str",
    return_value="{}",
)
def test__to_grpc(mock_json_to_str, mock_control_vec, mock_input, mock_sim):
    case = MagicMock()
    case.wells.model_dump_json.return_value = "{}"
    case.control_vector = {}
    SimulationService._to_grpc(case)
    mock_input.assert_called_once_with(wells="{}")
    mock_control_vec.assert_called_once_with(content="{}")
    mock_sim.assert_called_once()


@patch("services.simulation_service.core.service.simulation_service.str_to_json")
def test__from_grpc(mock_str_to_json):
    from services.simulation_service.core.models import SimulationCase

    # Order of str_to_json calls:
    # 1. Parse result.result -> {"Heat": 0.0}
    # 2. Parse input.wells -> {"wells": []}
    # 3. Parse control_vector.content -> {"x": 1}
    mock_str_to_json.side_effect = [
        {"Heat": 0.0},
        {"wells": []},
        {"x": 1},
    ]

    sim = MagicMock()
    sim.input.wells = "{}"
    sim.result.result = '{"Heat": 0.0}'
    sim.control_vector.content = "{}"
    result = SimulationService._from_grpc(sim)
    assert isinstance(result, SimulationCase)
    assert result.wells is not None
    assert result.results is not None
    assert result.control_vector == {"x": 1}
