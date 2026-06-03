from pathlib import Path
from unittest.mock import MagicMock, patch

import grpc
import pytest

import orchestration.risk_management_service.core.service.risk_management_service as rms
from common.models import RunMode
from services.problem_dispatcher_service import ServiceType


@patch(
    "orchestration.risk_management_service.core.service.risk_management_service.simulation_cluster_context_manager"
)
@patch(
    "orchestration.risk_management_service.core.service.risk_management_service.SimulationService"
)
@patch(
    "orchestration.risk_management_service.core.service.risk_management_service.SolutionUpdaterService"
)
@patch(
    "orchestration.risk_management_service.core.service.risk_management_service.ProblemDispatcherService"
)
def test_run_risk_management_happy_path(
    mock_dispatcher,
    mock_su,
    mock_sim_service,
    mock_sim_cluster_ctx,
):
    mock_ctx = MagicMock()
    mock_sim_cluster_ctx.return_value.__enter__.return_value = mock_ctx
    mock_sim_service.transfer_simulation_model.return_value = None
    mock_sim_service.process_request.return_value = MagicMock(
        simulation_cases=[
            MagicMock(
                control_vector={"a": 1}, results=MagicMock(model_dump=lambda: {"r": 2})
            )
        ]
    )

    mock_dispatcher_inst = MagicMock()
    mock_dispatcher.return_value = mock_dispatcher_inst
    mock_dispatcher_inst.expected_optimization_function_names = ["r"]
    mock_dispatcher_inst.optimization_objectives = {"r": "minimize"}
    mock_dispatcher_inst.population_size = 1
    mock_dispatcher_inst.max_generation = 1
    mock_dispatcher_inst.max_stall_generations = 1
    mock_dispatcher_inst.full_key_boundaries = {}
    mock_dispatcher_inst.full_key_linear_inequalities = None
    minimal_well = {
        "well_type": "IWell",
        "name": "W1",
        "wellhead": {"x": 0.0, "y": 0.0, "z": 0.0},
        "md": 100.0,
        "md_step": 0.5,
    }

    mock_dispatcher_inst.process_iteration.side_effect = [
        MagicMock(
            solution_candidates=[
                MagicMock(
                    tasks={
                        ServiceType.DomainService: MagicMock(
                            request=[minimal_well],
                            control_vector=MagicMock(items={"a": 1}),
                        )
                    }
                )
            ]
        ),
        MagicMock(solution_candidates=[]),
    ]

    mock_su_inst = MagicMock()
    mock_su.return_value = mock_su_inst
    mock_su_inst.loop_controller.running.side_effect = [True, False]
    mock_su_inst.loop_controller.current_generation = 1
    mock_su_inst.get_generation_summary.return_value = MagicMock(
        global_best=1.0,
        min=1.0,
        max=2.0,
        avg=1.5,
        std=0.5,
        population=[1.0],
    )
    mock_su_inst.process_request.return_value = MagicMock(
        next_iter_solutions=[{"a": 1}]
    )

    mock_su_inst.global_best_result = 1.23
    mock_su_inst.global_best_control_vector = MagicMock(items={"x": 5})

    with (
        patch(
            "orchestration.risk_management_service.core.service.risk_management_service.parse_flat_dict_to_nested",
            return_value={"x": 5},
        ),
        patch.dict("os.environ", {"RUNNER_MODE": "docker"}),
    ):
        mock_problem_def = MagicMock()
        mock_problem_def.optimization_parameters.worker_count = 1
        mock_problem_def.optimization_parameters.population_size = 1
        mock_problem_def.optimization_parameters.max_stall_generations = 1
        mock_problem_def.optimization_parameters.max_generations = 1
        mock_problem_def.run_mode.value = "optimization"
        mock_problem_def.plugins.connector = "opendarts"
        mock_problem_def.plugins.optimizer = "pso"
        mock_problem_def.plugins.domain_service = "builtin"

        rms.run_risk_management(
            mock_problem_def, b"model", model_hash="brooks_was_here"
        )

    mock_sim_service.transfer_simulation_model.assert_called_once()
    assert mock_dispatcher_inst.process_iteration.call_count == 1
    mock_su_inst.process_request.assert_called()


def test_prepare_simulation_cases_basic():
    minimal_well = {
        "well_type": "IWell",
        "name": "W1",
        "wellhead": {"x": 0.0, "y": 0.0, "z": 0.0},
        "md": 100.0,
        "md_step": 0.5,
    }
    fake_task = MagicMock()
    fake_task.request = [minimal_well]
    fake_task.control_vector = MagicMock(items={"a": 1})
    fake_solution = MagicMock()
    fake_solution.tasks = {ServiceType.DomainService: fake_task}
    solutions = MagicMock(solution_candidates=[fake_solution])

    mock_handler = MagicMock()
    mock_handler.process_request.return_value = {"well": 1}
    fake_expected_cost_function_names = ["cost_function_1", "cost_function_2"]
    sim_cases = rms._prepare_simulation_cases(
        solutions, fake_expected_cost_function_names, mock_handler
    )
    assert isinstance(sim_cases, list)
    assert sim_cases[0]["wells"] == {"well": 1}
    assert sim_cases[0]["control_vector"] == {"a": 1}
    mock_handler.process_request.assert_called_once_with({"models": [minimal_well]})


def test_prepare_simulation_cases_empty_tasks():
    fake_solution = MagicMock()
    fake_solution.tasks = {}
    solutions = MagicMock(solution_candidates=[fake_solution])
    fake_expected_cost_function_names = ["cost_function_1", "cost_function_2"]

    mock_handler = MagicMock()
    sim_cases = rms._prepare_simulation_cases(
        solutions, fake_expected_cost_function_names, mock_handler
    )
    assert isinstance(sim_cases, list)
    assert "control_vector" in sim_cases[0]
    mock_handler.process_request.assert_not_called()


@patch(
    "orchestration.risk_management_service.core.service.risk_management_service.simulation_cluster_context_manager"
)
@patch(
    "orchestration.risk_management_service.core.service.risk_management_service.SimulationService"
)
def test_run_risk_management_exception(
    mock_sim_service,
    mock_sim_cluster_ctx,
):
    # We have so many patches unused because we patch all dependencies that the function under test touches,
    # to prevent side effects and to control the test environment.
    mock_ctx = MagicMock()
    mock_sim_cluster_ctx.return_value.__enter__.return_value = mock_ctx
    mock_sim_service.transfer_simulation_model.side_effect = Exception("fail")

    mock_problem_def = MagicMock()
    mock_problem_def.optimization_parameters.worker_count = 1
    mock_problem_def.plugins.connector = "opendarts"

    with pytest.raises(Exception):
        rms.run_risk_management(mock_problem_def, b"model")


# ---------------------------------------------------------------------------
# Helper: reusable patches for run_risk_management tests
# ---------------------------------------------------------------------------

_RMS = "orchestration.risk_management_service.core.service.risk_management_service"


def _make_problem_def(run_mode=RunMode.Optimization):
    pd = MagicMock()
    pd.run_mode = run_mode  # real enum — .value is read from the enum directly
    pd.plugins.connector = "opendarts"
    pd.plugins.optimizer = "pso"
    pd.plugins.domain_service = "builtin"
    pd.optimization_parameters.worker_count = 1
    pd.optimization_parameters.seed = 42
    pd.simulation_config.worker_count = 1
    pd.simulation_config.worker_simulation_timeout_seconds = 900
    pd.simulation_config.server_job_timeout_seconds = 60
    pd.simulation_config.checkpoint_interval = 10
    pd.simulation_config.checkpoint_path = "/tmp"
    pd.well_design = []
    return pd


# ---------------------------------------------------------------------------
# Evaluation mode (lines 118-138)
# ---------------------------------------------------------------------------


@patch(f"{_RMS}.simulation_process_context_manager")
@patch(f"{_RMS}.SimulationService")
@patch(f"{_RMS}.ProblemDispatcherService")
def test_run_risk_management_evaluation_mode(mock_dispatcher, mock_sim, mock_ctx):
    mock_ctx.return_value.__enter__ = MagicMock(return_value=None)
    mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

    mock_dispatcher_inst = MagicMock()
    mock_dispatcher.return_value = mock_dispatcher_inst
    mock_dispatcher_inst.expected_optimization_function_names = ["cost"]
    mock_dispatcher_inst.process_iteration.return_value = MagicMock(
        solution_candidates=[]
    )

    mock_sim.process_request.return_value = MagicMock(
        simulation_cases=[MagicMock(results={"cost": 1.0})]
    )

    problem_def = _make_problem_def(RunMode.Evaluation)

    result = rms.run_risk_management(problem_def, b"model", model_hash="abc")

    assert result is None
    mock_dispatcher_inst.process_iteration.assert_called_once_with(None)


# ---------------------------------------------------------------------------
# KeyboardInterrupt is caught and returns None (lines 239-241)
# ---------------------------------------------------------------------------


@patch(f"{_RMS}.simulation_process_context_manager")
@patch(f"{_RMS}.SimulationService")
@patch(f"{_RMS}.ProblemDispatcherService")
@patch(f"{_RMS}.SolutionUpdaterService")
def test_run_risk_management_keyboard_interrupt(
    mock_su, mock_dispatcher, mock_sim, mock_ctx
):
    mock_ctx.return_value.__enter__ = MagicMock(return_value=None)
    mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

    mock_sim.transfer_simulation_model.side_effect = KeyboardInterrupt()

    problem_def = _make_problem_def()

    result = rms.run_risk_management(problem_def, b"model", model_hash="abc")

    assert result is None


# ---------------------------------------------------------------------------
# gRPC shutdown error is re-raised (lines 243-254)
# ---------------------------------------------------------------------------


class _FakeRpcError(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode, details: str = "") -> None:
        self._code = code
        self._details = details

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details


@patch(f"{_RMS}.simulation_process_context_manager")
@patch(f"{_RMS}.SimulationService")
def test_run_risk_management_grpc_aborted_reraises(mock_sim, mock_ctx):
    mock_ctx.return_value.__enter__ = MagicMock(return_value=None)
    mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
    mock_sim.transfer_simulation_model.side_effect = _FakeRpcError(
        grpc.StatusCode.ABORTED, "job cancelled"
    )

    problem_def = _make_problem_def()

    with pytest.raises(grpc.RpcError):
        rms.run_risk_management(problem_def, b"model", model_hash="abc")


@patch(f"{_RMS}.simulation_process_context_manager")
@patch(f"{_RMS}.SimulationService")
def test_run_risk_management_grpc_unexpected_reraises(mock_sim, mock_ctx):
    mock_ctx.return_value.__enter__ = MagicMock(return_value=None)
    mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
    mock_sim.transfer_simulation_model.side_effect = _FakeRpcError(
        grpc.StatusCode.INTERNAL, "internal error"
    )

    problem_def = _make_problem_def()

    with pytest.raises(grpc.RpcError):
        rms.run_risk_management(problem_def, b"model", model_hash="abc")


# ---------------------------------------------------------------------------
# Multi-solution (list) best_control_vector path (lines 266-273)
# ---------------------------------------------------------------------------


@patch(f"{_RMS}.simulation_process_context_manager")
@patch(f"{_RMS}.SimulationService")
@patch(f"{_RMS}.ProblemDispatcherService")
@patch(f"{_RMS}.SolutionUpdaterService")
@patch(f"{_RMS}.parse_flat_dict_to_nested", return_value={"x": 1})
def test_run_risk_management_list_control_vector(
    _mock_parse, mock_su, mock_dispatcher, mock_sim, mock_ctx
):
    mock_ctx.return_value.__enter__ = MagicMock(return_value=None)
    mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

    mock_dispatcher_inst = MagicMock()
    mock_dispatcher.return_value = mock_dispatcher_inst
    mock_dispatcher_inst.expected_optimization_function_names = ["cost"]
    mock_dispatcher_inst.optimization_objectives = {"cost": "minimize"}
    mock_dispatcher_inst.full_key_boundaries = {}
    mock_dispatcher_inst.full_key_linear_inequalities = None

    mock_su_inst = MagicMock()
    mock_su.return_value = mock_su_inst
    mock_su_inst.loop_controller.running.return_value = False
    mock_su_inst.global_best_result = MagicMock()
    mock_su_inst.global_best_result_descriptive = [{"cost": 1.0}]
    cv1 = MagicMock()
    cv1.items = {"x": 1}
    cv2 = MagicMock()
    cv2.items = {"x": 2}
    mock_su_inst.global_best_control_vector = [cv1, cv2]

    problem_def = _make_problem_def()

    result = rms.run_risk_management(problem_def, b"model", model_hash="abc")

    assert result is not None
    best, cv = result
    assert isinstance(cv, list)


# ---------------------------------------------------------------------------
# _validate_well_initial_states raises ValueError on ValidationError (lines 344-352)
# ---------------------------------------------------------------------------


def test_validate_well_initial_states_validation_error():
    from pydantic import BaseModel, TypeAdapter

    class _Schema(BaseModel):
        value: int

    type_adapter = TypeAdapter(_Schema)

    mock_item = MagicMock()
    mock_item.well_name = "W1"
    mock_item.initial_state = {"value": "not_a_number"}

    mock_plugin = MagicMock()
    mock_plugin.implementation.get_well_state_adapter.return_value = type_adapter

    mock_pd = MagicMock()
    mock_pd.well_design = [mock_item]

    with pytest.raises(ValueError, match="W1"):
        rms._validate_well_initial_states(mock_pd, mock_plugin, "test_plugin")


def test_validate_well_initial_states_happy():
    from pydantic import BaseModel, TypeAdapter

    class _Schema(BaseModel):
        value: int

    type_adapter = TypeAdapter(_Schema)

    mock_item = MagicMock()
    mock_item.well_name = "W1"
    mock_item.initial_state = {"value": 42}

    mock_plugin = MagicMock()
    mock_plugin.implementation.get_well_state_adapter.return_value = type_adapter

    mock_pd = MagicMock()
    mock_pd.well_design = [mock_item]

    rms._validate_well_initial_states(mock_pd, mock_plugin, "test_plugin")


# ---------------------------------------------------------------------------
# _save_checkpoint happy path and exception swallowing (lines 362-373)
# ---------------------------------------------------------------------------


def test_save_checkpoint_happy(tmp_path: Path):
    mock_su = MagicMock()
    mock_su.get_checkpoint_state.return_value = {}
    mock_su.loop_controller.current_generation = 5

    with patch(f"{_RMS}.save_checkpoint") as mock_save:
        rms._save_checkpoint(
            mock_su, [], tmp_path / "ckpt.json", MagicMock(), "hash123", "run1"
        )

    mock_save.assert_called_once()


def test_save_checkpoint_swallows_exception(tmp_path: Path):
    mock_su = MagicMock()
    mock_su.get_checkpoint_state.side_effect = RuntimeError("disk full")

    # Should not raise — exception is caught and logged as a warning.
    rms._save_checkpoint(
        mock_su, [], tmp_path / "ckpt.json", MagicMock(), "hash123", "run1"
    )


# ---------------------------------------------------------------------------
# Checkpoint restore path (lines 163-168) and checkpoint clear (lines 235-237)
# ---------------------------------------------------------------------------


@patch(f"{_RMS}.simulation_process_context_manager")
@patch(f"{_RMS}.SimulationService")
@patch(f"{_RMS}.ProblemDispatcherService")
@patch(f"{_RMS}.SolutionUpdaterService")
@patch(f"{_RMS}.parse_flat_dict_to_nested", return_value={"x": 1})
def test_run_risk_management_with_checkpoint_restore(
    _mock_parse, mock_su, mock_dispatcher, mock_sim, mock_ctx
):
    mock_ctx.return_value.__enter__ = MagicMock(return_value=None)
    mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

    mock_dispatcher_inst = MagicMock()
    mock_dispatcher.return_value = mock_dispatcher_inst
    mock_dispatcher_inst.expected_optimization_function_names = ["cost"]
    mock_dispatcher_inst.optimization_objectives = {"cost": "minimize"}
    mock_dispatcher_inst.full_key_boundaries = {}
    mock_dispatcher_inst.full_key_linear_inequalities = None

    mock_su_inst = MagicMock()
    mock_su.return_value = mock_su_inst
    mock_su_inst.loop_controller.running.return_value = False
    mock_su_inst.restore_checkpoint_state.return_value = []
    mock_su_inst.global_best_result = MagicMock()
    mock_su_inst.global_best_result_descriptive = {"cost": 1.0}
    mock_su_inst.global_best_control_vector = MagicMock(items={"x": 1})

    problem_def = _make_problem_def()

    fake_checkpoint = MagicMock()

    result = rms.run_risk_management(
        problem_def, b"model", model_hash="abc", checkpoint=fake_checkpoint
    )

    mock_su_inst.restore_checkpoint_state.assert_called_once_with(fake_checkpoint)
    mock_su_inst.loop_controller.increment_generation.assert_called_once()
    assert result is not None


# ---------------------------------------------------------------------------
# Checkpoint file cleared after successful run (lines 235-237)
# ---------------------------------------------------------------------------


@patch(f"{_RMS}.simulation_process_context_manager")
@patch(f"{_RMS}.SimulationService")
@patch(f"{_RMS}.ProblemDispatcherService")
@patch(f"{_RMS}.SolutionUpdaterService")
@patch(f"{_RMS}.parse_flat_dict_to_nested", return_value={"x": 1})
def test_run_risk_management_checkpoint_file_cleared(
    _mock_parse, mock_su, mock_dispatcher, mock_sim, mock_ctx, tmp_path
):
    mock_ctx.return_value.__enter__ = MagicMock(return_value=None)
    mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

    mock_dispatcher_inst = MagicMock()
    mock_dispatcher.return_value = mock_dispatcher_inst
    mock_dispatcher_inst.expected_optimization_function_names = ["cost"]
    mock_dispatcher_inst.optimization_objectives = {"cost": "minimize"}
    mock_dispatcher_inst.full_key_boundaries = {}
    mock_dispatcher_inst.full_key_linear_inequalities = None

    mock_su_inst = MagicMock()
    mock_su.return_value = mock_su_inst
    mock_su_inst.loop_controller.running.return_value = False
    mock_su_inst.global_best_result = MagicMock()
    mock_su_inst.global_best_result_descriptive = {"cost": 1.0}
    mock_su_inst.global_best_control_vector = MagicMock(items={"x": 1})

    problem_def = _make_problem_def()
    problem_def.simulation_config.checkpoint_path = str(tmp_path)

    with patch(f"{_RMS}.checkpoint_filename", return_value="ckpt.json"):
        ckpt_file = tmp_path / "ckpt.json"
        ckpt_file.write_text("{}")
        assert ckpt_file.exists()

        rms.run_risk_management(problem_def, b"model", model_hash="abc")

    assert not ckpt_file.exists()
