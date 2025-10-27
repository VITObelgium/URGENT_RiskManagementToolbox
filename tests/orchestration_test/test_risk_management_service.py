from unittest.mock import MagicMock, patch

import pytest

import orchestration.risk_management_service.core.service.risk_management_service as rms


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
@patch(
    "orchestration.risk_management_service.core.service.risk_management_service.get_csv_logger"
)
@patch(
    "orchestration.risk_management_service.core.service.risk_management_service.ControlVectorMapper"
)
def test_run_risk_management_happy_path(
    mock_cvm,
    mock_csv_logger,
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
    mock_dispatcher_inst.get_boundaries.return_value = {"b": 3}
    mock_dispatcher_inst.process_iteration.side_effect = [
        MagicMock(
            solution_candidates=[
                MagicMock(
                    tasks={
                        "WellManagementService": MagicMock(
                            request=[1], control_vector=MagicMock(items={"a": 1})
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
    mock_su_inst.get_optimization_metrics.return_value = MagicMock(
        global_best=1.0,
        last_population_min=1.0,
        last_population_max=2.0,
        last_population_avg=1.5,
        last_population_std=0.5,
    )
    mock_su_inst.process_request.return_value = MagicMock(
        next_iter_solutions=[{"a": 1}]
    )
    mock_cvm.convert_su_to_pd.return_value = None
    mock_csv_logger.return_value = MagicMock(info=MagicMock())
    mock_su_inst.global_best_result = 1.23
    mock_su_inst.global_best_controll_vector = MagicMock(items={"x": 5})
    with patch(
        "orchestration.risk_management_service.core.service.risk_management_service.parse_flat_dict_to_nested",
        return_value={"x": 5},
    ):
        rms.run_risk_management(
            {"foo": "bar"}, b"model", n_size=1, patience=1, max_generations=1
        )
    mock_sim_service.transfer_simulation_model.assert_called_once()
    mock_dispatcher_inst.get_boundaries.assert_called_once()
    assert mock_dispatcher_inst.process_iteration.call_count == 1
    mock_su_inst.process_request.assert_called()
    mock_csv_logger.return_value.info.assert_called()


def test_prepare_simulation_cases_basic():
    fake_task = MagicMock()
    fake_task.request = [1]
    fake_task.control_vector = MagicMock(items={"a": 1})
    fake_solution = MagicMock()
    fake_solution.tasks = {rms.ServiceType.WellManagementService: fake_task}
    solutions = MagicMock(solution_candidates=[fake_solution])
    with patch(
        "orchestration.risk_management_service.core.service.risk_management_service.WellManagementService.process_request"
    ) as mock_well:
        mock_well.return_value.model_dump.return_value = {"well": 1}
        sim_cases = rms._prepare_simulation_cases(solutions)
        assert isinstance(sim_cases, list)
        assert sim_cases[0]["wells"] == {"well": 1}
        assert sim_cases[0]["control_vector"] == {"a": 1}


def test_prepare_simulation_cases_unhandled_service():
    fake_task = MagicMock()
    fake_solution = MagicMock()
    fake_solution.tasks = {"UnknownService": fake_task}
    solutions = MagicMock(solution_candidates=[fake_solution])

    sim_cases = rms._prepare_simulation_cases(solutions)
    assert isinstance(sim_cases, list)
    assert "control_vector" in sim_cases[0]


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
@patch(
    "orchestration.risk_management_service.core.service.risk_management_service.get_csv_logger"
)
@patch(
    "orchestration.risk_management_service.core.service.risk_management_service.ControlVectorMapper"
)
def test_run_risk_management_exception(
    _mock_cvm,
    _mock_csv_logger,
    _mock_dispatcher,
    _mock_su,
    mock_sim_service,
    mock_sim_cluster_ctx,
):
    # We have so many patches unused because we patch all dependencies that the function under test touches,
    # to prevent side effects and to control the test environment.
    mock_ctx = MagicMock()
    mock_sim_cluster_ctx.return_value.__enter__.return_value = mock_ctx
    mock_sim_service.transfer_simulation_model.side_effect = Exception("fail")
    with pytest.raises(Exception):
        rms.run_risk_management({"foo": "bar"}, b"model")
