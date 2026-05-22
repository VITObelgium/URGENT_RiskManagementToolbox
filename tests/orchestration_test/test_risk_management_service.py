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
    fake_solution.tasks = {rms.ServiceType.WellDesignService: fake_task}
    solutions = MagicMock(solution_candidates=[fake_solution])

    mock_handler = MagicMock()
    mock_handler.build.return_value = MagicMock(
        model_dump=MagicMock(return_value={"well": 1})
    )
    fake_expected_cost_function_names = ["cost_function_1", "cost_function_2"]
    sim_cases = rms._prepare_simulation_cases(
        solutions, fake_expected_cost_function_names, mock_handler
    )
    assert isinstance(sim_cases, list)
    assert sim_cases[0]["wells"] == {"well": 1}
    assert sim_cases[0]["control_vector"] == {"a": 1}


def test_prepare_simulation_cases_unhandled_service():
    fake_task = MagicMock()
    fake_solution = MagicMock()
    fake_solution.tasks = {"UnknownService": fake_task}
    solutions = MagicMock(solution_candidates=[fake_solution])
    fake_expected_cost_function_names = ["cost_function_1", "cost_function_2"]

    mock_handler = MagicMock()
    sim_cases = rms._prepare_simulation_cases(
        solutions, fake_expected_cost_function_names, mock_handler
    )
    assert isinstance(sim_cases, list)
    assert "control_vector" in sim_cases[0]


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
