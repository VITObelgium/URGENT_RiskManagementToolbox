import os
import uuid
from typing import Any

import grpc
import numpy as np
import numpy.typing as npt

from common.models import RunMode
from logger import get_logger
from services.problem_dispatcher_service import (
    ProblemDispatcherService,
    ServiceType,
)
from services.problem_dispatcher_service.core.models import (
    ProblemDispatcherDefinition,
    ProblemDispatcherServiceResponse,
)
from services.problem_dispatcher_service.core.utils.utils import (
    parse_flat_dict_to_nested,
)
from services.shared import ensure_not_none
from services.simulation_service import (
    SimulationService,
    simulation_cluster_context_manager,
    simulation_process_context_manager,
)
from services.solution_updater_service import (
    OptimizationEngine,
    SolutionUpdaterService,
)

from services.well_management_service import WellDesignService

logger = get_logger(__name__)


def run_risk_management(
    problem_definition: ProblemDispatcherDefinition,
    simulation_model_archive: bytes | str,
) -> tuple[npt.NDArray[np.float64], Any] | None:
    """
    Main entry point for running risk management.

    Args:
        problem_definition: The problem definition used by the dispatcher.
        simulation_model_archive: The simulation model archive to transfer.

    Returns:
        Single-objective: (scalar_result, nested_control_vector_dict)
        Multi-objective:  (pareto_results_array, list[nested_control_vector_dict])
        None on evaluation mode or keyboard interrupt.
    """
    run_id = uuid.uuid4().hex[:8]
    os.environ["URGENT_RUN_ID"] = run_id

    logger.info("Starting risk management process (run_id=%s)...", run_id)
    logger.debug(
        "Input problem definition: %s, simulation_model_archive: %s",
        problem_definition,
        type(simulation_model_archive),
    )

    runner_mode = os.getenv("OPEN_DARTS_RUNNER", "thread").lower()
    os.environ["RUN_MODE"] = problem_definition.run_mode.value
    os.environ["WORKER_SIMULATION_TIMEOUT_SECONDS"] = str(
        problem_definition.simulation_config.worker_simulation_timeout_seconds
    )
    os.environ["SEVER_JOB_TIMEOUT_SECONDS"] = str(
        problem_definition.simulation_config.server_job_timeout_seconds
    )

    worker_count = problem_definition.simulation_config.worker_count
    cm = (
        simulation_process_context_manager(worker_count=worker_count)
        if runner_mode == "thread"
        else simulation_cluster_context_manager(worker_count=worker_count)
    )

    with cm:
        try:
            SimulationService.transfer_simulation_model(
                simulation_model_archive=simulation_model_archive
            )

            dispatcher = ProblemDispatcherService(problem_definition=problem_definition)
            run_mode = problem_definition.run_mode

            if run_mode == RunMode.Evaluation:
                logger.info(
                    "Run in evaluation mode: Running a single validation simulation."
                )
                solutions = dispatcher.process_iteration(None)
                expected_cost_function_names = (
                    dispatcher.expected_optimization_function_names
                )
                sim_cases = _prepare_simulation_cases(
                    solutions, expected_cost_function_names
                )
                logger.info(
                    "Submitting evaluation simulation case to SimulationService."
                )
                completed_cases = SimulationService.process_request(
                    {"simulation_cases": sim_cases}
                )
                logger.info(
                    f"Evaluation simulation completed successfully with results: "
                    f"{completed_cases.simulation_cases[0].results}"
                )
                return None

            solution_updater = SolutionUpdaterService(
                optimization_engine=OptimizationEngine.PSO,
                max_generations=dispatcher.max_generation,
                max_stall_generations=dispatcher.max_stall_generations,
                objectives=dispatcher.optimization_objectives,
                seed=problem_definition.optimization_parameters.seed,
            )

            logger.debug("Fetching full key boundaries from ProblemDispatcherService.")
            full_key_boundaries = dispatcher.full_key_boundaries
            logger.debug(f"Boundaries retrieved: {full_key_boundaries}")
            logger.debug("Fetching linear inequalities from ProblemDispatcherService.")
            full_key_linear_inequalities = dispatcher.full_key_linear_inequalities
            logger.debug(
                f"Linear inequalities retrieved: {full_key_linear_inequalities}"
            )

            next_solutions = None
            loop_controller = solution_updater.loop_controller

            while loop_controller.running():
                logger.info(
                    f"*** Starting generation {loop_controller.current_generation} ***",
                )
                solutions = dispatcher.process_iteration(next_solutions)
                logger.debug(f"Generated solutions: {solutions}")

                sim_cases = _prepare_simulation_cases(
                    solutions, dispatcher.expected_optimization_function_names
                )
                logger.debug(f"Prepared simulation cases: {sim_cases}")

                logger.info("Submitting simulation cases to SimulationService.")
                completed_cases = SimulationService.process_request(
                    {"simulation_cases": sim_cases}
                )
                logger.debug(f"Completed simulation cases: {completed_cases}")

                updated_solutions = [
                    {
                        "control_vector": {"items": simulation_case.control_vector},
                        "cost_function_results": {
                            "values": ensure_not_none(simulation_case.results)
                        },
                    }
                    for simulation_case in completed_cases.simulation_cases
                ]
                logger.debug(
                    f"Updated solutions for next iteration: {updated_solutions}"
                )

                response = solution_updater.process_request(
                    {
                        "solution_candidates": updated_solutions,
                        "optimization_constrains": {
                            "boundaries": full_key_boundaries,
                            "linear_inequalities": full_key_linear_inequalities,
                        },
                    }
                )

                next_solutions = response.next_iter_solutions

                logger.info(
                    f"Generation {loop_controller.current_generation} successfully completed."
                )

                loop_controller.increment_generation()

            logger.info(
                f"Loop controller stopped at generation {loop_controller.current_generation}. "
                f"Info: {loop_controller.info}"
            )

        except KeyboardInterrupt:
            logger.warning("Risk management process interrupted by user.")
            return None

        except grpc.RpcError as e:
            code = e.code() if hasattr(e, "code") else None
            details = e.details() if hasattr(e, "details") else None

            if code in (grpc.StatusCode.ABORTED, grpc.StatusCode.CANCELLED) or (
                code == grpc.StatusCode.UNAVAILABLE
                and details == "Server shutting down"
            ):
                logger.info(
                    "Risk management process stopped due to gRPC server shutdown."
                )
                raise

            logger.error(f"Error in risk management process: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error in risk management process: {str(e)}")
            raise

    best_result = solution_updater.global_best_result
    best_result_descriptive = solution_updater.global_best_result_descriptive
    best_control_vector = solution_updater.global_best_control_vector

    if isinstance(best_control_vector, list):
        # Multi-objective: one ControlVector per Pareto solution.
        best_control_vector_nested: list[dict[str, Any]] | dict[str, Any] = [
            parse_flat_dict_to_nested(cv.items) for cv in best_control_vector
        ]
        logger.info(
            f"Optimization results (Pareto front, {len(best_control_vector)} solutions): "
            f"Fitness values = {best_result_descriptive}, "
            f"Control vectors = {best_control_vector_nested}"
        )
    else:
        # Single-objective: one ControlVector.
        best_control_vector_nested = parse_flat_dict_to_nested(
            best_control_vector.items
        )
        logger.info(
            f"Optimization results: Fitness value(s) = {best_result_descriptive}, "
            f"Control vector = {best_control_vector_nested}"
        )

    return best_result, best_control_vector_nested


def _prepare_simulation_cases(
    solutions: ProblemDispatcherServiceResponse,
    expected_cost_function_names: list[str],
) -> list[dict[str, Any]]:
    """
    Prepare simulation cases from generated candidates.

    Args:
        solutions: The generated solution candidates from the dispatcher.
        expected_cost_function_names: Names of the cost functions to initialise
            result placeholders with NaN.

    Returns:
        Simulation cases ready for processing by the simulation service.
    """
    logger.debug("Preparing simulation cases.")
    sim_cases: list[dict[str, Any]] = []

    for index, solution in enumerate(solutions.solution_candidates):
        logger.debug(f"Processing solution candidate #{index + 1}: {solution}")
        sim_case: dict[str, Any] = {}
        control_vector: dict[str, Any] = {}

        for service, task in solution.tasks.items():
            match service:
                case ServiceType.WellDesignService:
                    logger.debug(
                        f"Processing task for service: {service}. Task details: {task}"
                    )
                    wells = WellDesignService.process_request({"models": task.request})
                    sim_case["wells"] = wells.model_dump()
                    control_vector.update(task.control_vector.items)
                    logger.debug("Processed wells: %s", wells)
                case _:
                    logger.warning(f"Service not implemented: {service}")

        sim_case["control_vector"] = control_vector
        sim_case["results"] = {k: float("nan") for k in expected_cost_function_names}
        sim_cases.append(sim_case)
        logger.debug(f"Simulation case #{index + 1} prepared: {sim_case}")

    logger.info(f"All simulation cases prepared. Total count: {len(sim_cases)}")
    return sim_cases
