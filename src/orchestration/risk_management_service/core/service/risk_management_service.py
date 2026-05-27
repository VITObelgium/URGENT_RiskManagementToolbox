import os
from pathlib import Path
from typing import Any

import grpc
import numpy as np
import numpy.typing as npt
from common.models import RunMode
from logger import get_logger
from orchestration.risk_management_service.core.service.checkpoint import (
    checkpoint_filename,
    save_checkpoint,
    LoadedCheckpointData,
)
from services.problem_dispatcher_service import ProblemDispatcherService
from services.problem_dispatcher_service.core.models import (
    ProblemDispatcherDefinition,
    ProblemDispatcherServiceResponse,
)
from services.problem_dispatcher_service.core.utils import (
    parse_flat_dict_to_nested,
)
from services.shared import ensure_not_none
from services.simulation_service import (
    SimulationService,
    simulation_cluster_context_manager,
    simulation_process_context_manager,
)
from services.solution_updater_service import SolutionUpdaterService
from urgent_plugins import (
    PluginKind,
    get_registry,
    resolve_connector_plugin,
    resolve_domain_service_plugin,
    resolve_optimizer_plugin,
)

logger = get_logger(__name__)


def run_risk_management(
    problem_definition: ProblemDispatcherDefinition,
    simulation_model_archive: bytes | str,
    model_hash: str,
    checkpoint: LoadedCheckpointData | None = None,
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
    run_id = os.environ.get("URGENT_RUN_ID", "default")

    logger.info("Starting risk management process (run_id=%s)...", run_id)
    logger.debug(
        "Input: problem_definition=%s, archive_type=%s",
        problem_definition,
        type(simulation_model_archive),
    )

    runner_mode = os.getenv("RUNNER_MODE", "thread").lower()
    os.environ["RUN_MODE"] = problem_definition.run_mode.value
    connector_name = resolve_connector_plugin(problem_definition.plugins.connector)
    optimizer_name = resolve_optimizer_plugin(problem_definition.plugins.optimizer)
    domain_service_name = resolve_domain_service_plugin(
        problem_definition.plugins.domain_service
    )
    domain_service_plugin = get_registry().require(
        PluginKind.DOMAIN_SERVICE, domain_service_name
    )
    domain_service = domain_service_plugin.implementation()
    logger.info(
        "Running with %s Connector, %s Optimizer, and %s DomainService",
        connector_name,
        optimizer_name,
        domain_service_name,
    )
    os.environ["WORKER_SIMULATION_TIMEOUT_SECONDS"] = str(
        problem_definition.simulation_config.worker_simulation_timeout_seconds
    )
    os.environ["SERVER_JOB_TIMEOUT_SECONDS"] = str(
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

            dispatcher = ProblemDispatcherService(
                problem_definition=problem_definition,
            )
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
                    solutions, expected_cost_function_names, domain_service
                )
                logger.info(
                    "Submitting evaluation simulation case to SimulationService."
                )
                completed_cases = SimulationService.process_request(
                    {"simulation_cases": sim_cases, "connector": connector_name}
                )
                logger.info(
                    f"Evaluation simulation completed successfully with results: "
                    f"{completed_cases.simulation_cases[0].results}"
                )
                return None

            solution_updater = SolutionUpdaterService(
                optimization_engine=optimizer_name,
                max_generations=dispatcher.max_generation,
                max_stall_generations=dispatcher.max_stall_generations,
                objectives=dispatcher.optimization_objectives,
                seed=problem_definition.optimization_parameters.seed,
            )

            full_key_boundaries = dispatcher.full_key_boundaries
            logger.debug("Boundaries retrieved: %s", full_key_boundaries)
            full_key_linear_inequalities = dispatcher.full_key_linear_inequalities
            logger.debug("Linear inequalities: %s", full_key_linear_inequalities)

            checkpoint_interval = (
                problem_definition.simulation_config.checkpoint_interval
            )
            checkpoint_dir = Path(problem_definition.simulation_config.checkpoint_path)
            checkpoint_file = checkpoint_dir / checkpoint_filename(run_id)

            loop_controller = solution_updater.loop_controller

            next_solutions = None
            if checkpoint is not None:
                logger.info("Restoring optimizer state from checkpoint.")
                next_solutions = solution_updater.restore_checkpoint_state(checkpoint)
                logger.info(
                    f"Resuming from generation {solution_updater.loop_controller.current_generation}."
                )
                loop_controller.increment_generation()

            while loop_controller.running():
                logger.info(
                    f"*** Starting generation {loop_controller.current_generation} ***",
                )
                solutions = dispatcher.process_iteration(next_solutions)
                logger.debug(
                    "Generated %d solution candidates.",
                    len(solutions.solution_candidates),
                )

                sim_cases = _prepare_simulation_cases(
                    solutions,
                    dispatcher.expected_optimization_function_names,
                    domain_service,
                )

                logger.debug("Prepared %d simulation cases.", len(sim_cases))
                logger.info("Submitting simulation cases to SimulationService.")
                completed_cases = SimulationService.process_request(
                    {"simulation_cases": sim_cases, "connector": connector_name}
                )

                updated_solutions = [
                    {
                        "control_vector": {"items": simulation_case.control_vector},
                        "cost_function_results": {
                            "values": ensure_not_none(simulation_case.results)
                        },
                    }
                    for simulation_case in completed_cases.simulation_cases
                ]

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

                if loop_controller.current_generation % checkpoint_interval == 0:
                    _save_checkpoint(
                        solution_updater,
                        next_solutions,
                        checkpoint_file,
                        problem_definition,
                        model_hash,
                        run_id,
                    )

                loop_controller.increment_generation()

            logger.info(
                f"Loop controller stopped at generation {loop_controller.current_generation}. "
                f"Info: {loop_controller.info}"
            )

            if checkpoint_file.exists():
                checkpoint_file.unlink()
                logger.info("Checkpoint cleared after successful completion.")

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
        best_control_vector_nested: list[dict[str, Any]] | dict[str, Any] = [
            parse_flat_dict_to_nested(cv.items) for cv in best_control_vector
        ]
        logger.info(
            f"Optimization results (Pareto front, {len(best_control_vector)} solutions): "
            f"Fitness values = {best_result_descriptive}, "
            f"Control vectors = {best_control_vector_nested}"
        )
    else:
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
    domain_service: Any,
) -> list[dict[str, Any]]:
    """Convert solution candidates into simulation-ready case dicts.

    Passes raw domain-service request dicts to ``domain_service.process_request``
    to produce simulation payloads, and initialises result
    placeholders for every expected cost-function name.

    Args:
        solutions: Candidates produced by ``ProblemDispatcherService``.
        expected_cost_function_names: Objective keys; results seeded with NaN.
        domain_service: Resolved domain service plugin instance.

    Returns:
        List of simulation case dicts ready for ``SimulationService.process_request``.
    """
    logger.debug("Preparing %d simulation cases.", len(solutions.solution_candidates))
    sim_cases: list[dict[str, Any]] = []

    for index, solution in enumerate(solutions.solution_candidates):
        logger.debug("Processing solution candidate #%d.", index + 1)
        sim_case: dict[str, Any] = {}
        control_vector: dict[str, Any] = {}

        for task in solution.tasks.values():
            result = domain_service.process_request({"models": task.request})
            sim_case["wells"] = result.model_dump()
            control_vector.update(task.control_vector.items)
            logger.debug(
                "Built %d domain item(s) for candidate #%d.",
                len(task.request),
                index + 1,
            )

        sim_case["control_vector"] = control_vector
        sim_case["results"] = {k: float("nan") for k in expected_cost_function_names}
        sim_cases.append(sim_case)

    logger.debug("All %d simulation cases prepared.", len(sim_cases))
    return sim_cases


def _save_checkpoint(
    solution_updater: SolutionUpdaterService,
    next_solutions,
    checkpoint_file: Path,
    problem_definition: ProblemDispatcherDefinition,
    model_hash: str,
    run_id: str,
) -> None:
    try:
        state = solution_updater.get_checkpoint_state(next_solutions)
        save_checkpoint(checkpoint_file, problem_definition, state, model_hash, run_id)
        logger.info(
            "Checkpoint for run ID %s saved at generation %s: %s",
            run_id,
            solution_updater.loop_controller.current_generation,
            checkpoint_file,
        )
    except Exception as e:
        logger.warning("Failed to save checkpoint: %s", e)
