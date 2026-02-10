import os
from typing import Any

import numpy as np
import numpy.typing as npt

from logger import get_csv_logger, get_logger
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
from services.simulation_service import (
    SimulationService,
    simulation_cluster_context_manager,
    simulation_process_context_manager,
)
from services.solution_updater_service import (
    OptimizationEngine,
    SolutionUpdaterService,
)
from services.solution_updater_service.core.utils import ensure_not_none
from services.well_management_service import WellDesignService

logger = get_logger(__name__)


def run_risk_management(
    problem_definition: ProblemDispatcherDefinition,
    simulation_model_archive: bytes | str,
) -> tuple[float | npt.NDArray[np.float64], Any] | None:
    """
    Main entry point for running risk management.

    Args:
        problem_definition (ProblemDispatcherDefinition): The problem definition used by the dispatcher.
        simulation_model_archive (bytes | str): The simulation model archive to transfer.
    """
    logger.info("Starting risk management process...")
    logger.debug(
        "Input problem definition: %s, simulation_model_archive: %s",
        problem_definition,
        type(simulation_model_archive),
    )

    runner_mode = os.getenv("OPEN_DARTS_RUNNER", "thread").lower()

    worker_count = problem_definition.optimization_parameters.worker_count
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

            solution_updater = SolutionUpdaterService(
                optimization_engine=OptimizationEngine.PSO,
                max_generations=dispatcher.max_generation,
                max_stall_generations=dispatcher.max_stall_generations,
                objectives=dispatcher.optimization_objectives,
            )

            n_objectives = len(dispatcher.optimization_objectives)
            population_size = dispatcher.population_size

            # Initialize generation summary logger
            generation_summary_logger = get_csv_logger(
                "generation_summary.csv",
                logger_name="generation_summary_logger",
                columns=_generation_csv_columns(
                    dispatcher.expected_optimization_function_names, population_size
                ),
            )

            logger.debug("Fetching full key boundaries from ProblemDispatcherService.")
            full_key_boundaries = dispatcher.full_key_boundaries
            logger.debug("Boundaries retrieved: %s", full_key_boundaries)
            logger.debug("Fetching linear inequalities from ProblemDispatcherService.")
            full_key_linear_inequalities = dispatcher.full_key_linear_inequalities
            logger.debug(
                "Linear inequalities retrieved: %s", full_key_linear_inequalities
            )

            # Initialize solutions
            next_solutions = None

            loop_controller = solution_updater.loop_controller

            while loop_controller.running():
                logger.info(
                    f"Starting generation {loop_controller.current_generation}",
                )
                # Generate or update solutions
                solutions = dispatcher.process_iteration(next_solutions)
                logger.debug("Generated solutions: %s", solutions)

                # Prepare simulation cases
                sim_cases = _prepare_simulation_cases(
                    solutions, dispatcher.expected_optimization_function_names
                )
                logger.debug("Prepared simulation cases: %s", sim_cases)

                # Process simulation with the simulation service
                logger.info("Submitting simulation cases to SimulationService.")
                completed_cases = SimulationService.process_request(
                    {"simulation_cases": sim_cases}
                )
                logger.debug("Completed simulation cases: %s", completed_cases)

                # Update solutions based on simulation results
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
                    "Updated solutions for next iteration: %s", updated_solutions
                )

                # Map simulation service solutions to the ProblemDispatcherService format
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

                _log_generation_summary(
                    solution_updater,
                    generation_summary_logger,
                    loop_controller,
                    n_objectives=n_objectives,
                    population_size=population_size,
                )

                logger.info(
                    "Generation %d successfully completed.",
                    loop_controller.current_generation,
                )

                loop_controller.increment_generation()

            logger.info(
                "Loop controller stopped at generation %d. Info: %s",
                loop_controller.current_generation,
                loop_controller.info,
            )

        except KeyboardInterrupt:
            logger.warning("Risk management process interrupted by user.")
            return None
        except Exception as e:
            # logger.error("Error in risk management process: %s", str(e), exc_info=True)
            logger.error("Error in risk management process: %s", str(e))
            raise

    logger.info(
        "Optimization results: Fitness value = %f Control vector = %s",
        solution_updater.global_best_result,
        parse_flat_dict_to_nested(solution_updater.global_best_control_vector.items),
    )
    return solution_updater.global_best_result, parse_flat_dict_to_nested(
        solution_updater.global_best_control_vector.items
    )


def _prepare_simulation_cases(
    solutions: ProblemDispatcherServiceResponse, expected_cost_function_names: list[str]
) -> list[dict[(str, Any)]]:
    """
    Prepare simulation cases from generated candidates.

    Args:
        solutions: The generated solution candidates from the dispatcher.

    Returns:
        list: Simulation cases ready for processing by the simulation service.
    """
    logger.debug("Preparing simulation cases.")
    sim_cases = []

    for index, solution in enumerate(solutions.solution_candidates):
        logger.debug("Processing solution candidate #%d: %s", index + 1, solution)
        sim_case, control_vector = (
            {},
            {},
        )  # sim_case should contain mandatory keys as implemented in SimulationCase (simulation_service/core/models/user.py

        for service, task in solution.tasks.items():
            match service:
                case ServiceType.WellDesignService:
                    logger.debug(
                        "Processing task for service: %s. Task details: %s",
                        service,
                        task,
                    )
                    wells = WellDesignService.process_request({"models": task.request})
                    sim_case["wells"] = wells.model_dump()
                    control_vector.update(task.control_vector.items)
                    logger.debug("Processed wells: %s", wells)
                case _:
                    logger.warning("Service not implemented: %s", service)

        sim_case["control_vector"] = control_vector
        sim_case["results"] = {k: float("nan") for k in expected_cost_function_names}
        sim_cases.append(sim_case)
        logger.debug("Simulation case #%d prepared: %s", index + 1, sim_case)

    logger.info("All simulation cases prepared. Total count: %d", len(sim_cases))
    return sim_cases


def _log_generation_summary(
    solution_updater,
    generation_summary_logger,
    loop_controller,
    *,
    n_objectives: int,
    population_size: int,
) -> None:
    generation_summary = solution_updater.get_generation_summary()

    def _to_obj_list(x: float | npt.NDArray[np.float64], *, name: str) -> list[float]:
        arr = np.asarray(x, dtype=float).ravel()
        if arr.size == 1 and n_objectives == 1:
            return [float(arr[0])]
        if arr.size != n_objectives:
            raise ValueError(
                f"Expected {name} to have {n_objectives} objective values, got {arr.size}: {arr!r}"
            )
        return [float(v) for v in arr.tolist()]

    # Log to normal logger (compact but still numeric)
    gb = _to_obj_list(generation_summary.global_best, name="global_best")
    mn = _to_obj_list(generation_summary.min, name="min")
    mx = _to_obj_list(generation_summary.max, name="max")
    av = _to_obj_list(generation_summary.avg, name="avg")
    sd = _to_obj_list(generation_summary.std, name="std")

    logger.info(
        "Generation statistics: global_best=%s, min=%s, max=%s, avg=%s, std=%s",
        gb,
        mn,
        mx,
        av,
        sd,
    )

    pop = generation_summary.population
    if len(pop) != population_size:
        raise ValueError(f"Expected population_size={population_size}, got {len(pop)}")

    pop_values: list[float] = []
    for idx, item in enumerate(pop):
        item_vals = _to_obj_list(item, name=f"population[{idx}]")
        pop_values.extend(item_vals)

    row_values: list[float] = gb + mn + mx + av + sd + pop_values
    row_str = ",".join(f"{v:.9f}" for v in row_values)

    generation_summary_logger.info(f"{loop_controller.current_generation},{row_str}")


def _sanitize_col(name: str) -> str:
    # Keep CSV headers simple/stable (avoid commas/spaces)
    return (
        name.strip()
        .replace(",", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def _generation_csv_columns(
    expected_optimization_function_names: list[str],
    pop_size: int,
) -> list[str]:
    metrics = ["global_best", "min", "max", "avg", "std"]
    obj_cols = [_sanitize_col(o) for o in expected_optimization_function_names]

    cols = ["generation"]
    for metric in metrics:
        cols += [f"{metric}_{obj}" for obj in obj_cols]
    cols += [f"ind_{i}_{obj}" for i in range(pop_size) for obj in obj_cols]
    return cols
