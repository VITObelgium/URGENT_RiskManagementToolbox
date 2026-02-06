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
                patience=dispatcher.patience,
                objectives=dispatcher.optimization_objectives,
            )

            # Initialize generation summary logger
            generation_summary_logger = get_csv_logger(
                "generation_summary.csv",
                logger_name="generation_summary_logger",
                columns=[
                    "generation",
                    "global_best",
                    "min",
                    "max",
                    "avg",
                    "std",
                ]
                + ["ind_" + str(idx) for idx in range(dispatcher.population_size)],
            )

            logger.debug("Fetching boundaries from ProblemDispatcherService.")
            boundaries = dispatcher.boundaries
            logger.debug("Boundaries retrieved: %s", boundaries)
            logger.debug("Fetching linear inequalities from ProblemDispatcherService.")
            linear_inequalities = dispatcher.linear_inequalities
            logger.debug("Linear inequalities retrieved: %s", linear_inequalities)

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
                sim_cases = _prepare_simulation_cases(solutions)
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
                            "values": ensure_not_none(
                                simulation_case.results
                            ).model_dump()
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
                        "optimization_constraints": {
                            "boundaries": boundaries,
                            "A": linear_inequalities["A"],
                            "b": linear_inequalities["b"],
                            "sense": linear_inequalities["sense"],
                        },
                    }
                )

                next_solutions = response.next_iter_solutions

                _log_generation_summary(
                    solution_updater, generation_summary_logger, loop_controller
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
            logger.error("Error in risk management process: %s", str(e), exc_info=True)
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
    solutions: ProblemDispatcherServiceResponse,
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
        sim_cases.append(sim_case)
        logger.debug("Simulation case #%d prepared: %s", index + 1, sim_case)

    logger.info("All simulation cases prepared. Total count: %d", len(sim_cases))
    return sim_cases


def _configure_generation_summary_logger(generation_summary_logger) -> None:
    generation_summary_logger.info("generation,global_best,min,max,avg,std,population")


def _log_generation_summary(
    solution_updater, generation_summary_logger, loop_controller
) -> None:
    generation_summary = solution_updater.get_generation_summary()

    logger.info(
        "Generation statistics: global_best=%.6f, min=%.6f, max=%.6f, avg=%.6f, std=%.6f",
        generation_summary.global_best,
        generation_summary.min,
        generation_summary.max,
        generation_summary.avg,
        generation_summary.std,
    )

    population_str = ",".join(f"{val:.9f}" for val in generation_summary.population)

    generation_summary_logger.info(
        f"{loop_controller.current_generation},"
        f"{generation_summary.global_best:.9f},"
        f"{generation_summary.min:.9f},"
        f"{generation_summary.max:.9f},"
        f"{generation_summary.avg:.9f},"
        f"{generation_summary.std:.9f},"
        f"{population_str}"
    )
