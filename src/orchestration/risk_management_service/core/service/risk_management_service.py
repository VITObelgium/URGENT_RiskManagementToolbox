from typing import Any

from logger.numeric_logger import get_csv_logger
from logger import get_logger

from orchestration.risk_management_service.core.mappers import ControlVectorMapper
from services.problem_dispatcher_service import (
    ProblemDispatcherService,
    ServiceType,
)
from services.problem_dispatcher_service.core.utils.utils import (
    parse_flat_dict_to_nested,
)
from services.simulation_service import (
    SimulationService,
    simulation_cluster_context_manager,
)
from services.solution_updater_service import (
    OptimizationEngine,
    SolutionUpdaterService,
)
from services.solution_updater_service.core.utils import ensure_not_none
from services.well_management_service import WellManagementService

logger = get_logger(__name__)


def run_risk_management(
    problem_definition: dict[str, Any],
    simulation_model_archive: bytes | str,
    n_size: int = 10,
    patience: int = 10,
    max_generations: int = 10,
):
    """
    Main entry point for running risk management.

    Args:
        patience: Patience limit for the optimization process.
        max_generations: Maximum number of generations for the optimization process.
        problem_definition (dict[str, Any]): The problem definition used by the dispatcher.
        simulation_model_archive (bytes | str): The simulation model archive to transfer.
        n_size (int, optional): Number of samples for the dispatcher. Defaults to 10.
    """
    logger.info("Starting risk management process...")
    logger.debug(
        "Input problem definition: %s, simulation_model_archive: %s, n_size: %d",
        problem_definition,
        type(simulation_model_archive),
        n_size,
    )

    with simulation_cluster_context_manager():
        try:
            logger.info("Transferring simulation model archive to the cluster.")
            SimulationService.transfer_simulation_model(
                simulation_model_archive=simulation_model_archive
            )

            logger.info(
                "Initializing SolutionUpdaterService and ProblemDispatcherService."
            )
            solution_updater = SolutionUpdaterService(
                optimization_engine=OptimizationEngine.PSO,
                max_generations=max_generations,
                patience=patience,
            )
            dispatcher = ProblemDispatcherService(
                problem_definition=problem_definition, n_size=n_size
            )

            # Initialize metrics logger
            metrics_logger = get_csv_logger(
                "optimization_metrics.csv",
                columns=[
                    "generation",
                    "global_min",
                    "population_min",
                    "population_max",
                    "population_avg",
                    "population_std",
                ],
            )

            logger.info("Fetching boundaries from ProblemDispatcherService.")
            boundaries = dispatcher.get_boundaries()
            logger.debug("Boundaries retrieved: %s", boundaries)

            # Initialize solutions
            next_solutions = None

            loop_controller = solution_updater.loop_controller

            while loop_controller.running():
                logger.info(
                    "Starting generation %d for risk management.",
                    loop_controller.current_generation,
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
                        "optimization_constraints": {"boundaries": boundaries},
                    }
                )

                next_solutions = ControlVectorMapper.convert_su_to_pd(
                    response.next_iter_solutions
                )

                logger.info(
                    "Generation %d successfully completed for risk management.",
                    loop_controller.current_generation,
                )
                metrics = solution_updater.get_optimization_metrics()
                logger.info(
                    "Generation statistics: global_min=%.6f, population_min=%.6f, population_max=%.6f, population_avg=%.6f, population_std=%.6f",
                    metrics.global_min,
                    metrics.last_population_min,
                    metrics.last_population_max,
                    metrics.last_population_avg,
                    metrics.last_population_std,
                )

                # Log metrics to CSV
                metrics_logger.info(
                    f"{loop_controller.current_generation},{metrics.global_min:.9f},{metrics.last_population_min:.9f},{metrics.last_population_max:.9f},{metrics.last_population_avg:.9f},{metrics.last_population_std:.9f}"
                )

            logger.info(
                "Loop controller stopped at generation %d. Info: %s",
                loop_controller.current_generation,
                loop_controller.info,
            )

        except Exception as e:
            logger.error("Error in risk management process: %s", str(e), exc_info=True)
            raise

    logger.info(
        "Optimization results: Fitness value = %f Control vector = %s",
        solution_updater.global_best_result,
        parse_flat_dict_to_nested(solution_updater.global_best_controll_vector.items),
    )


def _prepare_simulation_cases(solutions):
    """
    Prepare simulation cases from generated candidates.

    Args:
        solutions: The generated solution candidates from the dispatcher.

    Returns:
        list: Simulation cases ready for processing by the simulation service.
    """
    logger.info("Preparing simulation cases.")
    sim_cases = []

    for index, solution in enumerate(solutions.solution_candidates):
        logger.debug("Processing solution candidate #%d: %s", index + 1, solution)
        sim_case, control_vector = (
            {},
            {},
        )  # sim_case should contain mandatory keys as implemented in SimulationCase (simulation_service/core/models/user.py

        for service, task in solution.tasks.items():
            match service:
                case ServiceType.WellManagementService:
                    logger.debug(
                        "Processing task for service: %s. Task details: %s",
                        service,
                        task,
                    )
                    wells = WellManagementService.process_request(
                        {"models": task.request}
                    )
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
