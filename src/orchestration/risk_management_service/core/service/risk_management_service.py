"""Entry point and orchestration for the risk management optimization run.

The public entry point is :func:`run_risk_management`. The flow is:

1. Resolve runner mode and configure process-wide environment.
2. Transfer the simulation model and construct the dispatcher.
3. Either run a single evaluation simulation (``RunMode.Evaluation``) or the
   full optimization loop with periodic checkpointing.
4. Extract and return the best result(s).
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

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
from orchestration.risk_management_service.core.service.plugin_bootstrap import (
    bootstrap_run_plugins,
    ResolvedRunPlugins,
)
from services.problem_dispatcher_service import ProblemDispatcherService
from services.problem_dispatcher_service.core.models import (
    ProblemDispatcherDefinition,
    ProblemDispatcherServiceResponse,
)
from services.problem_dispatcher_service.core.service.interface import (
    DomainServiceInterface,
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

logger = get_logger(__name__)

_VALID_RUNNER_MODES = frozenset({"thread", "docker"})


class SimulationCaseDict(TypedDict):
    """Shape of a single simulation case sent to ``SimulationService``."""

    payload: dict[str, Any]
    control_vector: dict[str, Any]
    results: dict[str, float]


@dataclass(frozen=True)
class RiskManagementResult:
    """Outcome of an optimization run.

    Attributes:
        values: Scalar best fitness (single-objective) or Pareto-front array
            (multi-objective).
        control_vectors: Nested control-vector dict (single-objective) or a
            list of nested dicts, one per Pareto solution (multi-objective).
        is_pareto: True when the result represents a Pareto front.
    """

    values: npt.NDArray[np.float64] | float
    control_vectors: dict[str, Any] | list[dict[str, Any]]
    is_pareto: bool


def run_risk_management(
    problem_definition: ProblemDispatcherDefinition,
    simulation_model_archive: bytes | str,
    model_hash: str,
    checkpoint: LoadedCheckpointData | None = None,
) -> RiskManagementResult | None:
    """Main entry point for running risk management.

    Args:
        problem_definition: The problem definition used by the dispatcher.
        simulation_model_archive: The simulation model archive to transfer.
        model_hash: Hash of the simulation model, stored alongside checkpoints
            so a resumed run can verify it uses the same model.
        checkpoint: Previously saved optimizer state to resume from, or None
            to start fresh.

    Returns:
        A :class:`RiskManagementResult` with the best fitness value(s) and
        control vector(s), or ``None`` when running in evaluation mode or when
        the run was interrupted by the user (KeyboardInterrupt).
    """
    # "URGENT_RUN_ID" is a legacy variable name kept for compatibility with
    # existing deployment tooling; it is simply the unique run identifier.
    run_id = os.environ.get("URGENT_RUN_ID", "default")

    logger.info("Starting risk management process (run_id=%s)...", run_id)
    logger.debug(
        "Input: problem_definition=%s, archive_type=%s",
        problem_definition,
        type(simulation_model_archive),
    )

    runner_mode = _resolve_runner_mode()
    _configure_environment(problem_definition)

    run_plugins = bootstrap_run_plugins(problem_definition)

    worker_count = problem_definition.simulation_config.worker_count
    cm = _make_simulation_context_manager(runner_mode, worker_count)

    interrupted = False
    solution_updater: SolutionUpdaterService | None = None

    with cm:
        try:
            SimulationService.transfer_simulation_model(
                simulation_model_archive=simulation_model_archive
            )

            dispatcher = ProblemDispatcherService(
                problem_definition=problem_definition,
            )

            if problem_definition.run_mode == RunMode.Evaluation:
                _run_evaluation(dispatcher, run_plugins)
                return None

            solution_updater = SolutionUpdaterService(
                optimization_engine=run_plugins.optimizer_name,
                max_generations=dispatcher.max_generation,
                max_stall_generations=dispatcher.max_stall_generations,
                objectives=dispatcher.optimization_objectives,
                seed=problem_definition.optimization_parameters.seed,
            )

            interrupted = not _run_optimization_loop(
                dispatcher=dispatcher,
                solution_updater=solution_updater,
                run_plugins=run_plugins,
                problem_definition=problem_definition,
                model_hash=model_hash,
                run_id=run_id,
                checkpoint=checkpoint,
            )

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
            else:
                logger.exception("gRPC error in risk management process.")
            raise
        except Exception:
            logger.exception("Error in risk management process.")
            raise

    if interrupted or solution_updater is None:
        return None

    return _extract_best_result(solution_updater)


def _resolve_runner_mode() -> str:
    """Read and validate RUNNER_MODE, failing loudly on unknown values."""
    runner_mode = os.getenv("RUNNER_MODE", "thread").lower()
    if runner_mode not in _VALID_RUNNER_MODES:
        raise ValueError(
            f"Invalid RUNNER_MODE={runner_mode!r}; "
            f"expected one of {sorted(_VALID_RUNNER_MODES)}."
        )
    return runner_mode


def _make_simulation_context_manager(runner_mode: str, worker_count: int):
    """Select the simulation context manager for the validated runner mode."""
    if runner_mode == "thread":
        return simulation_process_context_manager(worker_count=worker_count)
    return simulation_cluster_context_manager(worker_count=worker_count)


def _configure_environment(problem_definition: ProblemDispatcherDefinition) -> None:
    """Export run configuration as environment variables.

    NOTE: These variables are read by the simulation workers/services
    (RUN_MODE, WORKER_SIMULATION_TIMEOUT_SECONDS, SERVER_JOB_TIMEOUT_SECONDS).
    This is process-global state: it is not restored afterwards and is not
    safe if multiple runs share one process. Prefer passing these values
    explicitly once the consuming services support it.
    """
    os.environ["RUN_MODE"] = problem_definition.run_mode.value
    os.environ["WORKER_SIMULATION_TIMEOUT_SECONDS"] = str(
        problem_definition.simulation_config.worker_simulation_timeout_seconds
    )
    os.environ["SERVER_JOB_TIMEOUT_SECONDS"] = str(
        problem_definition.simulation_config.server_job_timeout_seconds
    )


def _run_evaluation(
    dispatcher: ProblemDispatcherService,
    run_plugins: ResolvedRunPlugins,
) -> None:
    """Run a single validation simulation (RunMode.Evaluation)."""
    logger.info("Run in evaluation mode: Running a single validation simulation.")
    solutions = dispatcher.process_iteration(None)
    sim_cases = _prepare_simulation_cases(
        solutions,
        dispatcher.expected_optimization_function_names,
        run_plugins.domain_services,
    )
    logger.info("Submitting evaluation simulation case(s) to SimulationService.")
    completed_cases = SimulationService.process_request(
        {"simulation_cases": sim_cases, "connector": run_plugins.connector_name}
    )
    for index, case in enumerate(completed_cases.simulation_cases):
        logger.info(
            "Evaluation simulation case #%d completed successfully with results: %s",
            index + 1,
            case.results,
        )


def _run_optimization_loop(
    dispatcher: ProblemDispatcherService,
    solution_updater: SolutionUpdaterService,
    run_plugins: ResolvedRunPlugins,
    problem_definition: ProblemDispatcherDefinition,
    model_hash: str,
    run_id: str,
    checkpoint: LoadedCheckpointData | None,
) -> bool:
    """Run the generation loop until the loop controller stops.

    Returns:
        True if the loop completed normally, False if interrupted by the user
        (a best-effort checkpoint is saved before returning in that case).
    """
    full_key_boundaries = dispatcher.full_key_boundaries
    logger.debug("Boundaries retrieved: %s", full_key_boundaries)
    full_key_linear_inequalities = dispatcher.full_key_linear_inequalities
    logger.debug("Linear inequalities: %s", full_key_linear_inequalities)

    checkpoint_interval = problem_definition.simulation_config.checkpoint_interval
    checkpoint_dir = Path(problem_definition.simulation_config.checkpoint_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / checkpoint_filename(run_id)

    loop_controller = solution_updater.loop_controller

    next_solutions = None
    if checkpoint is not None:
        logger.info("Restoring optimizer state from checkpoint.")
        next_solutions = solution_updater.restore_checkpoint_state(checkpoint)
        logger.info("Resuming from generation %s.", loop_controller.current_generation)
        # The restored generation was already fully processed and checkpointed
        # after completion, so resume at the following generation. (Covered by
        # a resume round-trip test: save at N, restore, verify N+1 runs next.)
        loop_controller.increment_generation()

    try:
        while loop_controller.running():
            logger.info(
                "*** Starting generation %s ***", loop_controller.current_generation
            )
            solutions = dispatcher.process_iteration(next_solutions)
            logger.debug(
                "Generated %d solution candidates.",
                len(solutions.solution_candidates),
            )

            sim_cases = _prepare_simulation_cases(
                solutions,
                dispatcher.expected_optimization_function_names,
                run_plugins.domain_services,
            )

            logger.debug("Prepared %d simulation cases.", len(sim_cases))
            logger.info("Submitting simulation cases to SimulationService.")
            completed_cases = SimulationService.process_request(
                {
                    "simulation_cases": sim_cases,
                    "connector": run_plugins.connector_name,
                }
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
                "Generation %s successfully completed.",
                loop_controller.current_generation,
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

    except KeyboardInterrupt:
        logger.warning(
            "Risk management process interrupted by user; "
            "saving best-effort checkpoint before exiting."
        )
        _save_checkpoint(
            solution_updater,
            next_solutions,
            checkpoint_file,
            problem_definition,
            model_hash,
            run_id,
        )
        return False

    logger.info(
        "Loop controller stopped at generation %s. Info: %s",
        loop_controller.current_generation,
        loop_controller.info,
    )

    if checkpoint_file.exists():
        checkpoint_file.unlink()
        logger.info("Checkpoint cleared after successful completion.")

    return True


def _extract_best_result(
    solution_updater: SolutionUpdaterService,
) -> RiskManagementResult:
    """Build the final result object from the solution updater's best state."""
    best_result = solution_updater.global_best_result
    best_result_descriptive = solution_updater.global_best_result_descriptive
    best_control_vector = solution_updater.global_best_control_vector

    if isinstance(best_control_vector, list):
        control_vectors: list[dict[str, Any]] | dict[str, Any] = [
            parse_flat_dict_to_nested(cv.items) for cv in best_control_vector
        ]
        logger.info(
            "Optimization completed (Pareto front, %d solutions). Fitness values = %s",
            len(best_control_vector),
            best_result_descriptive,
        )
        logger.debug("Pareto control vectors: %s", control_vectors)
        return RiskManagementResult(
            values=best_result, control_vectors=control_vectors, is_pareto=True
        )

    control_vectors = parse_flat_dict_to_nested(best_control_vector.items)
    logger.info(
        "Optimization completed. Fitness value(s) = %s", best_result_descriptive
    )
    logger.debug("Best control vector: %s", control_vectors)
    return RiskManagementResult(
        values=best_result, control_vectors=control_vectors, is_pareto=False
    )


def _prepare_simulation_cases(
    solutions: ProblemDispatcherServiceResponse,
    expected_cost_function_names: list[str],
    domain_services: dict[str, DomainServiceInterface],
) -> list[SimulationCaseDict]:
    """Convert solution candidates into simulation-ready case dicts.

    Each configured domain service builds its own payload column from its
    task's item states; the columns are flattened into one ``payload`` dict
    keyed by service name. Result placeholders are initialised for every
    expected cost-function name.

    Args:
        solutions: Candidates produced by ``ProblemDispatcherService``.
        expected_cost_function_names: Objective keys; results seeded with NaN.
        domain_services: Resolved domain service instances keyed by name.

    Returns:
        List of simulation case dicts ready for ``SimulationService.process_request``.
    """
    logger.debug("Preparing %d simulation cases.", len(solutions.solution_candidates))
    sim_cases: list[SimulationCaseDict] = []

    for index, solution in enumerate(solutions.solution_candidates):
        logger.debug("Processing solution candidate #%d.", index + 1)
        payload: dict[str, Any] = {}
        control_vector: dict[str, Any] = {}

        for service_name, task in solution.tasks.items():
            payload[service_name] = domain_services[service_name].build_payload(
                task.request
            )
            control_vector.update(task.control_vector.items)
            logger.debug(
                "Built %d %r item(s) for candidate #%d.",
                len(task.request),
                service_name,
                index + 1,
            )

        sim_cases.append(
            {
                "payload": payload,
                "control_vector": control_vector,
                "results": {k: float("nan") for k in expected_cost_function_names},
            }
        )

    logger.debug("All %d simulation cases prepared.", len(sim_cases))
    return sim_cases


def _save_checkpoint(
    solution_updater: SolutionUpdaterService,
    next_solutions: Any,
    checkpoint_file: Path,
    problem_definition: ProblemDispatcherDefinition,
    model_hash: str,
    run_id: str,
) -> None:
    """Persist optimizer state; failures are logged (with traceback) but not fatal."""
    try:
        state = solution_updater.get_checkpoint_state(next_solutions)
        save_checkpoint(checkpoint_file, problem_definition, state, model_hash, run_id)
        logger.info(
            "Checkpoint for run ID %s saved at generation %s: %s",
            run_id,
            solution_updater.loop_controller.current_generation,
            checkpoint_file,
        )
    except Exception:
        logger.exception("Failed to save checkpoint.")
