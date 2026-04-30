import json
import os
import uuid
from enum import StrEnum
from pathlib import Path
from typing import Any, Never

if "URGENT_RUN_ID" not in os.environ:
    os.environ["URGENT_RUN_ID"] = uuid.uuid4().hex[:8]

import grpc
import numpy as np
import numpy.typing as npt

from logger import configure_logger, get_logger
from logger.utils import zip_results
from orchestration.risk_management_service import run_risk_management
from orchestration.risk_management_service.core.service.checkpoint import (
    load_checkpoint,
    compute_file_hash,
    LoadedCheckpointData,
)
from services.problem_dispatcher_service import ProblemDispatcherDefinition


class RunStatus(StrEnum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


def _try_zip_results(logger, run_status: RunStatus) -> None:
    try:
        zip_path = zip_results(None, run_status)
        logger.info("Created results archive: %s", zip_path)
    except Exception as ze:
        logger.error("Failed to create results archive: %s", ze)


def _handle_grpc_error(e: grpc.RpcError, logger) -> None | Never:
    """Returns None if the error is a known graceful shutdown, otherwise re-raises."""
    code, details = None, None
    try:
        code = e.code()
        details = e.details()
    except Exception:
        pass

    if code == grpc.StatusCode.ABORTED:
        logger.critical(
            "Risk management stopped because simulation server aborted (details=%s).",
            details,
        )
        return None
    if code == grpc.StatusCode.UNAVAILABLE and details == "Server shutting down":
        logger.info(
            "Risk management stopped because simulation server is shutting down."
        )
        return None

    logger.error("An error occurred: %s", e)
    raise e


def _load_problem_definition(
    config_file: str | None,
    resume_file: str | None,
    model_hash: str,
    logger,
) -> tuple[ProblemDispatcherDefinition, LoadedCheckpointData | None]:
    if resume_file is not None:
        logger.info("Checkpoint file provided; resuming optimization.")
        checkpoint_data = load_checkpoint(Path(resume_file))

        cp_hash = checkpoint_data.get("model_hash")
        if cp_hash and cp_hash != model_hash:
            raise ValueError(
                "Model mismatch! The model file must match the one used to create this checkpoint."
            )

        logger.info(
            "Run %s resuming checkpoint with run ID: %s",
            os.environ["URGENT_RUN_ID"],
            checkpoint_data["run_id"],
        )
        return checkpoint_data["config"], checkpoint_data

    with open(config_file) as f:  # type: ignore[arg-type]
        return ProblemDispatcherDefinition.model_validate(json.load(f)), None


def risk_management(
    model_file: str,
    config_file: str | None = None,
    resume_file: str | None = None,
    use_docker: bool = False,
    disable_external_log_terminals: bool = False,
) -> tuple[float | npt.NDArray[np.float64], dict[str, Any]] | None:
    """
    Run risk management with specified parameters without using argparse.

    Args:
        model_file (str): Path to the simulation model archive file.
        config_file (str | None): Path to the configuration file (JSON format) for risk management.
        resume_file (str | None): Path to a checkpoint (.npz) file to resume an interrupted run.
        use_docker (bool): Flag to indicate whether to use Docker for simulations. Default is False.
        disable_external_log_terminals (bool): Disable opening extra terminal windows for log tails.
    """
    if config_file is None and resume_file is None:
        raise ValueError("Either config_file or resume_file must be provided.")

    configure_logger()
    logger = get_logger(__name__)
    logger.info("Risk management toolbox started...")

    if disable_external_log_terminals:
        os.environ["URGENT_EXTERNAL_DOCKER_LOG_CONSOLE"] = "false"
        logger.info("External log terminals disabled; using file logging only.")

    os.environ["OPEN_DARTS_RUNNER"] = "docker" if use_docker else "thread"
    logger.info(
        "Using %s for simulations.", "Docker" if use_docker else "multi-threading"
    )

    try:
        model_hash = compute_file_hash(model_file)
    except Exception as e:
        logger.error("Failed to compute model hash: %s", e)
        raise

    try:
        problem_definition, checkpoint = _load_problem_definition(
            config_file, resume_file, model_hash, logger
        )
    except Exception as e:
        logger.error("Failed to load configuration: %s", e)
        _try_zip_results(logger, RunStatus.FAILED)
        raise

    try:
        result = run_risk_management(
            problem_definition=problem_definition,
            simulation_model_archive=model_file,
            model_hash=model_hash,
            checkpoint=checkpoint,
        )
        logger.info("Risk management process completed successfully.")
        run_status = RunStatus.COMPLETED
        return result
    except grpc.RpcError as e:
        run_status = RunStatus.FAILED
        return _handle_grpc_error(e, logger)
    except Exception as e:
        run_status = RunStatus.FAILED
        logger.error("An error occurred: %s", e)
        raise
    finally:
        _try_zip_results(logger, run_status)
