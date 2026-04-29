import json
import os
from pathlib import Path
from typing import Any

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
    logger.info("Risk management toolbox started programmatically.")

    if disable_external_log_terminals:
        os.environ["URGENT_EXTERNAL_DOCKER_LOG_CONSOLE"] = "false"
        logger.info("External log terminals disabled; using file logging only.")

    if use_docker:
        logger.info("Using Docker for simulations.")
        os.environ["OPEN_DARTS_RUNNER"] = "docker"
    else:
        logger.info("Using multi-threading for simulations.")
        os.environ["OPEN_DARTS_RUNNER"] = "thread"

    model_hash = compute_file_hash(model_file)

    checkpoint: LoadedCheckpointData | None = None
    try:
        if resume_file is not None:
            logger.info("Checkpoint file provided; resuming optimization.")
            checkpoint_data = load_checkpoint(Path(resume_file))

            cp_hash = checkpoint_data.get("model_hash")
            if cp_hash and cp_hash != model_hash:
                raise ValueError(
                    "Model mismatch! Make sure the model file used for this checkpoint is the same as the one provided now."
                )

            problem_definition = checkpoint_data["config"]
            checkpoint = checkpoint_data
        else:
            with open(config_file) as file:  # type: ignore[arg-type]
                problem_definition = ProblemDispatcherDefinition.model_validate(
                    json.load(file)
                )
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")
        try:
            zip_path = zip_results()
            logger.info("Created results archive: %s", zip_path)
        except Exception as ze:
            logger.error("Failed to create results archive: %s", ze)
        raise

    # Invoke run_risk_management with the required arguments
    try:
        result = run_risk_management(
            problem_definition=problem_definition,
            simulation_model_archive=model_file,
            model_hash=model_hash,
            checkpoint=checkpoint,
        )
        logger.info("Risk management process completed successfully.")
        return result
    except grpc.RpcError as e:
        code = None
        details = None
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
        elif code == grpc.StatusCode.UNAVAILABLE and details == "Server shutting down":
            logger.info(
                "Risk management stopped because simulation server is shutting down."
            )
            return None

        logger.error("An error occurred: %s", e)
        raise
    except Exception as e:
        logger.error("An error occurred: %s", e)
        raise
    finally:
        try:
            zip_path = zip_results()
            logger.info("Created results archive: %s", zip_path)
        except Exception as ze:
            logger.error("Failed to create results archive: %s", ze)
            raise
