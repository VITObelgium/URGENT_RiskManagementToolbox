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
)
from services.problem_dispatcher_service import ProblemDispatcherDefinition


def risk_management(
    config_file: str,
    model_file: str,
    use_docker: bool = False,
    disable_external_log_terminals: bool = False,
) -> tuple[float | npt.NDArray[np.float64], dict[str, Any]] | None:
    """
    Run risk management with specified parameters without using argparse.

    Args:
        config_file (str): Path to the configuration file (JSON format) for risk management.
        model_file (str): Path to the simulation model archive file.
        use_docker (bool): Flag to indicate whether to use Docker for simulations. Default is False.
        disable_external_log_terminals (bool): Disable opening extra terminal windows for log tails.
    """
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

    checkpoint: dict[str, Any] | None = None
    try:
        if config_file.endswith(".npz"):
            logger.info("Checkpoint file detected; resuming optimization.")
            checkpoint_data = load_checkpoint(Path(config_file))
            problem_definition = checkpoint_data["config"]
            checkpoint = checkpoint_data
        else:
            with open(config_file) as file:
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
