import json
import os
from typing import Any

from logger import configure_logger, get_logger
from logger.utils import zip_results
from orchestration.risk_management_service import run_risk_management
from services.problem_dispatcher_service.core.models import ProblemDispatcherDefinition


def risk_management(
    config_file: str, model_file: str, use_docker: bool = False
) -> tuple[float, dict[str, Any]] | None:
    """
    Run risk management with specified parameters without using argparse.

    Args:
        config_file (str): Path to the configuration file (JSON format) for risk management.
        model_file (str): Path to the simulation model archive file.
        use_docker (bool): Flag to indicate whether to use Docker for simulations. Default is False.
    """
    configure_logger()
    logger = get_logger(__name__)
    logger.info("Risk management toolbox started programmatically.")

    if use_docker:
        logger.info("Using Docker for simulations.")
        os.environ["OPEN_DARTS_RUNNER"] = "docker"
    else:
        logger.info("Using multi-threading for simulations.")
        os.environ["OPEN_DARTS_RUNNER"] = "thread"

    # Load the problem_definition from the JSON file
    try:
        with open(config_file, "r") as file:
            problem_definition = ProblemDispatcherDefinition.model_validate(
                json.load(file)
            )
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")
        # Always attempt to zip (even on failure)
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
        )
        logger.info("Risk management process completed successfully.")
        return result
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
    finally:
        try:
            zip_path = zip_results()
            logger.info("Created results archive: %s", zip_path)
        except Exception as ze:
            logger.error("Failed to create results archive: %s", ze)
