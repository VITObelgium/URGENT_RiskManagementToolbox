import argparse
import json
import os

from logger import configure_logger, get_logger
from orchestration.risk_management_service import run_risk_management
from services.problem_dispatcher_service.core.models import ProblemDispatcherDefinition


def cli():
    """
    Main function to parse command-line arguments and invoke the `run_risk_management` function.
    """
    parser = argparse.ArgumentParser(
        description="CLI to run risk management simulations."
    )

    # Adding arguments for CLI
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to the configuration file (JSON format) for risk management.",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        required=True,
        help="Path to the simulation model archive file.",
    )
    parser.add_argument(
        "--use-docker",
        action="store_true",
        help="Flag to indicate whether to use Docker for simulations, or use multi-threading-based local execution. Default is False (i.e., use multi-threading).",
    )

    args = parser.parse_args()

    configure_logger()
    logger = get_logger(__name__)
    logger.info("Risk management toolbox started from CLI.")

    if args.use_docker:
        logger.info("Using Docker for simulations.")
        os.environ["OPEN_DARTS_RUNNER"] = "docker"
    else:
        logger.info("Using multi-threading for simulations.")
        os.environ["OPEN_DARTS_RUNNER"] = "thread"

    # Load the problem_definition from the JSON file
    try:
        with open(args.config_file, "r") as file:
            problem_definition = ProblemDispatcherDefinition.model_validate(
                json.load(file)
            )
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")
        exit(1)

    # Invoke run_risk_management with the required arguments
    try:
        run_risk_management(
            problem_definition=problem_definition,
            simulation_model_archive=args.model_file,
        )
        logger.info("Risk management process completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        exit(1)
