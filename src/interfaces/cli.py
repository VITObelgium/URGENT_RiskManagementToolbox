import argparse
import json

from logger import configure_logger, get_logger
from orchestration.risk_management_service import run_risk_management
from services.simulation_service import (
    web_app_context_manager,
)


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
        required=False,
        help="Path to the configuration file (JSON format) for risk management.",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        required=False,
        help="Path to the simulation model archive file.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=10,
        help="Population size. Default is 10",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Iterations limit without better result. Default is 10.",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=10,
        help="Maximum number of generations. Default is 10.",
    )
    parser.add_argument(
        "--run-with-web-app",
        action="store_true",
        default=False,
        help="Run the web application for visualization and GUI. Default is False.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Apply the logging level
    configure_logger()
    logger = get_logger(__name__)
    logger.info("Risk management toolbox started from CLI.")

    if not args.config_file or not args.model_file:
        logger.info(
            "Running in standby mode, because config_file or model_file was not provided."
        )
        logger.info("Enabling web app to allow user to upload files.")
        args.run_with_web_app = True

    if args.run_with_web_app:
        logger.info("Web app mode is enabled. Starting web application...")
        try:
            with web_app_context_manager():
                logger.info(
                    "Web application is running. Access it via http://localhost:50001"
                )
                logger.info("Press Ctrl+C to stop the web application.")
                while True:
                    pass
        except Exception as e:
            logger.error(f"Failed to start web application: {e}")

    if args.config_file and args.model_file:
        # Load the problem_definition from the JSON file
        try:
            with open(args.config_file, "r") as file:
                problem_definition = json.load(file)
        except Exception as e:
            logger.error(f"Failed to load configuration file: {e}")
            exit(1)

        # Invoke run_risk_management with the required arguments
        try:
            run_risk_management(
                problem_definition=problem_definition,
                simulation_model_archive=args.model_file,
                n_size=args.population_size,
                patience=args.patience,
                max_generations=args.max_generations,
            )
            logger.info("Risk management process completed successfully.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            exit(1)
