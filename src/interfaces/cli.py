import argparse
import json

from logger import configure_logger, get_logger
from orchestration.risk_management_service import run_risk_management


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

    # Parse the arguments
    args = parser.parse_args()

    # Apply the logging level
    configure_logger()
    logger = get_logger(__name__)
    logger.info("Risk management toolbox started from CLI.")

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
