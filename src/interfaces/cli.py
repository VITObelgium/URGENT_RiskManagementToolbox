import argparse

from interfaces.common import risk_management


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

    # Use the new function with parsed arguments
    try:
        _ = risk_management(
            config_file=args.config_file,
            model_file=args.model_file,
            use_docker=args.use_docker,
        )
    except Exception:
        exit(1)
