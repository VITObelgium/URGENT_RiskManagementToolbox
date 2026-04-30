import argparse

from interfaces import risk_management


def cli():
    """
    Main function to parse command-line arguments and invoke the `run_risk_management` function.
    """
    parser = argparse.ArgumentParser(
        description="CLI to run risk management simulations."
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--config-file",
        type=str,
        help="Path to the configuration file (JSON format) for risk management.",
    )
    source_group.add_argument(
        "--resume",
        type=str,
        metavar="CHECKPOINT_PATH",
        help="Path to a checkpoint (.npz) file to resume an interrupted optimization run.",
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
    parser.add_argument(
        "--no-external-log-terminals",
        action="store_true",
        help="Disable opening extra terminal windows for Docker/server log tails and keep file logging only.",
    )

    args = parser.parse_args()

    try:
        risk_management(
            config_file=args.config_file,
            model_file=args.model_file,
            resume_file=args.resume,
            use_docker=args.use_docker,
            disable_external_log_terminals=args.no_external_log_terminals,
        )
    except Exception:
        exit(1)
