# numeric_logger.py
import logging
import os
from typing import List, Optional


def get_csv_logger(
    filename: str,
    logger_name: str = "csv",
    level: int = logging.INFO,
    columns: List[str] | None = None,
) -> logging.Logger:
    """
    Get a logger for CSV data output.

    Args:
        filename (str): Name of the CSV file (without path)
        logger_name (str, optional): Name of the logger. Defaults to "csv"
        level (int, optional): Logging level. Defaults to logging.INFO
        columns (Optional[List[str]], optional): List of column names. Defaults to None

    Returns:
        logging.Logger: Configured logger instance

    Example:
        ```python
        logger = get_csv_logger("data.csv", columns=["value1", "value2"])
        logger.info("1.23,4.56")  # Log data directly
        ```
    """
    # Get the log directory path (same as in u_logger.py)
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../log")
    output_path = os.path.join(log_dir, filename)

    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create file handler
    file_handler = logging.FileHandler(output_path, mode="a")
    file_handler.setLevel(level)

    # Create formatter (simple CSV format)
    formatter = logging.Formatter(
        "%(asctime)s,%(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)

    # Write header if columns are provided and file is empty
    if columns and os.path.getsize(output_path) == 0:
        header_str = ",".join(
            columns
        )  # Remove timestamp from header as it's added by formatter
        logger.info(header_str)

    return logger
