import subprocess
from contextlib import contextmanager

from logger import get_logger
from services.simulation_service.core.service.utils import core_directory

logger = get_logger(__name__)


class WebUiManager:
    @staticmethod
    def build_web_ui() -> None:
        logger.info("Building web application...")
        with core_directory():
            try:
                result = subprocess.run(
                    ["docker", "compose", "--profile", "webui", "build", "webui"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logger.debug("Docker build output:\n%s", result.stdout)
                logger.info("Web application successfully built.")
            except subprocess.CalledProcessError as e:
                logger.error("Error building web application:\n%s", e.stderr)
                raise

    @staticmethod
    def start_web_ui() -> None:
        logger.info("Web application will be started for visualization.")
        try:
            with core_directory():
                result = subprocess.run(
                    ["docker", "compose", "--profile", "webui", "up", "-d", "webui"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logger.debug("Docker compose output:\n%s", result.stdout)
                logger.info("Web application successfully started.")
        except subprocess.CalledProcessError as e:
            logger.error(
                "Failed to start web application. stderr:\n%s\nstdout:\n%s",
                e.stderr,
                e.stdout,
            )
            raise
        except Exception as e:
            logger.error("Failed to start web application: %s", e)
            raise

    @staticmethod
    def shutdown_web_ui() -> None:
        logger.info("Shutting down web application...")
        with core_directory():
            try:
                subprocess.run(
                    ["docker", "compose", "--profile", "webui", "down"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logger.info("Web application successfully shut down.")
            except subprocess.CalledProcessError as e:
                logger.error("Error shutting down web application:\n%s", e.stderr)
                raise


@contextmanager
def web_app_context_manager():
    """
    Context manager for managing the web application lifecycle.
    """
    logger.info("Building web application...")
    WebUiManager.build_web_ui()
    WebUiManager.start_web_ui()
    try:
        yield
    finally:
        logger.info("Stopping web application...")
        WebUiManager.shutdown_web_ui()
