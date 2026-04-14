from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

from common.models import RunMode


class SimulationServiceConfig(BaseSettings):
    server_host: str = Field(
        default="localhost",
        validation_alias="SERVER_HOST",
        description="gRPC server hostname",
    )
    server_port: int = Field(
        default=50051, validation_alias="SERVER_PORT", description="gRPC server port"
    )

    grpc_max_message_size: int = Field(
        default=100 * 1024 * 1024,
        description="Maximum gRPC message size in bytes (default: 100MB)",
    )
    server_startup_timeout: float = Field(
        default=25.0, description="Timeout in seconds for server startup"
    )

    job_timeout_seconds: int = Field(
        default=3600,
        validation_alias="JOB_TIMEOUT_SECONDS",
        description="Timeout in seconds for simulation jobs (default: 1 hour)",
    )

    simulation_timeout_seconds: int = Field(
        default=900,
        validation_alias="SIMULATION_TIMEOUT",
        description="Timeout in seconds for simulation to respond (default: 15 minutes)",
    )

    log_interval_seconds: int = Field(
        default=30,
        description="Interval in seconds for logging progress (default: 30 seconds)",
    )
    long_poll_timeout_seconds: int = Field(
        default=30,
        description="Timeout in seconds for long-polling for simulation jobs (default: 30 seconds)",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    run_mode: RunMode = Field(default=RunMode.Optimization)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def grpc_target(self) -> str:
        return f"{self.server_host}:{self.server_port}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def channel_options(self) -> list[tuple[str, int]]:
        return [
            ("grpc.max_send_message_length", self.grpc_max_message_size),
            ("grpc.max_receive_message_length", self.grpc_max_message_size),
        ]


def get_simulation_config() -> SimulationServiceConfig:
    """
    Factory function to get the singleton configuration instance.
    """
    return SimulationServiceConfig()
