from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PluginSettings(BaseSettings):
    plugin_path: Path | None = Field(
        default=None, validation_alias="URGENT_PLUGIN_PATH"
    )
    connector_plugin_name: str = Field(
        default="", validation_alias="URGENT_CONNECTOR_PLUGIN_NAME"
    )
    optimizer_plugin_name: str = Field(
        default="", validation_alias="URGENT_OPTIMIZER_PLUGIN_NAME"
    )
    domain_service_plugin_names: str = Field(
        default="", validation_alias="URGENT_DOMAIN_SERVICE_PLUGIN_NAMES"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )


def get_plugin_settings() -> PluginSettings:
    return PluginSettings()
