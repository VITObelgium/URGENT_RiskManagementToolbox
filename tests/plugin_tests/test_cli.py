"""Tests for urgent_plugins.cli — the create-plugin scaffolder entry point."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from urgent_plugins.cli import main


@pytest.fixture(autouse=True)
def _patch_get_kind_specs():
    """Always expose at least one alias so argparse choices are non-empty."""
    mock_spec = MagicMock()
    mock_spec.cli_alias = "connector"
    with patch("urgent_plugins.cli.get_kind_specs", return_value=[mock_spec]):
        yield mock_spec


def _make_spec(alias: str = "connector") -> MagicMock:
    spec = MagicMock()
    spec.cli_alias = alias
    spec.kind.value = alias
    return spec


class TestMainSuccess:
    def test_returns_zero_on_success(self, tmp_path: Path):
        spec = _make_spec()
        with (
            patch("urgent_plugins.cli.find_kind_spec_by_alias", return_value=spec),
            patch("urgent_plugins.cli.write_plugin", return_value=tmp_path / "out.py"),
        ):
            result = main(["connector", "MyConnector"])

        assert result == 0

    def test_prints_created_path(self, tmp_path: Path, capsys):
        spec = _make_spec()
        out_path = tmp_path / "plugins" / "connectors" / "my_connector.py"
        with (
            patch("urgent_plugins.cli.find_kind_spec_by_alias", return_value=spec),
            patch("urgent_plugins.cli.write_plugin", return_value=out_path),
        ):
            main(["connector", "MyConnector"])

        captured = capsys.readouterr()
        assert str(out_path) in captured.out

    def test_plugin_name_flag_forwarded(self, tmp_path: Path):
        spec = _make_spec()
        with (
            patch("urgent_plugins.cli.find_kind_spec_by_alias", return_value=spec),
            patch(
                "urgent_plugins.cli.write_plugin", return_value=tmp_path / "out.py"
            ) as mock_write,
        ):
            main(["connector", "MyConnector", "--plugin-name", "custom_name"])

        mock_write.assert_called_once()
        _, kwargs = mock_write.call_args
        assert kwargs.get("plugin_name") == "custom_name"

    def test_overwrite_flag_forwarded(self, tmp_path: Path):
        spec = _make_spec()
        with (
            patch("urgent_plugins.cli.find_kind_spec_by_alias", return_value=spec),
            patch(
                "urgent_plugins.cli.write_plugin", return_value=tmp_path / "out.py"
            ) as mock_write,
        ):
            main(["connector", "MyConnector", "--overwrite"])

        mock_write.assert_called_once()
        _, kwargs = mock_write.call_args
        assert kwargs.get("overwrite") is True


class TestMainFileExistsError:
    def test_returns_one_on_file_exists(self):
        spec = _make_spec()
        with (
            patch("urgent_plugins.cli.find_kind_spec_by_alias", return_value=spec),
            patch(
                "urgent_plugins.cli.write_plugin",
                side_effect=FileExistsError("already exists"),
            ),
        ):
            result = main(["connector", "MyConnector"])

        assert result == 1

    def test_prints_error_message_to_stderr(self, capsys):
        spec = _make_spec()
        with (
            patch("urgent_plugins.cli.find_kind_spec_by_alias", return_value=spec),
            patch(
                "urgent_plugins.cli.write_plugin",
                side_effect=FileExistsError("file at path"),
            ),
        ):
            main(["connector", "MyConnector"])

        captured = capsys.readouterr()
        assert "error:" in captured.err
