"""Tests for urgent_plugins.utils — list_plugins and _resolve_plugin."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from urgent_plugins.api import PluginKind
import urgent_plugins.utils as plugin_utils


class TestListPlugins:
    def test_lists_py_files_in_kind_directory(self, tmp_path: Path, capsys):
        connectors_dir = tmp_path / "connectors"
        connectors_dir.mkdir(parents=True)
        (connectors_dir / "my_connector.py").write_text("# stub")
        (connectors_dir / "__init__.py").write_text("")

        with patch("urgent_plugins.utils.default_plugins_root", return_value=tmp_path):
            plugin_utils.list_plugins()

        out = capsys.readouterr().out
        assert "my_connector" in out

    def test_shows_none_when_kind_directory_missing(self, tmp_path: Path, capsys):
        # Don't create any subdirectory — kind_dir.is_dir() will be False.
        with patch("urgent_plugins.utils.default_plugins_root", return_value=tmp_path):
            plugin_utils.list_plugins()

        out = capsys.readouterr().out
        assert "(none)" in out

    def test_prints_plugin_root(self, tmp_path: Path, capsys):
        with patch("urgent_plugins.utils.default_plugins_root", return_value=tmp_path):
            plugin_utils.list_plugins()

        out = capsys.readouterr().out
        assert str(tmp_path) in out


class TestResolvePlugin:
    def test_raises_if_name_empty(self):
        with pytest.raises(ValueError, match="must be specified"):
            plugin_utils._resolve_plugin(PluginKind.CONNECTOR, "")

    def test_raises_if_name_whitespace(self):
        with pytest.raises(ValueError, match="must be specified"):
            plugin_utils._resolve_plugin(PluginKind.CONNECTOR, "   ")

    def test_returns_existing_registered_plugin(self, tmp_path: Path):
        mock_registry = MagicMock()
        mock_descriptor = MagicMock()
        mock_descriptor.name = "my_connector"
        mock_registry.get.return_value = mock_descriptor

        with (
            patch.dict(os.environ, {}, clear=False),
            patch("urgent_plugins.utils.get_registry", return_value=mock_registry),
            patch("urgent_plugins.utils.default_plugins_root", return_value=tmp_path),
        ):
            name = plugin_utils._resolve_plugin(PluginKind.CONNECTOR, "my_connector")

        assert name == "my_connector"
        mock_registry.get.assert_called_once_with(PluginKind.CONNECTOR, "my_connector")

    def test_loads_and_returns_new_plugin(self, tmp_path: Path):
        mock_registry = MagicMock()
        mock_registry.get.return_value = None  # not yet loaded
        mock_descriptor = MagicMock()
        mock_descriptor.name = "new_connector"

        with (
            patch.dict(os.environ, {}, clear=False),
            patch("urgent_plugins.utils.get_registry", return_value=mock_registry),
            patch("urgent_plugins.utils.default_plugins_root", return_value=tmp_path),
            patch(
                "urgent_plugins.utils.load_local_plugin", return_value=mock_descriptor
            ),
        ):
            name = plugin_utils._resolve_plugin(PluginKind.CONNECTOR, "new_connector")

        assert name == "new_connector"

    def test_sets_env_vars(self, tmp_path: Path):
        mock_registry = MagicMock()
        mock_descriptor = MagicMock()
        mock_descriptor.name = "env_connector"
        mock_registry.get.return_value = mock_descriptor

        with (
            patch.dict(os.environ, {}, clear=False),
            patch("urgent_plugins.utils.get_registry", return_value=mock_registry),
            patch("urgent_plugins.utils.default_plugins_root", return_value=tmp_path),
        ):
            plugin_utils._resolve_plugin(PluginKind.CONNECTOR, "env_connector")
            assert os.environ.get("URGENT_PLUGIN_PATH") == str(tmp_path)
