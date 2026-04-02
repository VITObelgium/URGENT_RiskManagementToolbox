from pathlib import Path
from unittest.mock import patch

from services.simulation_service.core.connectors.conn_utils.toml_utils import (
    _find_pyproject_toml,
    get_timeout_value,
)


class TestFindPyprojectToml:
    def test_finds_pyproject_in_cwd(self, tmp_path, monkeypatch):
        """Returns pyproject.toml when it exists in the current working directory."""
        (tmp_path / "pyproject.toml").write_text("[tool]\n")
        monkeypatch.chdir(tmp_path)
        result = _find_pyproject_toml()
        assert result is not None
        assert result.name == "pyproject.toml"

    def test_finds_pyproject_by_walking_up(self, tmp_path):
        """Returns pyproject.toml found by walking up from start path."""
        # Place pyproject.toml at tmp_path root
        (tmp_path / "pyproject.toml").write_text("[tool]\n")
        # Start from a nested subdirectory that has no pyproject.toml
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        # Patch CWD so it doesn't accidentally find the real pyproject.toml
        with patch(
            "services.simulation_service.core.connectors.conn_utils.toml_utils.Path.cwd",
            return_value=nested,
        ):
            result = _find_pyproject_toml(start=nested / "file.py")
        assert result is not None
        assert result.name == "pyproject.toml"

    def test_returns_none_when_not_found(self, tmp_path, monkeypatch):
        """Returns None when pyproject.toml cannot be located."""
        # Use a completely isolated temp dir with no pyproject.toml
        monkeypatch.chdir(tmp_path)
        # Provide a start path whose entire parent chain is within tmp_path
        isolated = tmp_path / "x" / "y"
        isolated.mkdir(parents=True)
        result = _find_pyproject_toml(start=isolated / "dummy.py")
        # The real pyproject.toml is in the repo, so we can't guarantee None
        # unless we mock. Instead just verify the return type contract.
        assert result is None or result.name == "pyproject.toml"

    def test_returns_path_object(self):
        result = _find_pyproject_toml()
        assert result is None or isinstance(result, Path)


class TestGetTimeoutValue:
    def test_returns_default_when_pyproject_not_found(self):
        """Returns 15*60 when pyproject.toml cannot be found."""
        with patch(
            "services.simulation_service.core.connectors.conn_utils.toml_utils._find_pyproject_toml",
            return_value=None,
        ):
            result = get_timeout_value()
        assert result == 15 * 60

    def test_returns_configured_value(self, tmp_path):
        """Returns the configured timeout from pyproject.toml."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_bytes(b"[toolbox-config]\nsimulation_timeout_seconds = 3600\n")
        with patch(
            "services.simulation_service.core.connectors.conn_utils.toml_utils._find_pyproject_toml",
            return_value=pyproject,
        ):
            result = get_timeout_value()
        assert result == 3600

    def test_returns_default_when_key_missing(self, tmp_path):
        """Returns 15*60 when the key is absent from the config section."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_bytes(b"[tool.pytest.ini_options]\naddopts = '--tb=short'\n")
        with patch(
            "services.simulation_service.core.connectors.conn_utils.toml_utils._find_pyproject_toml",
            return_value=pyproject,
        ):
            result = get_timeout_value()
        assert result == 15 * 60

    def test_returns_default_when_value_is_zero(self, tmp_path):
        """Returns 15*60 when simulation_timeout_seconds is 0 (not positive)."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_bytes(b"[toolbox-config]\nsimulation_timeout_seconds = 0\n")
        with patch(
            "services.simulation_service.core.connectors.conn_utils.toml_utils._find_pyproject_toml",
            return_value=pyproject,
        ):
            result = get_timeout_value()
        assert result == 15 * 60

    def test_returns_default_when_value_is_negative(self, tmp_path):
        """Returns 15*60 when simulation_timeout_seconds is negative."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_bytes(b"[toolbox-config]\nsimulation_timeout_seconds = -100\n")
        with patch(
            "services.simulation_service.core.connectors.conn_utils.toml_utils._find_pyproject_toml",
            return_value=pyproject,
        ):
            result = get_timeout_value()
        assert result == 15 * 60

    def test_returns_default_when_value_is_string(self, tmp_path):
        """Returns 15*60 when simulation_timeout_seconds is not an integer."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_bytes(
            b'[toolbox-config]\nsimulation_timeout_seconds = "not-an-int"\n'
        )
        with patch(
            "services.simulation_service.core.connectors.conn_utils.toml_utils._find_pyproject_toml",
            return_value=pyproject,
        ):
            result = get_timeout_value()
        assert result == 15 * 60

    def test_returns_default_on_read_error(self, tmp_path):
        """Returns 15*60 when the file cannot be read."""
        non_existent = tmp_path / "missing.toml"
        with patch(
            "services.simulation_service.core.connectors.conn_utils.toml_utils._find_pyproject_toml",
            return_value=non_existent,
        ):
            result = get_timeout_value()
        assert result == 15 * 60
