import json
import tempfile
from pathlib import Path

from utils import strip_descriptions, write_schema_rmt_schema


class TestStripDescriptions:
    def test_removes_description_key_from_flat_dict(self):
        schema = {"type": "object", "description": "A test object", "title": "Test"}
        result = strip_descriptions(schema)
        assert "description" not in result
        assert result["type"] == "object"
        assert result["title"] == "Test"

    def test_removes_description_recursively_from_nested_dict(self):
        schema = {
            "type": "object",
            "description": "outer",
            "properties": {
                "name": {"type": "string", "description": "inner"},
            },
        }
        result = strip_descriptions(schema)
        assert "description" not in result
        assert "description" not in result["properties"]["name"]
        assert result["properties"]["name"]["type"] == "string"

    def test_handles_list_of_dicts(self):
        schema = [
            {"type": "string", "description": "first"},
            {"type": "integer", "description": "second"},
        ]
        result = strip_descriptions(schema)
        assert isinstance(result, list)
        assert len(result) == 2
        assert "description" not in result[0]
        assert "description" not in result[1]
        assert result[0]["type"] == "string"

    def test_returns_scalars_unchanged(self):
        assert strip_descriptions("hello") == "hello"
        assert strip_descriptions(42) == 42
        assert strip_descriptions(3.14) == 3.14
        assert strip_descriptions(None) is None
        assert strip_descriptions(True) is True

    def test_empty_dict(self):
        assert strip_descriptions({}) == {}

    def test_empty_list(self):
        assert strip_descriptions([]) == []

    def test_deeply_nested(self):
        schema = {
            "description": "root",
            "items": {
                "description": "item",
                "properties": {
                    "field": {
                        "description": "field-level",
                        "type": "number",
                    }
                },
            },
        }
        result = strip_descriptions(schema)
        assert "description" not in result
        assert "description" not in result["items"]
        assert "description" not in result["items"]["properties"]["field"]
        assert result["items"]["properties"]["field"]["type"] == "number"

    def test_list_containing_nested_structures(self):
        schema = {
            "anyOf": [
                {"type": "string", "description": "str option"},
                [{"type": "integer", "description": "int in list"}],
            ]
        }
        result = strip_descriptions(schema)
        assert "description" not in result["anyOf"][0]
        assert "description" not in result["anyOf"][1][0]


class TestWriteSchemaRmtSchema:
    def test_creates_json_file_in_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            write_schema_rmt_schema(tmpdir, "0.1.0")
            out_path = Path(tmpdir) / "0.1.0.json"
            assert out_path.exists()

    def test_written_file_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            write_schema_rmt_schema(tmpdir, "0.1.0")
            out_path = Path(tmpdir) / "0.1.0.json"
            content = json.loads(out_path.read_text())
            assert isinstance(content, dict)

    def test_written_file_has_no_description_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            write_schema_rmt_schema(tmpdir, "0.2.0")
            out_path = Path(tmpdir) / "0.2.0.json"
            content = json.loads(out_path.read_text())

            def has_description(obj):
                if isinstance(obj, dict):
                    if "description" in obj:
                        return True
                    return any(has_description(v) for v in obj.values())
                if isinstance(obj, list):
                    return any(has_description(i) for i in obj)
                return False

            assert not has_description(content)

    def test_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "a" / "b" / "c"
            write_schema_rmt_schema(nested, "1.0.0")
            assert (nested / "1.0.0.json").exists()

    def test_accepts_path_object(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            write_schema_rmt_schema(Path(tmpdir), "0.3.0")
            assert (Path(tmpdir) / "0.3.0.json").exists()

    def test_version_used_as_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            write_schema_rmt_schema(tmpdir, "custom-version")
            assert (Path(tmpdir) / "custom-version.json").exists()
