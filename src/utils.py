import json
from pathlib import Path

from services.problem_dispatcher_service import ProblemDispatcherDefinition


def strip_descriptions(schema: dict) -> dict:
    if isinstance(schema, dict):
        return {
            k: strip_descriptions(v) for k, v in schema.items() if k != "description"
        }
    if isinstance(schema, list):
        return [strip_descriptions(item) for item in schema]
    return schema


def write_schema_rmt_schema(schema_folder_path: str | Path, version: str) -> None:
    schema = ProblemDispatcherDefinition.model_json_schema(mode="serialization")
    clean_schema = strip_descriptions(schema)

    schema_folder_path = Path(schema_folder_path)
    schema_folder_path.mkdir(parents=True, exist_ok=True)

    out_path = schema_folder_path / f"{version}.json"
    out_path.write_text(json.dumps(clean_schema, indent=2), encoding="utf-8")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent  # src/ -> project root
    schemas_folder = project_root / "schemas"

    version = "0.2.0"
    write_schema_rmt_schema(schemas_folder, version)
