import json
from typing import Any, TypeAlias

JSON: TypeAlias = dict[str, Any]


def json_to_str(
    json_message: JSON,
) -> str:
    return json.dumps(json_message)


def str_to_json(str_message: str) -> JSON:
    return json.loads(str_message)  # type: ignore[no-any-return]
