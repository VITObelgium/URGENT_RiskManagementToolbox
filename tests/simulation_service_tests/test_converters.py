import pytest

from services.simulation_service.core.utils.converters import json_to_str, str_to_json


@pytest.mark.parametrize(
    "input_data, function, expected_output, expected_exception",
    [
        # Valid JSON to string conversion
        ({"key": "value"}, json_to_str, '{"key": "value"}', None),
        ({}, json_to_str, "{}", None),
        (
            {"key": 1234, "nested": {"subkey": "value"}},
            json_to_str,
            '{"key": 1234, "nested": {"subkey": "value"}}',
            None,
        ),
        # Valid string to JSON conversion
        ('{"key": "value"}', str_to_json, {"key": "value"}, None),
        ("{}", str_to_json, {}, None),
        (
            '{"key": 1234, "nested": {"subkey": "value"}}',
            str_to_json,
            {"key": 1234, "nested": {"subkey": "value"}},
            None,
        ),
    ],
)
def test_converters(input_data, function, expected_output, expected_exception):  # type: ignore
    if expected_exception:
        # Expecting an exception to be raised
        with pytest.raises(expected_exception):
            function(input_data)
    else:
        # Valid behavior, ensure the result matches the expected output
        output = function(input_data)
        assert output == expected_output, (
            f"Expected {expected_output}, but got {output}"
        )
