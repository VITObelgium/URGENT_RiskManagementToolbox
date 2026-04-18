from typing import TypeVar

T = TypeVar("T")


def ensure_not_none(
    obj: T | None, error_message: str = "Expected a non-None object"
) -> T:
    """
    Ensures the provided object is not None. If it is None, raises an exception
    with a descriptive error message.

    :param obj: The object to check.
    :param error_message: Optional custom error message to raise if obj is None.
    :return: The non-None object.
    :raises ValueError: If obj is None.
    """
    if obj is None:
        raise ValueError(error_message)
    return obj
