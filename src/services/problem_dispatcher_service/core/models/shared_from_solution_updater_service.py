from pydantic import BaseModel

type OptimizationVariable = str


class ControlVector(BaseModel, extra="forbid"):
    """
    Represents a control vector used in optimization processes.

    Attributes:
        items (dict[str, float]): A dictionary where keys are strings
            representing the parameter names and values are floats
            defining the parameter values.

    Notes:
        The `extra="forbid"` option ensures that only the defined fields
        are allowed when creating an instance of this model.
    """

    items: dict[OptimizationVariable, float]
