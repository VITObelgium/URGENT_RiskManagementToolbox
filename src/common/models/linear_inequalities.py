from __future__ import annotations

from typing import Self

from pydantic import BaseModel, Field, model_validator


class LinearInequalities(BaseModel, extra="forbid"):
    """Model for linear inequality constraints: A @ x (sense) b.

    Each row of ``A`` is a sparse dict mapping ``"WellName.attribute"`` to its
    coefficient.  ``b`` is the RHS vector.  ``sense`` defines the direction
    per row (``"<="`` | ``">="`` | ``"<"`` | ``">"``); defaults to ``"<="``
    for every row when omitted.
    """

    A: list[dict[str, float]] = Field(
        ...,
        min_length=1,
        description="Sparse coefficient matrix (list of row-dicts)",
    )
    b: list[float] = Field(
        ...,
        min_length=1,
        description="Right-hand-side vector",
    )
    sense: list[str] | None = Field(
        default=None,
        description="Inequality direction per row; defaults to '<=' when omitted",
    )

    _ALLOWED_SENSES: frozenset[str] = frozenset({"<=", ">=", "<", ">"})

    @model_validator(mode="after")
    def validate_linear_inequalities(self) -> Self:
        if len(self.A) != len(self.b):
            raise ValueError("A row count must equal length of b")

        if self.sense is not None:
            if len(self.sense) != len(self.A):
                raise ValueError("sense length must match number of A rows")
            for s in self.sense:
                if s not in self._ALLOWED_SENSES:
                    raise ValueError(
                        f"Invalid inequality direction '{s}'. "
                        f"Allowed: {sorted(self._ALLOWED_SENSES)}"
                    )
        else:
            self.sense = ["<="] * len(self.A)

        for idx, row in enumerate(self.A):
            if not isinstance(row, dict) or not row:
                raise ValueError(f"Row {idx} in A must be a non-empty dict")
            for var, coef in row.items():
                if not isinstance(coef, (int, float)):
                    raise TypeError(
                        f"Coefficient for variable '{var}' in row {idx} must be numeric"
                    )
                if "." not in var:
                    raise ValueError(
                        f"Variable '{var}' must contain '.' separating well "
                        "and attribute (e.g., 'INJ.md')"
                    )

        for i, bi in enumerate(self.b):
            if not isinstance(bi, (int, float)):
                raise TypeError(f"b[{i}] must be numeric")

        return self
