def validate_linear_inequalities(
    A: list[dict[str, float]] | None, b: list[float] | None, senses: list[str] | None
) -> None:
    if not isinstance(A, list) or not isinstance(b, list):
        raise ValueError("'A' and 'b' in linear_inequalities must be lists")
    if len(A) != len(b):
        raise ValueError("Number of rows in A must match length of b")

    if senses is not None:
        if not isinstance(senses, list):
            raise ValueError("'sense' must be a list when provided")
        if len(senses) != len(A):
            raise ValueError("Length of 'sense' must match number of rows in A")
        allowed = {"<=", ">=", "<", ">"}
        if not all(s in allowed for s in senses):
            raise ValueError(
                f"Invalid inequality direction(s) in 'sense'. Allowed: {sorted(allowed)}"
            )

    for row_idx, row in enumerate(A):
        if not isinstance(row, dict):
            raise TypeError(
                f"Row {row_idx} in A must be a dict mapping variable to coefficient"
            )
        if len(row) == 0:
            raise ValueError(f"Row {row_idx} in A is empty")
        for var, coef in row.items():
            if not isinstance(coef, (int, float)):
                raise TypeError(
                    f"Coefficient for {var} in row {row_idx} must be numeric"
                )
            if "." not in var:
                raise ValueError(
                    f"Variable '{var}' in linear inequalities must contain a '.' separating well and attribute (e.g., 'INJ.md')"
                )
        if not isinstance(b[row_idx], (int, float)):
            raise TypeError(f"b[{row_idx}] must be numeric")


def validate_boundaries(lb: float, ub: float) -> None:
    if lb >= ub:
        raise ValueError(
            f"Lower bound must be less than upper bound. Found lb={lb} and ub={ub}"
        )
