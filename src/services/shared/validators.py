def validate_linear_inequalities(
    A: list[dict[str, float]] | None, b: list[float] | None, senses: list[str] | None
) -> None:
    # Early return if all are None
    if A is None and b is None and senses is None:
        return

    # Validate all required parts are present
    if A is None or b is None:
        raise ValueError(
            "Both 'A' and 'b' must be provided together in linear_inequalities"
        )

    # Type validation
    if not isinstance(A, list):
        raise TypeError(f"'A' must be a list, got {type(A).__name__}")
    if not isinstance(b, list):
        raise TypeError(f"'b' must be a list, got {type(b).__name__}")

    # Length validation
    num_constraints = len(A)
    if len(b) != num_constraints:
        raise ValueError(
            f"Number of rows in A ({len(A)}) must match length of b ({len(b)})"
        )

    # Senses validation
    if senses is not None:
        if not isinstance(senses, list):
            raise TypeError(f"'sense' must be a list, got {type(senses).__name__}")
        if len(senses) != num_constraints:
            raise ValueError(
                f"Length of 'sense' ({len(senses)}) must match number of constraints ({num_constraints})"
            )
        allowed = {"<=", ">=", "<", ">"}
        invalid_senses = [s for s in senses if s not in allowed]
        if invalid_senses:
            raise ValueError(
                f"Invalid inequality direction(s): {invalid_senses}. "
                f"Allowed: {sorted(allowed)}"
            )

    # Row-by-row validation
    for row_idx, (row, b_val) in enumerate(zip(A, b)):
        # Row type and content
        if not isinstance(row, dict):
            raise TypeError(
                f"Row {row_idx} in A must be a dict, got {type(row).__name__}"
            )
        if not row:
            raise ValueError(f"Row {row_idx} in A is empty")

        # Coefficient validation
        for var, coef in row.items():
            if not isinstance(coef, (int, float)):
                raise TypeError(
                    f"Coefficient for '{var}' in row {row_idx} must be numeric, "
                    f"got {type(coef).__name__}"
                )
            if "." not in var:
                raise ValueError(
                    f"Variable '{var}' in row {row_idx} must contain a '.' "
                    f"separating well and attribute (e.g., 'INJ.md')"
                )

        # b value validation
        if not isinstance(b_val, (int, float)):
            raise TypeError(f"b[{row_idx}] must be numeric, got {type(b_val).__name__}")


def validate_boundaries(lb: float, ub: float) -> None:
    # Type validation
    if not isinstance(lb, (int, float)):
        raise TypeError(f"Lower bound must be numeric, got {type(lb).__name__}")
    if not isinstance(ub, (int, float)):
        raise TypeError(f"Upper bound must be numeric, got {type(ub).__name__}")

    # Validate bounds are finite (optional, depends on your use case)
    import math

    if math.isnan(lb) or math.isnan(ub):
        raise ValueError("Bounds cannot be NaN")

    # Main validation
    if lb >= ub:
        raise ValueError(
            f"Lower bound must be strictly less than upper bound. Got lb={lb}, ub={ub}"
        )
