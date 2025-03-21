import pytest

from services.solution_updater_service.core.models.user import (
    ControlVector,
    CostFunctionResults,
    SolutionCandidate,
    SolutionUpdaterServiceRequest,
)


@pytest.mark.parametrize(
    "solution_candidates,expected_exception",
    [
        # Valid case where all candidates have the same  parameters and metrics
        (
            [
                SolutionCandidate(
                    control_vector=ControlVector(items={"param1": 1.0, "param2": 4.0}),
                    cost_function_results=CostFunctionResults(
                        values={"metric1": 1.5, "metric2": 0.5}
                    ),
                ),
                SolutionCandidate(
                    control_vector=ControlVector(items={"param1": 2.0, "param2": 3.0}),
                    cost_function_results=CostFunctionResults(
                        values={
                            "metric1": 2.5,
                            "metric2": 1.0,
                        }
                    ),
                ),
            ],
            None,  # No exception expected
        ),
        # Valid case where all candidates have the same  parameters and metrics but in random order
        (
            [
                SolutionCandidate(
                    control_vector=ControlVector(items={"param2": 1.0, "param1": 1.2}),
                    cost_function_results=CostFunctionResults(
                        values={"metric2": 1.5, "metric1": 0.3}
                    ),
                ),
                SolutionCandidate(
                    control_vector=ControlVector(items={"param1": 2.0, "param2": 3.0}),
                    cost_function_results=CostFunctionResults(
                        values={"metric1": 2.5, "metric2": 0.7}
                    ),
                ),
            ],
            None,  # No exception expected
        ),
        # Invalid case where all candidates have the same parameters
        (
            [
                SolutionCandidate(
                    control_vector=ControlVector(items={"param1": 1.0, "param2": 6.7}),
                    cost_function_results=CostFunctionResults(
                        values={"metric1": 1.5, "metric2": 0.3}
                    ),
                ),
                SolutionCandidate(
                    control_vector=ControlVector(
                        items={"param1": 2.0, "param_unmatched": 3.0}
                    ),
                    cost_function_results=CostFunctionResults(
                        values={"metric1": 2.5, "metric2": 0.7}
                    ),
                ),
            ],
            ValueError,  # No exception expected
        ),
        # Invalid case where metrics are inconsistent across candidates
        (
            [
                SolutionCandidate(
                    control_vector=ControlVector(items={"param1": 1.0, "param2": 9.8}),
                    cost_function_results=CostFunctionResults(
                        values={"metric1": 1.5, "metric2": 0.3}
                    ),
                ),
                SolutionCandidate(
                    control_vector=ControlVector(items={"param1": 2.0, "param2": 3.0}),
                    cost_function_results=CostFunctionResults(values={"metric1": 2.5}),
                ),
            ],
            ValueError,  # Expecting a ValueError due to inconsistent metrics
        ),
    ],
)
def test_solution_candidates_params_in_order(solution_candidates, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            # Create the OptimizationServiceConfig instance
            config = SolutionUpdaterServiceRequest(
                solution_candidates=solution_candidates
            )
            # Extract sorted keys from the first candidate's control vector
            sorted_keys = sorted(
                config.solution_candidates[0].control_vector.items.keys()
            )
            for candidate in config.solution_candidates:
                if list(candidate.control_vector.items.keys()) != sorted_keys:
                    raise ValueError(
                        "Control vector parameters are not in the correct order."
                    )
    else:
        # Create the OptimizationServiceConfig instance
        config = SolutionUpdaterServiceRequest(
            solution_candidates=solution_candidates,
        )
        # Extract sorted keys from the first candidate's control vector
        sorted_keys = sorted(config.solution_candidates[0].control_vector.items.keys())
        for candidate in config.solution_candidates:
            assert list(candidate.control_vector.items.keys()) == sorted_keys
