import numpy as np
import pytest

from common import OptimizationStrategy
from services.solution_updater_service.core.reports.population_statistic import (
    PopulationStatistic,
    PopulationStatisticGenerator,
)


class TestPopulationStatisticDataclass:
    def test_fields_are_frozen(self):
        stat = PopulationStatistic(
            generation=0,
            min=1.0,
            max=5.0,
            avg=3.0,
            std=1.0,
            n_feasible=3,
            results=np.array([1.0, 2.0, 3.0]),
        )
        with pytest.raises((AttributeError, TypeError)):
            stat.generation = 99  # type: ignore[misc]


class TestPopulationStatisticGeneratorSingle:
    def test_generate_with_1d_results(self):
        """1D results array should be expanded to column vector."""
        results = np.array([1.0, 2.0, 3.0, 4.0])
        objectives = {0: OptimizationStrategy.MINIMIZE}
        stat = PopulationStatisticGenerator.generate(0, results, objectives)
        assert stat.generation == 0
        assert stat.min == pytest.approx(1.0)
        assert stat.max == pytest.approx(4.0)
        assert stat.n_feasible == 4

    def test_generate_single_minimize_basic(self):
        results = np.array([[1.0], [3.0], [2.0]])
        objectives = {0: OptimizationStrategy.MINIMIZE}
        stat = PopulationStatisticGenerator.generate(5, results, objectives)
        assert stat.generation == 5
        assert stat.min == pytest.approx(1.0)
        assert stat.max == pytest.approx(3.0)
        assert stat.avg == pytest.approx(2.0)
        assert stat.n_feasible == 3

    def test_generate_single_maximize_basic(self):
        results = np.array([[10.0], [30.0], [20.0]])
        objectives = {0: OptimizationStrategy.MAXIMIZE}
        stat = PopulationStatisticGenerator.generate(1, results, objectives)
        assert stat.min == pytest.approx(10.0)
        assert stat.max == pytest.approx(30.0)
        assert stat.n_feasible == 3

    def test_generate_single_all_infeasible_minimize(self):
        """All-inf results should return sentinel inf values for MINIMIZE."""
        results = np.array([[np.inf], [np.inf], [np.inf]])
        objectives = {0: OptimizationStrategy.MINIMIZE}
        stat = PopulationStatisticGenerator.generate(0, results, objectives)
        assert stat.n_feasible == 0
        assert stat.min == np.inf
        assert stat.max == np.inf
        assert np.isnan(stat.avg)
        assert np.isnan(stat.std)

    def test_generate_single_all_infeasible_maximize(self):
        """All-inf results should return sentinel -inf values for MAXIMIZE."""
        results = np.array([[-np.inf], [-np.inf]])
        objectives = {0: OptimizationStrategy.MAXIMIZE}
        stat = PopulationStatisticGenerator.generate(0, results, objectives)
        assert stat.n_feasible == 0
        assert stat.min == -np.inf
        assert stat.max == -np.inf
        assert np.isnan(stat.avg)
        assert np.isnan(stat.std)

    def test_generate_single_partial_infeasible(self):
        """Mix of finite and infinite values - only finite should be counted."""
        results = np.array([[1.0], [np.inf], [3.0], [np.inf]])
        objectives = {0: OptimizationStrategy.MINIMIZE}
        stat = PopulationStatisticGenerator.generate(2, results, objectives)
        assert stat.n_feasible == 2
        assert stat.min == pytest.approx(1.0)
        assert stat.max == pytest.approx(3.0)
        assert stat.avg == pytest.approx(2.0)


class TestPopulationStatisticGeneratorMulti:
    def test_generate_multi_basic(self):
        results = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        objectives = {
            0: OptimizationStrategy.MINIMIZE,
            1: OptimizationStrategy.MAXIMIZE,
        }
        stat = PopulationStatisticGenerator.generate(0, results, objectives)
        assert stat.n_feasible == 3
        assert np.allclose(stat.min, [1.0, 4.0])
        assert np.allclose(stat.max, [3.0, 6.0])
        assert np.allclose(stat.avg, [2.0, 5.0])

    def test_generate_multi_all_infeasible(self):
        """All-inf results for multi-objective should yield nan arrays."""
        results = np.array([[np.inf, np.inf], [np.inf, np.inf]])
        objectives = {
            0: OptimizationStrategy.MINIMIZE,
            1: OptimizationStrategy.MINIMIZE,
        }
        stat = PopulationStatisticGenerator.generate(0, results, objectives)
        assert stat.n_feasible == 0
        assert np.all(np.isnan(stat.min))
        assert np.all(np.isnan(stat.max))
        assert np.all(np.isnan(stat.avg))
        assert np.all(np.isnan(stat.std))

    def test_generate_multi_partial_infeasible(self):
        """Rows with any inf are excluded in multi-objective."""
        results = np.array([[1.0, 2.0], [np.inf, 3.0], [4.0, 5.0]])
        objectives = {
            0: OptimizationStrategy.MINIMIZE,
            1: OptimizationStrategy.MAXIMIZE,
        }
        stat = PopulationStatisticGenerator.generate(0, results, objectives)
        assert stat.n_feasible == 2
        assert np.allclose(stat.min, [1.0, 2.0])
        assert np.allclose(stat.max, [4.0, 5.0])

    def test_generate_multi_results_stored_as_2d(self):
        results = np.array([[1.0, 2.0], [3.0, 4.0]])
        objectives = {
            0: OptimizationStrategy.MINIMIZE,
            1: OptimizationStrategy.MAXIMIZE,
        }
        stat = PopulationStatisticGenerator.generate(0, results, objectives)
        assert stat.results.ndim == 2
        assert stat.results.shape == (2, 2)

    def test_generate_multi_std_computation(self):
        results = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
        objectives = {
            0: OptimizationStrategy.MINIMIZE,
            1: OptimizationStrategy.MAXIMIZE,
        }
        stat = PopulationStatisticGenerator.generate(0, results, objectives)
        expected_std_0 = np.std([1.0, 3.0, 5.0])
        expected_std_1 = np.std([10.0, 20.0, 30.0])
        assert np.allclose(stat.std, [expected_std_0, expected_std_1])
