"""Tests for ReportGenerator covering uncovered branches."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from services.solution_updater_service.core.reports.report_generator import (
    ReportGenerator,
    _to_obj_list,
    _sanitize_col,
    _generation_csv_columns,
)
from services.solution_updater_service.core.reports import PopulationStatistic


def _make_stat(results: np.ndarray, generation: int = 1) -> PopulationStatistic:
    return PopulationStatistic(
        generation=generation,
        n_feasible=len(results),
        results=results,
        min=results.min(axis=0) if results.ndim == 2 else float(results.min()),
        max=results.max(axis=0) if results.ndim == 2 else float(results.max()),
        avg=results.mean(axis=0) if results.ndim == 2 else float(results.mean()),
        std=results.std(axis=0) if results.ndim == 2 else float(results.std()),
    )


class TestLogPopulationStatisticBranches:
    def test_multi_row_results_skips_transpose(self):
        """results.shape[0] > 1 → branch 59→62 NOT taken."""
        gen = ReportGenerator(is_multi_objective=False)
        results = np.array([[1.0], [2.0], [3.0]])  # 3 individuals, 1 objective
        stat = _make_stat(results)

        with patch(
            "services.solution_updater_service.core.reports.report_generator.get_csv_logger"
        ) as mock_csv:
            mock_logger = MagicMock()
            mock_csv.return_value = mock_logger
            gen.log_population_statistic(
                generation=1,
                population_statistic=stat,
                results_name=["cost"],
            )

        mock_logger.info.assert_called()

    def test_single_row_1d_results_transposes(self):
        """results.shape[0]==1 and ndim==1 → branch 59→60 taken (transpose)."""
        gen = ReportGenerator(is_multi_objective=False)
        results = np.array([5.0])  # 1-D, single element
        stat = _make_stat(results)

        with patch(
            "services.solution_updater_service.core.reports.report_generator.get_csv_logger"
        ) as mock_csv:
            mock_logger = MagicMock()
            mock_csv.return_value = mock_logger
            gen.log_population_statistic(
                generation=1,
                population_statistic=stat,
                results_name=["cost"],
            )

        mock_logger.info.assert_called()


class TestLogBestResultMultiObjective:
    def test_multi_objective_happy_path(self):
        gen = ReportGenerator(is_multi_objective=True)
        pareto = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 solutions, 2 objectives

        with patch(
            "services.solution_updater_service.core.reports.report_generator.get_csv_logger"
        ) as mock_csv:
            mock_logger = MagicMock()
            mock_csv.return_value = mock_logger
            gen.log_best_result(
                generation=1,
                best_result=pareto,
                results_name=["obj1", "obj2"],
            )

        assert mock_logger.info.call_count == 2  # one row per Pareto solution

    def test_multi_objective_reuses_existing_logger(self):
        gen = ReportGenerator(is_multi_objective=True)
        pareto = np.array([[1.0, 2.0]])

        with patch(
            "services.solution_updater_service.core.reports.report_generator.get_csv_logger"
        ) as mock_csv:
            mock_logger = MagicMock()
            mock_csv.return_value = mock_logger
            gen.log_best_result(
                generation=1, best_result=pareto, results_name=["a", "b"]
            )
            gen.log_best_result(
                generation=2, best_result=pareto, results_name=["a", "b"]
            )

        # csv_logger created only once
        assert mock_csv.call_count == 1

    def test_multi_objective_wrong_size_raises(self):
        gen = ReportGenerator(is_multi_objective=True)
        pareto = np.array([[1.0, 2.0, 3.0]])  # 3 values but only 2 names

        with patch(
            "services.solution_updater_service.core.reports.report_generator.get_csv_logger"
        ):
            with pytest.raises(ValueError, match="expected 2"):
                gen.log_best_result(
                    generation=1,
                    best_result=pareto,
                    results_name=["obj1", "obj2"],
                )

    def test_single_objective_wrong_size_raises(self):
        gen = ReportGenerator(is_multi_objective=False)
        # 2 values but only 1 name
        with patch(
            "services.solution_updater_service.core.reports.report_generator.get_csv_logger"
        ):
            with pytest.raises(ValueError, match="Expected 1 values"):
                gen.log_best_result(
                    generation=1,
                    best_result=np.array([1.0, 2.0]),
                    results_name=["cost"],
                )


class TestToObjList:
    def test_scalar_single_objective(self):
        result = _to_obj_list(3.14, name="x", n_objectives=1)
        assert result == pytest.approx([3.14])

    def test_array_multi_objective(self):
        result = _to_obj_list(np.array([1.0, 2.0]), name="x", n_objectives=2)
        assert result == pytest.approx([1.0, 2.0])

    def test_size_mismatch_raises(self):
        with pytest.raises(ValueError, match="Expected x to have 3"):
            _to_obj_list(np.array([1.0, 2.0]), name="x", n_objectives=3)


class TestHelpers:
    def test_sanitize_col_replaces_special_chars(self):
        assert _sanitize_col("a,b c/d\\e") == "a_b_c_d_e"

    def test_generation_csv_columns_structure(self):
        cols = _generation_csv_columns(["obj1", "obj2"], pop_size=3)
        assert cols[0] == "generation"
        assert "min_obj1" in cols
        assert "max_obj2" in cols
        assert "ind_0_obj1" in cols
        assert "ind_2_obj2" in cols
