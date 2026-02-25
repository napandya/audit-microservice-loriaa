"""
Tests for utils.data_processor.

Validates that DataProcessor produces correct summaries and handles edge-cases
(empty DataFrames, missing columns, None input) gracefully.
"""

from __future__ import annotations

import pandas as pd
import pytest

from utils.data_processor import DataProcessor


@pytest.fixture()
def processor() -> DataProcessor:
    return DataProcessor()


# ---------------------------------------------------------------------------
# summarise_rent_roll
# ---------------------------------------------------------------------------


class TestSummariseRentRoll:
    def test_returns_string_for_valid_df(self, processor: DataProcessor) -> None:
        df = pd.DataFrame(
            {
                "unit_id": ["A1", "A2", "A3"],
                "tenant_name": ["Alice", "Bob", "Carol"],
                "monthly_rent": [1200, 1500, 1100],
                "occupancy_status": ["occupied", "occupied", "vacant"],
            }
        )
        result = processor.summarise_rent_roll(df)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_rent_stats_present(self, processor: DataProcessor) -> None:
        df = pd.DataFrame({"monthly_rent": [1000, 2000, 1500]})
        result = processor.summarise_rent_roll(df)
        assert "min=" in result
        assert "max=" in result
        assert "mean=" in result

    def test_column_aliases_normalised(self, processor: DataProcessor) -> None:
        df = pd.DataFrame({"rent": [1000, 1200], "unit": ["U1", "U2"]})
        result = processor.summarise_rent_roll(df)
        # After normalisation, monthly_rent stats should appear
        assert "min=" in result

    def test_occupancy_breakdown_present(self, processor: DataProcessor) -> None:
        df = pd.DataFrame(
            {
                "occupancy_status": ["occupied", "vacant", "occupied"],
                "monthly_rent": [1000, 0, 1200],
            }
        )
        result = processor.summarise_rent_roll(df)
        assert "occupied" in result

    def test_none_returns_no_data_message(self, processor: DataProcessor) -> None:
        result = processor.summarise_rent_roll(None)
        assert "No structured" in result

    def test_empty_df_returns_no_data_message(self, processor: DataProcessor) -> None:
        result = processor.summarise_rent_roll(pd.DataFrame())
        assert "No structured" in result

    def test_record_count_in_output(self, processor: DataProcessor) -> None:
        df = pd.DataFrame({"monthly_rent": [1000, 1100, 1200, 1300, 1400]})
        result = processor.summarise_rent_roll(df)
        assert "5 records" in result

    def test_all_non_numeric_rent_values_handled_gracefully(
        self, processor: DataProcessor
    ) -> None:
        # All values are non-numeric; pd.to_numeric(..., errors="coerce") will
        # produce an all-NaN Series. This should not raise.
        df = pd.DataFrame(
            {
                "monthly_rent": ["N/A", "unknown", "not available"],
                "unit_id": ["U1", "U2", "U3"],
            }
        )
        result = processor.summarise_rent_roll(df)
        assert isinstance(result, str)
        assert "no valid numeric values" in result

    def test_all_non_numeric_sqft_values_handled_gracefully(
        self, processor: DataProcessor
    ) -> None:
        df = pd.DataFrame({"sq_ft": ["n/a", "?", "unknown"]})
        result = processor.summarise_rent_roll(df)
        assert "no valid numeric values" in result


# ---------------------------------------------------------------------------
# summarise_rent_projections
# ---------------------------------------------------------------------------


class TestSummariseRentProjections:
    def test_returns_string_for_valid_df(self, processor: DataProcessor) -> None:
        df = pd.DataFrame(
            {
                "period": ["Jan", "Feb", "Mar"],
                "projected_rent": [10000, 10500, 11000],
                "actual_rent": [9800, 10600, 10900],
            }
        )
        result = processor.summarise_rent_projections(df)
        assert isinstance(result, str)
        assert "Rent Projections" in result

    def test_variance_stats_present_when_both_columns_available(
        self, processor: DataProcessor
    ) -> None:
        df = pd.DataFrame(
            {
                "projected_rent": [1000, 2000],
                "actual_rent": [900, 2100],
            }
        )
        result = processor.summarise_rent_projections(df)
        assert "variance" in result.lower()

    def test_alias_forecast_normalised(self, processor: DataProcessor) -> None:
        df = pd.DataFrame({"forecast": [5000, 5500], "actual": [4800, 5600]})
        result = processor.summarise_rent_projections(df)
        # Should not raise; columns should be normalised
        assert isinstance(result, str)

    def test_all_non_numeric_projected_rent_handled_gracefully(
        self, processor: DataProcessor
    ) -> None:
        df = pd.DataFrame(
            {"projected_rent": ["N/A", "?"], "actual_rent": ["N/A", "?"]}
        )
        result = processor.summarise_rent_projections(df)
        assert "no valid numeric values" in result

    def test_all_non_numeric_variance_handled_gracefully(
        self, processor: DataProcessor
    ) -> None:
        df = pd.DataFrame({"variance": ["N/A", "unknown"]})
        result = processor.summarise_rent_projections(df)
        assert "no valid numeric values" in result

    def test_none_returns_no_data_message(self, processor: DataProcessor) -> None:
        result = processor.summarise_rent_projections(None)
        assert "No structured" in result


# ---------------------------------------------------------------------------
# summarise_concessions
# ---------------------------------------------------------------------------


class TestSummariseConcessions:
    def test_returns_string_for_valid_df(self, processor: DataProcessor) -> None:
        df = pd.DataFrame(
            {
                "unit_id": ["A1", "B2"],
                "concession_amount": [200, 150],
                "concession_type": ["free_month", "discount"],
                "reason": ["new tenant", "renewal"],
            }
        )
        result = processor.summarise_concessions(df)
        assert isinstance(result, str)
        assert "Concessions" in result

    def test_total_concession_amount_in_output(self, processor: DataProcessor) -> None:
        df = pd.DataFrame({"concession_amount": [100, 200, 300]})
        result = processor.summarise_concessions(df)
        assert "total=" in result
        assert "600" in result  # sum of 100+200+300

    def test_alias_discount_normalised(self, processor: DataProcessor) -> None:
        df = pd.DataFrame({"discount": [50, 75]})
        result = processor.summarise_concessions(df)
        assert "total=" in result

    def test_all_non_numeric_concession_amounts_handled_gracefully(
        self, processor: DataProcessor
    ) -> None:
        df = pd.DataFrame({"concession_amount": ["N/A", "unknown", "n/a"]})
        result = processor.summarise_concessions(df)
        assert "no valid numeric values" in result

    def test_type_breakdown_present(self, processor: DataProcessor) -> None:
        df = pd.DataFrame(
            {
                "concession_type": ["free_month", "discount", "free_month"],
                "concession_amount": [1200, 100, 1300],
            }
        )
        result = processor.summarise_concessions(df)
        assert "free_month" in result

    def test_none_returns_no_data_message(self, processor: DataProcessor) -> None:
        result = processor.summarise_concessions(None)
        assert "No structured" in result

    def test_empty_df_returns_no_data_message(self, processor: DataProcessor) -> None:
        result = processor.summarise_concessions(pd.DataFrame())
        assert "No structured" in result


# ---------------------------------------------------------------------------
# _normalise_columns (via public methods to avoid testing private API directly)
# ---------------------------------------------------------------------------


class TestColumnNormalisation:
    def test_case_insensitive_alias_match(self, processor: DataProcessor) -> None:
        df = pd.DataFrame({"Monthly Rent": [1000], "Unit": ["A1"]})
        # Should work without raising
        result = processor.summarise_rent_roll(df)
        assert isinstance(result, str)

    def test_unrecognised_columns_preserved(self, processor: DataProcessor) -> None:
        df = pd.DataFrame({"custom_field": ["x"], "monthly_rent": [1000]})
        result = processor.summarise_rent_roll(df)
        assert "custom_field" in result
