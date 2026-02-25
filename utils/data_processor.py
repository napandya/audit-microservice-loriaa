"""
Data processor module.

Provides helpers for normalising and summarising structured data extracted
from rent roll, rent projections, and concessions documents before they are
forwarded to the AI audit agent.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


class DataProcessor:
    """Normalise and summarise tabular data for audit consumption.

    All public methods accept an optional ``pandas.DataFrame`` (which may be
    ``None`` for unstructured formats such as PDF/Word) and return enriched
    summary strings that the agent can reason over.
    """

    # Common column aliases → canonical names
    _RENT_ROLL_ALIASES: dict[str, str] = {
        "unit": "unit_id",
        "unit_id": "unit_id",
        "unit id": "unit_id",
        "tenant": "tenant_name",
        "tenant name": "tenant_name",
        "name": "tenant_name",
        "rent": "monthly_rent",
        "monthly rent": "monthly_rent",
        "monthly_rent": "monthly_rent",
        "lease start": "lease_start",
        "lease_start": "lease_start",
        "start date": "lease_start",
        "lease end": "lease_end",
        "lease_end": "lease_end",
        "end date": "lease_end",
        "sq ft": "sq_ft",
        "sqft": "sq_ft",
        "sq_ft": "sq_ft",
        "square feet": "sq_ft",
        "status": "occupancy_status",
        "occupancy": "occupancy_status",
        "occupancy_status": "occupancy_status",
    }

    _PROJECTION_ALIASES: dict[str, str] = {
        "month": "period",
        "period": "period",
        "date": "period",
        "projected rent": "projected_rent",
        "projected_rent": "projected_rent",
        "forecast": "projected_rent",
        "actual rent": "actual_rent",
        "actual_rent": "actual_rent",
        "actual": "actual_rent",
        "variance": "variance",
        "difference": "variance",
    }

    _CONCESSION_ALIASES: dict[str, str] = {
        "unit": "unit_id",
        "unit_id": "unit_id",
        "concession": "concession_amount",
        "concession amount": "concession_amount",
        "concession_amount": "concession_amount",
        "discount": "concession_amount",
        "type": "concession_type",
        "concession type": "concession_type",
        "concession_type": "concession_type",
        "reason": "reason",
        "notes": "reason",
        "approved by": "approved_by",
        "approved_by": "approved_by",
    }

    def summarise_rent_roll(self, df: Optional[pd.DataFrame]) -> str:
        """Return a human-readable summary of rent roll data."""
        if df is None or df.empty:
            return "No structured rent roll data available."

        df = self._normalise_columns(df, self._RENT_ROLL_ALIASES)
        lines: list[str] = [f"Rent Roll — {len(df)} records"]

        if "monthly_rent" in df.columns:
            rent_series = pd.to_numeric(df["monthly_rent"], errors="coerce")
            lines.append(
                f"  Monthly rent: min={rent_series.min():.2f}, "
                f"max={rent_series.max():.2f}, "
                f"mean={rent_series.mean():.2f}, "
                f"null_count={rent_series.isna().sum()}"
            )

        if "occupancy_status" in df.columns:
            status_counts = df["occupancy_status"].value_counts().to_dict()
            lines.append(f"  Occupancy status breakdown: {status_counts}")

        if "sq_ft" in df.columns:
            sqft = pd.to_numeric(df["sq_ft"], errors="coerce")
            lines.append(
                f"  Sq ft: min={sqft.min():.0f}, max={sqft.max():.0f}"
            )

        lines.append(f"  Columns present: {list(df.columns)}")
        lines.append(f"\nFull data sample (first 20 rows):\n{df.head(20).to_string(index=False)}")
        return "\n".join(lines)

    def summarise_rent_projections(self, df: Optional[pd.DataFrame]) -> str:
        """Return a human-readable summary of rent projection data."""
        if df is None or df.empty:
            return "No structured rent projection data available."

        df = self._normalise_columns(df, self._PROJECTION_ALIASES)
        lines: list[str] = [f"Rent Projections — {len(df)} records"]

        if "projected_rent" in df.columns and "actual_rent" in df.columns:
            proj = pd.to_numeric(df["projected_rent"], errors="coerce")
            actual = pd.to_numeric(df["actual_rent"], errors="coerce")
            computed_variance = (actual - proj).abs()
            lines.append(
                f"  Projected vs actual variance: "
                f"mean={computed_variance.mean():.2f}, "
                f"max={computed_variance.max():.2f}"
            )

        if "variance" in df.columns:
            var_series = pd.to_numeric(df["variance"], errors="coerce")
            lines.append(
                f"  Reported variance: "
                f"min={var_series.min():.2f}, max={var_series.max():.2f}"
            )

        lines.append(f"  Columns present: {list(df.columns)}")
        lines.append(f"\nFull data sample (first 20 rows):\n{df.head(20).to_string(index=False)}")
        return "\n".join(lines)

    def summarise_concessions(self, df: Optional[pd.DataFrame]) -> str:
        """Return a human-readable summary of concessions data."""
        if df is None or df.empty:
            return "No structured concessions data available."

        df = self._normalise_columns(df, self._CONCESSION_ALIASES)
        lines: list[str] = [f"Concessions — {len(df)} records"]

        if "concession_amount" in df.columns:
            amounts = pd.to_numeric(df["concession_amount"], errors="coerce")
            lines.append(
                f"  Concession amounts: total={amounts.sum():.2f}, "
                f"mean={amounts.mean():.2f}, "
                f"max={amounts.max():.2f}"
            )

        if "concession_type" in df.columns:
            type_counts = df["concession_type"].value_counts().to_dict()
            lines.append(f"  Concession type breakdown: {type_counts}")

        lines.append(f"  Columns present: {list(df.columns)}")
        lines.append(f"\nFull data sample (first 20 rows):\n{df.head(20).to_string(index=False)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_columns(
        df: pd.DataFrame, alias_map: dict[str, str]
    ) -> pd.DataFrame:
        """Rename dataframe columns using *alias_map* (case-insensitive)."""
        rename: dict[str, str] = {}
        for col in df.columns:
            canonical = alias_map.get(col.strip().lower())
            if canonical and canonical != col:
                rename[col] = canonical
        return df.rename(columns=rename)
