"""
Tests for agents.audit_agent.

Validates the AuditAgent helper methods (_extract_summary, _extract_anomalies),
the rule-based analysis helpers (_analyse_rent_roll, _analyse_projections,
_analyse_concessions, _consolidate_findings), and the AuditAgent.run() flow
using a mocked LangGraph agent so no real OpenAI API calls are made.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.audit_agent import (
    AuditAgent,
    AuditResult,
    _analyse_rent_roll,
    _analyse_projections,
    _analyse_concessions,
    _consolidate_findings,
)


# ---------------------------------------------------------------------------
# _analyse_rent_roll
# ---------------------------------------------------------------------------


class TestAnalyseRentRoll:
    def _summary(self, **kwargs) -> str:
        """Build a minimal data_summary string from keyword overrides."""
        defaults = {
            "records": 10,
            "min_rent": 1000.0,
            "max_rent": 2000.0,
            "mean_rent": 1500.0,
            "null_count": 0,
            "occupied": 8,
            "vacant": 2,
        }
        defaults.update(kwargs)
        return (
            f"Rent Roll — {defaults['records']} records\n"
            f"  Monthly rent: min={defaults['min_rent']:.2f}, "
            f"max={defaults['max_rent']:.2f}, "
            f"mean={defaults['mean_rent']:.2f}, "
            f"null_count={defaults['null_count']}\n"
            f"  Occupancy status breakdown: "
            f"{{'occupied': {defaults['occupied']}, 'vacant': {defaults['vacant']}}}\n"
        )

    def test_no_anomalies_for_clean_data(self) -> None:
        findings = _analyse_rent_roll(self._summary())
        # No rule-based anomalies should fire on clean data
        assert isinstance(findings, list)

    def test_detects_null_rent_values(self) -> None:
        findings = _analyse_rent_roll(self._summary(null_count=3))
        types = [f["type"] for f in findings]
        assert "missing_rent_values" in types
        assert any(f["severity"] == "high" for f in findings if f["type"] == "missing_rent_values")

    def test_detects_zero_rent(self) -> None:
        findings = _analyse_rent_roll(self._summary(min_rent=0.0))
        types = [f["type"] for f in findings]
        assert "zero_rent" in types
        assert any(f["severity"] == "high" for f in findings if f["type"] == "zero_rent")

    def test_detects_negative_rent(self) -> None:
        findings = _analyse_rent_roll(self._summary(min_rent=-500.0))
        types = [f["type"] for f in findings]
        assert "negative_rent" in types
        assert any(f["severity"] == "critical" for f in findings if f["type"] == "negative_rent")

    def test_detects_rent_outlier(self) -> None:
        # max > 3 × mean → outlier
        findings = _analyse_rent_roll(self._summary(min_rent=1000.0, max_rent=6000.0, mean_rent=1500.0))
        types = [f["type"] for f in findings]
        assert "rent_statistical_outlier" in types

    def test_no_outlier_when_max_below_threshold(self) -> None:
        findings = _analyse_rent_roll(self._summary(min_rent=1000.0, max_rent=2000.0, mean_rent=1500.0))
        types = [f["type"] for f in findings]
        assert "rent_statistical_outlier" not in types

    def test_detects_high_vacancy_rate(self) -> None:
        # 50 % vacancy (5 of 10)
        findings = _analyse_rent_roll(self._summary(occupied=5, vacant=5))
        types = [f["type"] for f in findings]
        assert "high_vacancy_rate" in types

    def test_no_vacancy_flag_below_threshold(self) -> None:
        findings = _analyse_rent_roll(self._summary(occupied=8, vacant=2))
        types = [f["type"] for f in findings]
        assert "high_vacancy_rate" not in types

    def test_detects_non_numeric_rent(self) -> None:
        summary = "Rent Roll — 3 records\n  Monthly rent: no valid numeric values, null_count=3\n"
        findings = _analyse_rent_roll(summary)
        types = [f["type"] for f in findings]
        assert "non_numeric_rent" in types
        assert any(f["severity"] == "critical" for f in findings)

    def test_each_finding_has_required_keys(self) -> None:
        findings = _analyse_rent_roll(self._summary(null_count=2, min_rent=0.0))
        for f in findings:
            assert "type" in f
            assert "severity" in f
            assert "affected" in f
            assert "description" in f
            assert "recommended_action" in f


# ---------------------------------------------------------------------------
# _analyse_projections
# ---------------------------------------------------------------------------


class TestAnalyseProjections:
    def _summary(self, **kwargs) -> str:
        defaults = {
            "var_mean": 100.0,
            "var_max": 200.0,
            "proj_mean": 5000.0,
        }
        defaults.update(kwargs)
        return (
            f"Rent Projections — 12 records\n"
            f"  Mean projected rent: {defaults['proj_mean']:.2f}\n"
            f"  Projected vs actual variance: "
            f"mean={defaults['var_mean']:.2f}, max={defaults['var_max']:.2f}\n"
        )

    def test_detects_critical_variance(self) -> None:
        # 2000/5000 = 40% > 25% threshold → critical
        findings = _analyse_projections(self._summary(var_max=2000.0, proj_mean=5000.0))
        types = [f["type"] for f in findings]
        assert "critical_projection_variance" in types
        assert any(f["severity"] == "critical" for f in findings)

    def test_detects_notable_variance(self) -> None:
        # 600/5000 = 12% > 10% but < 25% → high
        findings = _analyse_projections(self._summary(var_max=600.0, proj_mean=5000.0))
        types = [f["type"] for f in findings]
        assert "notable_projection_variance" in types
        assert any(f["severity"] == "high" for f in findings if f["type"] == "notable_projection_variance")

    def test_no_variance_flag_for_low_values(self) -> None:
        # 100/5000 = 2% — well below threshold
        findings = _analyse_projections(self._summary(var_mean=50.0, var_max=100.0, proj_mean=5000.0))
        types = [f["type"] for f in findings]
        assert "critical_projection_variance" not in types
        assert "notable_projection_variance" not in types

    def test_detects_non_numeric_projections(self) -> None:
        summary = "Rent Projections — 3 records\n  Projected vs actual variance: no valid numeric values\n"
        findings = _analyse_projections(summary)
        types = [f["type"] for f in findings]
        assert "non_numeric_projections" in types

    def test_detects_negative_reported_variance(self) -> None:
        summary = "Rent Projections — 5 records\n  Reported variance: min=-500.00, max=200.00\n  mean=3000.00\n"
        findings = _analyse_projections(summary)
        types = [f["type"] for f in findings]
        assert "negative_reported_variance" in types


# ---------------------------------------------------------------------------
# _analyse_concessions
# ---------------------------------------------------------------------------


class TestAnalyseConcessions:
    def _summary(self, **kwargs) -> str:
        defaults = {
            "records": 10,
            "total": 5000.0,
            "mean": 500.0,
            "max": 600.0,
            "types": "free_month",
        }
        defaults.update(kwargs)
        return (
            f"Concessions — {defaults['records']} records\n"
            f"  Concession amounts: total={defaults['total']:.2f}, "
            f"mean={defaults['mean']:.2f}, max={defaults['max']:.2f}\n"
            f"  Concession type breakdown: {{'{defaults['types']}': {defaults['records']}}}\n"
        )

    def test_detects_concession_outlier(self) -> None:
        # max = 2000, mean = 400 → 2000 > 3×400 → outlier
        findings = _analyse_concessions(self._summary(mean=400.0, max=2000.0))
        types = [f["type"] for f in findings]
        assert "concession_outlier" in types
        assert any(f["severity"] == "high" for f in findings if f["type"] == "concession_outlier")

    def test_no_outlier_when_max_below_threshold(self) -> None:
        findings = _analyse_concessions(self._summary(mean=500.0, max=600.0))
        types = [f["type"] for f in findings]
        assert "concession_outlier" not in types

    def test_detects_unrecognised_concession_type(self) -> None:
        findings = _analyse_concessions(self._summary(types="custom_unknown_type"))
        types = [f["type"] for f in findings]
        assert "unrecognised_concession_type" in types

    def test_no_flag_for_approved_type(self) -> None:
        findings = _analyse_concessions(self._summary(types="free_month"))
        types = [f["type"] for f in findings]
        assert "unrecognised_concession_type" not in types

    def test_detects_non_numeric_concession_amount(self) -> None:
        summary = "Concessions — 3 records\n  Concession amounts: no valid numeric values\n"
        findings = _analyse_concessions(summary)
        types = [f["type"] for f in findings]
        assert "non_numeric_concessions" in types


# ---------------------------------------------------------------------------
# _consolidate_findings
# ---------------------------------------------------------------------------


class TestConsolidateFindings:
    def test_counts_by_severity(self) -> None:
        findings = [
            {"severity": "critical", "type": "x"},
            {"severity": "high", "type": "y"},
            {"severity": "high", "type": "z"},
            {"severity": "low", "type": "w"},
        ]
        result = _consolidate_findings(json.dumps(findings))
        assert result["severity_counts"]["critical"] == 1
        assert result["severity_counts"]["high"] == 2
        assert result["severity_counts"]["low"] == 1
        assert result["total_findings"] == 4

    def test_risk_level_critical_when_any_critical(self) -> None:
        findings = [{"severity": "critical", "type": "x"}]
        result = _consolidate_findings(json.dumps(findings))
        assert result["risk_level"] == "CRITICAL"

    def test_risk_level_high_when_no_critical(self) -> None:
        findings = [{"severity": "high", "type": "x"}]
        result = _consolidate_findings(json.dumps(findings))
        assert result["risk_level"] == "HIGH"

    def test_risk_level_low_for_empty(self) -> None:
        result = _consolidate_findings(json.dumps([]))
        assert result["risk_level"] == "LOW"
        assert result["total_findings"] == 0

    def test_handles_malformed_json(self) -> None:
        result = _consolidate_findings("not valid json{")
        assert result["total_findings"] == 0
        assert result["risk_level"] == "LOW"



# _extract_summary
# ---------------------------------------------------------------------------


class TestExtractSummary:
    def test_extracts_text_after_executive_summary_heading(self) -> None:
        response = (
            "## Executive Summary\n"
            "3 anomalies found across 2 document types.\n"
            "\n"
            "## Detailed Findings\n"
            "More details here."
        )
        result = AuditAgent._extract_summary(response)
        assert "3 anomalies found" in result
        assert "Detailed Findings" not in result

    def test_returns_first_500_chars_when_no_heading_found(self) -> None:
        response = "No summary heading here. Just a plain response."
        result = AuditAgent._extract_summary(response)
        assert result == response[:500]

    def test_case_insensitive_heading_match(self) -> None:
        response = "## EXECUTIVE SUMMARY\nSome content.\n## Next Section\nOther."
        result = AuditAgent._extract_summary(response)
        assert "Some content" in result

    def test_empty_response_returns_empty_string_slice(self) -> None:
        result = AuditAgent._extract_summary("")
        assert result == ""

    def test_summary_stops_at_next_h2(self) -> None:
        response = (
            "## Executive Summary\n"
            "Line A\n"
            "Line B\n"
            "## Risk Assessment\n"
            "Should not appear."
        )
        result = AuditAgent._extract_summary(response)
        assert "Line A" in result
        assert "Line B" in result
        assert "Should not appear" not in result


# ---------------------------------------------------------------------------
# _extract_anomalies
# ---------------------------------------------------------------------------


class TestExtractAnomalies:
    # --- JSON block parsing (primary path) ---

    def test_parses_valid_json_block(self) -> None:
        response = (
            "Some markdown report.\n\n"
            "```json\n"
            '[{"severity": "critical", "document_type": "rent_roll", '
            '"affected": "Unit 101", "description": "Rent is zero.", '
            '"recommended_action": "Fix it."}]\n'
            "```"
        )
        anomalies = AuditAgent._extract_anomalies(response)
        assert len(anomalies) == 1
        assert anomalies[0]["severity"] == "critical"
        assert anomalies[0]["document_type"] == "rent_roll"
        assert anomalies[0]["affected"] == "Unit 101"
        assert anomalies[0]["description"] == "Rent is zero."
        assert anomalies[0]["recommended_action"] == "Fix it."

    def test_parses_multiple_json_anomalies(self) -> None:
        response = (
            "Report text.\n\n"
            "```json\n"
            '[{"severity": "high", "document_type": "projections", "affected": "Jan", '
            '"description": "Variance >25%.", "recommended_action": "Investigate."},'
            ' {"severity": "low", "document_type": "concessions", "affected": "Unit 5", '
            '"description": "Missing reason.", "recommended_action": "Add notes."}]\n'
            "```"
        )
        anomalies = AuditAgent._extract_anomalies(response)
        assert len(anomalies) == 2
        severities = {a["severity"] for a in anomalies}
        assert severities == {"high", "low"}

    def test_returns_empty_list_for_empty_json_array(self) -> None:
        response = "All clear.\n\n```json\n[]\n```"
        anomalies = AuditAgent._extract_anomalies(response)
        assert anomalies == []

    def test_skips_items_with_invalid_severity_in_json(self) -> None:
        response = (
            "```json\n"
            '[{"severity": "unknown", "document_type": "rent_roll", '
            '"affected": "X", "description": "Bad.", "recommended_action": "None."}]\n'
            "```"
        )
        anomalies = AuditAgent._extract_anomalies(response)
        assert anomalies == []

    def test_handles_malformed_json_gracefully_via_fallback(self) -> None:
        # Malformed JSON → should not raise; falls back to keyword matching
        response = (
            "```json\n"
            "[{broken json\n"
            "```\n"
            "- **high** — some issue."
        )
        anomalies = AuditAgent._extract_anomalies(response)
        # Fallback should pick up the bold keyword
        assert any(a["severity"] == "high" for a in anomalies)

    # --- Fallback keyword matching ---

    def test_detects_bold_severity_keywords(self) -> None:
        response = (
            "- Unit 101: **critical** — rent is zero.\n"
            "- Unit 202: **high** — lease expired.\n"
            "- Unit 303: **low** — missing sq ft.\n"
        )
        anomalies = AuditAgent._extract_anomalies(response)
        severities = [a["severity"] for a in anomalies]
        assert "critical" in severities
        assert "high" in severities
        assert "low" in severities

    def test_detects_severity_colon_pattern(self) -> None:
        response = "Finding 1 severity: medium — variance exceeds 15%.\n"
        anomalies = AuditAgent._extract_anomalies(response)
        assert len(anomalies) == 1
        assert anomalies[0]["severity"] == "medium"

    def test_returns_empty_list_when_no_anomalies(self) -> None:
        response = "Everything looks fine. No issues detected."
        anomalies = AuditAgent._extract_anomalies(response)
        assert anomalies == []

    def test_each_anomaly_has_description_key(self) -> None:
        response = "- **high** — something bad happened.\n"
        anomalies = AuditAgent._extract_anomalies(response)
        assert len(anomalies) == 1
        assert "description" in anomalies[0]
        assert anomalies[0]["description"] != ""

    def test_empty_response_returns_empty_list(self) -> None:
        assert AuditAgent._extract_anomalies("") == []


# ---------------------------------------------------------------------------
# AuditAgent.run() — mocked LangGraph
# ---------------------------------------------------------------------------


_MOCK_REPORT = """\
## Executive Summary
2 anomalies found.

## Detailed Findings
- Unit 101: **critical** — rent is zero.
- Unit 202: **medium** — lease end date in the past.

## Risk Assessment
Critical issue requires immediate attention.

## Recommended Actions
Fix rent for Unit 101.

```json
[
  {
    "severity": "critical",
    "document_type": "rent_roll",
    "affected": "Unit 101",
    "description": "Monthly rent is $0 — possible data error.",
    "recommended_action": "Verify rent amount and tenant status for Unit 101."
  },
  {
    "severity": "medium",
    "document_type": "rent_roll",
    "affected": "Unit 202",
    "description": "Lease end date is in the past but unit is still marked occupied.",
    "recommended_action": "Confirm lease renewal or update occupancy status."
  }
]
```
"""


class TestAuditAgentRun:
    @patch("agents.audit_agent.create_react_agent")
    def test_run_returns_audit_result(self, mock_create_agent: MagicMock) -> None:
        # Set up the mock agent to return a response with a final message
        mock_agent = MagicMock()
        final_msg = MagicMock()
        final_msg.content = _MOCK_REPORT
        mock_agent.invoke.return_value = {"messages": [final_msg]}
        mock_create_agent.return_value = mock_agent

        agent = AuditAgent(api_key="test-key")
        result = agent.run(
            rent_roll_content="unit,rent\nA1,1200",
            projections_content="",
            concessions_content="",
        )

        assert isinstance(result, AuditResult)
        assert result.raw_response == _MOCK_REPORT

    @patch("agents.audit_agent.create_react_agent")
    def test_run_extracts_anomalies(self, mock_create_agent: MagicMock) -> None:
        mock_agent = MagicMock()
        final_msg = MagicMock()
        final_msg.content = _MOCK_REPORT
        mock_agent.invoke.return_value = {"messages": [final_msg]}
        mock_create_agent.return_value = mock_agent

        agent = AuditAgent(api_key="test-key")
        result = agent.run(
            rent_roll_content="unit,rent\nA1,0",
            projections_content="",
            concessions_content="",
        )

        assert len(result.anomalies) == 2  # parsed from JSON block
        assert any(a["severity"] == "critical" for a in result.anomalies)
        # Richer fields should be present when JSON path is taken
        critical = next(a for a in result.anomalies if a["severity"] == "critical")
        assert critical["document_type"] == "rent_roll"
        assert critical["affected"] == "Unit 101"
        assert "recommended_action" in critical

    @patch("agents.audit_agent.create_react_agent")
    def test_run_extracts_summary(self, mock_create_agent: MagicMock) -> None:
        mock_agent = MagicMock()
        final_msg = MagicMock()
        final_msg.content = _MOCK_REPORT
        mock_agent.invoke.return_value = {"messages": [final_msg]}
        mock_create_agent.return_value = mock_agent

        agent = AuditAgent(api_key="test-key")
        result = agent.run("data", "", "")

        assert "2 anomalies found" in result.summary

    @patch("agents.audit_agent.create_react_agent")
    def test_run_with_all_empty_content(self, mock_create_agent: MagicMock) -> None:
        mock_agent = MagicMock()
        final_msg = MagicMock()
        final_msg.content = "No content provided."
        mock_agent.invoke.return_value = {"messages": [final_msg]}
        mock_create_agent.return_value = mock_agent

        agent = AuditAgent(api_key="test-key")
        result = agent.run("", "", "")

        assert isinstance(result, AuditResult)
        assert result.raw_response == "No content provided."

    @patch("agents.audit_agent.create_react_agent")
    def test_api_key_passed_to_llm(self, mock_create_agent: MagicMock) -> None:
        """Verify api_key is forwarded to ChatOpenAI without touching os.environ."""
        with patch("agents.audit_agent.ChatOpenAI") as mock_llm_cls:
            mock_llm_cls.return_value = MagicMock()
            mock_create_agent.return_value = MagicMock()

            AuditAgent(api_key="sk-test-12345")

            call_kwargs = mock_llm_cls.call_args.kwargs
            assert call_kwargs.get("api_key") == "sk-test-12345"
