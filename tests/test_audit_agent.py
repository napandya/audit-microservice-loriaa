"""
Tests for agents.audit_agent.

Validates the AuditAgent helper methods (_extract_summary, _extract_anomalies)
and the AuditAgent.run() flow using a mocked LangGraph agent so no real OpenAI
API calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.audit_agent import AuditAgent, AuditResult


# ---------------------------------------------------------------------------
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

        assert len(result.anomalies) >= 1
        assert any(a["severity"] == "critical" for a in result.anomalies)

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
