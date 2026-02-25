"""
Audit agent module.

Implements a LangGraph-based ReAct agent that audits rent roll, rent
projections, and concessions data for anomalies.  The agent is powered by an
OpenAI chat model and exposes a set of structured analysis tools that it can
call autonomously before producing a final report.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class AuditResult:
    """Structured output of an audit run."""

    anomalies: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    raw_response: str = ""


# ---------------------------------------------------------------------------
# LangChain tools available to the agent
# ---------------------------------------------------------------------------


@tool
def identify_rent_roll_anomalies(data_summary: str) -> str:
    """Analyse a rent roll data summary and identify anomalies.

    Checks for: missing required fields, units with zero or negative rent,
    duplicate unit IDs, expired leases still marked as occupied, and
    statistical outliers in rent amounts.

    Parameters
    ----------
    data_summary:
        The textual summary / raw content of the rent roll data.

    Returns
    -------
    str
        A JSON-formatted list of anomaly findings.
    """
    # This tool's docstring guides the LLM. The agent itself will reason over
    # data_summary and produce the analysis in its chain-of-thought; the tool
    # return value signals back to the orchestration layer.
    return json.dumps(
        {
            "tool": "identify_rent_roll_anomalies",
            "input_received": True,
            "instruction": (
                "Carefully examine the rent roll data for: "
                "1) missing or null values in critical fields (unit_id, tenant_name, monthly_rent, lease dates), "
                "2) units with zero, negative, or unrealistically low/high rent amounts, "
                "3) duplicate unit IDs, "
                "4) lease end dates in the past for units still marked occupied, "
                "5) statistical outliers (rent > 3 std devs from mean), "
                "6) units with no tenant but marked as occupied. "
                "Report each anomaly with: type, affected_unit/row, description, severity (low/medium/high/critical)."
            ),
        }
    )


@tool
def identify_projection_anomalies(data_summary: str) -> str:
    """Analyse rent projection data and identify anomalies.

    Checks for: large variances between projected and actual rent, missing
    periods, inconsistent growth rates, and negative projections.

    Parameters
    ----------
    data_summary:
        The textual summary / raw content of the rent projection data.

    Returns
    -------
    str
        A JSON-formatted list of anomaly findings.
    """
    return json.dumps(
        {
            "tool": "identify_projection_anomalies",
            "input_received": True,
            "instruction": (
                "Carefully examine the rent projection data for: "
                "1) large variance between projected and actual rent (>10% is notable, >25% is critical), "
                "2) missing periods in the projection timeline, "
                "3) inconsistent month-over-month growth rates, "
                "4) negative projected or actual rent values, "
                "5) periods where actual rent dramatically exceeds projections without explanation. "
                "Report each anomaly with: type, affected_period, description, severity."
            ),
        }
    )


@tool
def identify_concession_anomalies(data_summary: str) -> str:
    """Analyse concessions data and identify anomalies.

    Checks for: unapproved or missing approval records, unusually large
    concessions, duplicate concessions for the same unit, concessions
    exceeding monthly rent, and missing justification/reason fields.

    Parameters
    ----------
    data_summary:
        The textual summary / raw content of the concessions data.

    Returns
    -------
    str
        A JSON-formatted list of anomaly findings.
    """
    return json.dumps(
        {
            "tool": "identify_concession_anomalies",
            "input_received": True,
            "instruction": (
                "Carefully examine the concessions data for: "
                "1) concession amounts that exceed the unit's monthly rent, "
                "2) multiple concessions for the same unit in the same period, "
                "3) missing or blank 'approved_by' field (unapproved concessions), "
                "4) missing reason/justification for a concession, "
                "5) concession amounts that are statistical outliers (>3 std devs), "
                "6) concession types that are unusual or not in an approved list. "
                "Report each anomaly with: type, affected_unit/row, description, severity."
            ),
        }
    )


@tool
def generate_audit_report(findings_json: str) -> str:
    """Consolidate all anomaly findings into a final audit report.

    Parameters
    ----------
    findings_json:
        A JSON string containing all anomaly findings from prior tool calls.

    Returns
    -------
    str
        A structured markdown audit report.
    """
    return json.dumps(
        {
            "tool": "generate_audit_report",
            "input_received": True,
            "instruction": (
                "Using the findings provided, generate a comprehensive audit report in markdown. "
                "The report should include: "
                "1) Executive Summary with total anomaly counts by severity, "
                "2) Detailed Findings section grouped by document type (rent roll / projections / concessions), "
                "3) Risk Assessment highlighting the most critical issues, "
                "4) Recommended Actions for each finding. "
                "Format the output clearly so it can be rendered in a Streamlit markdown component."
            ),
        }
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert real-estate audit AI agent specialising in \
identifying anomalies in rent rolls, rent projections, and concessions data. \
Your goal is to perform a thorough, systematic audit of the provided documents.

When given document content, you MUST:
1. Call `identify_rent_roll_anomalies` if rent roll data is present.
2. Call `identify_projection_anomalies` if rent projection data is present.
3. Call `identify_concession_anomalies` if concessions data is present.
4. Call `generate_audit_report` to consolidate all findings into a final report.

For each anomaly you discover, classify it with a severity level:
- **critical**: Requires immediate action (e.g., financial fraud indicators, data integrity failures)
- **high**: Significant issue that needs prompt resolution
- **medium**: Notable issue that should be investigated
- **low**: Minor discrepancy or best-practice violation

Be thorough, precise, and professional. Format your final report in clear markdown."""


class AuditAgent:
    """LangGraph-powered audit agent.

    Parameters
    ----------
    model:
        OpenAI model name. Defaults to the ``OPENAI_MODEL`` environment
        variable, falling back to ``"gpt-4o"``.
    max_tokens:
        Maximum tokens for LLM responses. Defaults to the
        ``OPENAI_MAX_TOKENS`` environment variable, falling back to ``4096``.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        token_limit = max_tokens or int(os.getenv("OPENAI_MAX_TOKENS", "4096"))

        llm = ChatOpenAI(
            model=model_name,
            max_tokens=token_limit,
            temperature=0,
        )
        tools = [
            identify_rent_roll_anomalies,
            identify_projection_anomalies,
            identify_concession_anomalies,
            generate_audit_report,
        ]
        self._agent = create_react_agent(llm, tools)

    def run(
        self,
        rent_roll_content: str,
        projections_content: str,
        concessions_content: str,
    ) -> AuditResult:
        """Run the audit agent over the provided document contents.

        Parameters
        ----------
        rent_roll_content:
            Text extracted from the rent roll document(s).
        projections_content:
            Text extracted from the rent projections document(s).
        concessions_content:
            Text extracted from the concessions document(s).

        Returns
        -------
        AuditResult
            Structured audit result with anomalies and summary.
        """
        user_message = self._build_user_message(
            rent_roll_content, projections_content, concessions_content
        )
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        response = self._agent.invoke({"messages": messages})
        final_message = response["messages"][-1].content

        return AuditResult(
            raw_response=final_message,
            summary=self._extract_summary(final_message),
            anomalies=self._extract_anomalies(final_message),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_message(
        rent_roll: str, projections: str, concessions: str
    ) -> str:
        sections: list[str] = []
        if rent_roll.strip():
            sections.append(f"## RENT ROLL DATA\n\n{rent_roll}")
        if projections.strip():
            sections.append(f"## RENT PROJECTIONS DATA\n\n{projections}")
        if concessions.strip():
            sections.append(f"## CONCESSIONS DATA\n\n{concessions}")

        if not sections:
            return "No document content was provided. Please upload at least one document."

        return (
            "Please audit the following real-estate documents for anomalies.\n\n"
            + "\n\n---\n\n".join(sections)
        )

    @staticmethod
    def _extract_summary(response: str) -> str:
        """Pull the executive summary from the agent's markdown response."""
        lines = response.splitlines()
        in_summary = False
        summary_lines: list[str] = []
        for line in lines:
            if "executive summary" in line.lower():
                in_summary = True
                continue
            if in_summary:
                if line.startswith("##") and "executive summary" not in line.lower():
                    break
                summary_lines.append(line)
        return "\n".join(summary_lines).strip() or response[:500]

    @staticmethod
    def _extract_anomalies(response: str) -> list[dict[str, Any]]:
        """Extract structured anomaly records from the agent response."""
        anomalies: list[dict[str, Any]] = []
        severity_keywords = ["critical", "high", "medium", "low"]
        for line in response.splitlines():
            lower = line.lower()
            for severity in severity_keywords:
                if f"**{severity}**" in lower or f"severity: {severity}" in lower:
                    anomalies.append({"severity": severity, "description": line.strip(" -•*")})
                    break
        return anomalies
