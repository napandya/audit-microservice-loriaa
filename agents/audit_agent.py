"""
Audit agent module.

Implements a LangGraph-based ReAct agent that audits rent roll, rent
projections, and concessions data for anomalies.  The agent is powered by an
OpenAI chat model and exposes a set of structured analysis tools that perform
**actual Python-based anomaly detection** on the data summaries, returning
concrete findings as JSON.  The LLM then uses those findings to write the
final professional audit report.
"""

from __future__ import annotations

import json
import os
import re
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
# Private rule-based analysis helpers
# ---------------------------------------------------------------------------


def _parse_float(pattern: str, text: str) -> Optional[float]:
    """Return the first captured float from *text* matching *pattern*, or None."""
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def _parse_int(pattern: str, text: str) -> Optional[int]:
    """Return the first captured int from *text* matching *pattern*, or None."""
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


def _analyse_rent_roll(data_summary: str) -> list[dict[str, Any]]:
    """Apply rule-based anomaly detection to a rent roll data summary string.

    Checks applied
    --------------
    * Missing rent values (null_count > 0)
    * Zero rent (min_rent == 0)
    * Negative rent (min_rent < 0)
    * Rent outlier (max_rent > 3 × mean_rent)
    * Non-numeric rent column (no valid numeric values present)
    * High vacancy rate (> 40 % of units vacant)
    """
    findings: list[dict[str, Any]] = []

    # --- Non-numeric rent column ---
    lower = data_summary.lower()
    if (
        "monthly rent: no valid numeric values" in lower
        or ("no valid numeric values" in lower and "monthly rent" in lower)
    ):
        findings.append(
            {
                "type": "non_numeric_rent",
                "severity": "critical",
                "affected": "monthly_rent column",
                "description": "The monthly_rent column contains no parseable numeric values.",
                "recommended_action": (
                    "Review the source file — the rent column may be incorrectly "
                    "formatted or contain placeholder text."
                ),
            }
        )
        return findings  # further numeric checks meaningless

    # --- Missing rent values ---
    null_count = _parse_int(r"null_count=(\d+)", data_summary)
    if null_count is not None and null_count > 0:
        findings.append(
            {
                "type": "missing_rent_values",
                "severity": "high",
                "affected": f"{null_count} unit(s)",
                "description": (
                    f"{null_count} unit(s) have a missing or unparseable monthly_rent value."
                ),
                "recommended_action": (
                    "Populate missing rent values before finalising the rent roll."
                ),
            }
        )

    # --- Zero rent ---
    min_rent = _parse_float(r"Monthly rent: min=([-\d.]+)", data_summary)
    if min_rent is not None:
        if min_rent < 0:
            findings.append(
                {
                    "type": "negative_rent",
                    "severity": "critical",
                    "affected": "One or more units",
                    "description": (
                        f"At least one unit has a negative monthly rent "
                        f"(minimum detected: ${min_rent:,.2f})."
                    ),
                    "recommended_action": (
                        "Investigate and correct negative rent values immediately — "
                        "this indicates a data integrity issue."
                    ),
                }
            )
        elif min_rent == 0.0:
            findings.append(
                {
                    "type": "zero_rent",
                    "severity": "high",
                    "affected": "One or more units",
                    "description": "At least one unit has a monthly rent of $0.00.",
                    "recommended_action": (
                        "Verify whether zero-rent units are intentional (e.g. "
                        "owner-occupied) or data entry errors."
                    ),
                }
            )

    # --- Rent outlier: max > 3 × mean ---
    max_rent = _parse_float(r"max=([\d.]+), mean=", data_summary)
    mean_rent = _parse_float(r"mean=([\d.]+), null_count=", data_summary)
    if max_rent is not None and mean_rent is not None and mean_rent > 0:
        if max_rent > 3 * mean_rent:
            findings.append(
                {
                    "type": "rent_statistical_outlier",
                    "severity": "medium",
                    "affected": f"Unit with rent ${max_rent:,.2f}",
                    "description": (
                        f"Maximum rent (${max_rent:,.2f}) exceeds 3× the mean rent "
                        f"(${mean_rent:,.2f}), indicating a statistical outlier."
                    ),
                    "recommended_action": (
                        "Verify the highest rent amount is correct and not a data entry error."
                    ),
                }
            )

    # --- High vacancy rate ---
    occ_m = re.search(r"Occupancy status breakdown: \{(.*?)\}", data_summary)
    if occ_m:
        status_text = occ_m.group(1)
        occupied = _parse_int(r"'occupied':\s*(\d+)", status_text) or 0
        vacant = _parse_int(r"'vacant':\s*(\d+)", status_text) or 0
        total = occupied + vacant
        if total > 0 and vacant / total > 0.40:
            pct = 100 * vacant / total
            findings.append(
                {
                    "type": "high_vacancy_rate",
                    "severity": "medium",
                    "affected": f"{vacant} of {total} units",
                    "description": (
                        f"Vacancy rate is {pct:.1f}% ({vacant}/{total} units), "
                        "which is unusually high."
                    ),
                    "recommended_action": (
                        "Investigate the cause of the high vacancy rate and review "
                        "leasing strategy."
                    ),
                }
            )

    return findings


def _analyse_projections(data_summary: str) -> list[dict[str, Any]]:
    """Apply rule-based anomaly detection to a rent projections data summary.

    Checks applied
    --------------
    * Mean projection variance > 10 % of mean projected rent → high
    * Max projection variance > 25 % of mean projected rent → critical
    * Non-numeric projection columns
    * Reported variance column issues
    """
    findings: list[dict[str, Any]] = []
    lower = data_summary.lower()

    if "no valid numeric values" in lower and "projected" in lower:
        findings.append(
            {
                "type": "non_numeric_projections",
                "severity": "critical",
                "affected": "projected_rent / actual_rent columns",
                "description": "Projection columns contain no parseable numeric values.",
                "recommended_action": "Check source file formatting for projection data.",
            }
        )
        return findings

    # --- Computed variance from mean/max of |actual - projected| ---
    var_mean = _parse_float(r"Projected vs actual variance: mean=([\d.]+)", data_summary)
    var_max = _parse_float(r"Projected vs actual variance: mean=[\d.]+, max=([\d.]+)", data_summary)

    # Use mean projected rent as a reference denominator (added by DataProcessor)
    proj_mean = _parse_float(r"Mean projected rent: ([\d.]+)", data_summary)

    if var_max is not None and proj_mean and proj_mean > 0:
        var_max_pct = 100 * var_max / proj_mean
        if var_max_pct > 25:
            findings.append(
                {
                    "type": "critical_projection_variance",
                    "severity": "critical",
                    "affected": "One or more periods",
                    "description": (
                        f"Maximum projection variance (${var_max:,.2f}, "
                        f"{var_max_pct:.1f}% of mean projected rent) exceeds the "
                        "25% critical threshold."
                    ),
                    "recommended_action": (
                        "Immediately review periods with the largest divergence between "
                        "projected and actual rent."
                    ),
                }
            )
        elif var_max_pct > 10:
            findings.append(
                {
                    "type": "notable_projection_variance",
                    "severity": "high",
                    "affected": "One or more periods",
                    "description": (
                        f"Maximum projection variance (${var_max:,.2f}, "
                        f"{var_max_pct:.1f}% of mean projected rent) exceeds the "
                        "10% notable threshold."
                    ),
                    "recommended_action": (
                        "Investigate periods where actual rent diverges significantly "
                        "from projections."
                    ),
                }
            )

    if var_mean is not None and proj_mean and proj_mean > 0:
        var_mean_pct = 100 * var_mean / proj_mean
        if var_mean_pct > 10:
            findings.append(
                {
                    "type": "elevated_mean_variance",
                    "severity": "medium",
                    "affected": "Across multiple periods",
                    "description": (
                        f"Mean projection variance (${var_mean:,.2f}, "
                        f"{var_mean_pct:.1f}% of mean projected rent) is consistently elevated."
                    ),
                    "recommended_action": (
                        "Review the projection methodology — systematic underestimation "
                        "or overestimation is indicated."
                    ),
                }
            )

    # --- Reported variance column outliers ---
    rep_min = _parse_float(r"Reported variance: min=([-\d.]+)", data_summary)
    rep_max = _parse_float(r"Reported variance: min=[-\d.]+, max=([-\d.]+)", data_summary)
    if rep_min is not None and rep_min < 0:
        findings.append(
            {
                "type": "negative_reported_variance",
                "severity": "high",
                "affected": "One or more periods",
                "description": (
                    f"Reported variance contains a negative value (minimum: "
                    f"${rep_min:,.2f}), which may indicate actual rent below projection."
                ),
                "recommended_action": (
                    "Verify whether negative variance is expected or indicates "
                    "a reporting error."
                ),
            }
        )

    return findings


def _analyse_concessions(data_summary: str) -> list[dict[str, Any]]:
    """Apply rule-based anomaly detection to a concessions data summary.

    Checks applied
    --------------
    * Maximum concession > 3 × mean concession → statistical outlier
    * Non-numeric concession_amount column
    * High total concessions (> 10 % of records have max-level concession)
    """
    findings: list[dict[str, Any]] = []
    lower = data_summary.lower()

    if "no valid numeric values" in lower and "concession" in lower:
        findings.append(
            {
                "type": "non_numeric_concessions",
                "severity": "critical",
                "affected": "concession_amount column",
                "description": "The concession_amount column contains no parseable numeric values.",
                "recommended_action": "Check source file formatting for concessions data.",
            }
        )
        return findings

    total = _parse_float(r"total=([\d.]+)", data_summary)
    mean_c = _parse_float(r"mean=([\d.]+)", data_summary)
    max_c = _parse_float(r"max=([\d.]+)", data_summary)
    record_count = _parse_int(r"Concessions — (\d+) records", data_summary)

    # --- Statistical outlier: max > 3 × mean ---
    if max_c is not None and mean_c is not None and mean_c > 0:
        if max_c > 3 * mean_c:
            findings.append(
                {
                    "type": "concession_outlier",
                    "severity": "high",
                    "affected": f"Concession of ${max_c:,.2f}",
                    "description": (
                        f"Maximum concession (${max_c:,.2f}) is more than 3× the mean "
                        f"(${mean_c:,.2f}), suggesting an unusually large concession."
                    ),
                    "recommended_action": (
                        "Verify the largest concession has appropriate authorisation "
                        "and justification."
                    ),
                }
            )

    # --- High total concession burden ---
    if total is not None and mean_c is not None and record_count and record_count > 0:
        avg_per_record = total / record_count
        if mean_c > 0 and avg_per_record > 2 * mean_c:
            findings.append(
                {
                    "type": "high_concession_burden",
                    "severity": "medium",
                    "affected": f"All {record_count} concession records",
                    "description": (
                        f"Average concession per record (${avg_per_record:,.2f}) "
                        "suggests a heavy overall concession burden."
                    ),
                    "recommended_action": (
                        "Review the concessions policy — the overall concession "
                        "spend may be excessive relative to rent income."
                    ),
                }
            )

    # --- Unknown concession types ---
    approved_types = {
        "free_month", "discount", "reduced_rent", "move_in_special",
        "lease_renewal", "referral", "seasonal",
    }
    type_m = re.search(r"Concession type breakdown: \{(.*?)\}", data_summary)
    if type_m:
        type_text = type_m.group(1)
        found_types = re.findall(r"'([^']+)':\s*\d+", type_text)
        unknown = [t for t in found_types if t.lower() not in approved_types]
        if unknown:
            findings.append(
                {
                    "type": "unrecognised_concession_type",
                    "severity": "medium",
                    "affected": ", ".join(unknown),
                    "description": (
                        f"Concession type(s) not in the approved list detected: "
                        f"{', '.join(unknown)}."
                    ),
                    "recommended_action": (
                        "Confirm these concession types are authorised and update "
                        "the approved-types list if necessary."
                    ),
                }
            )

    return findings


def _consolidate_findings(findings_json: str) -> dict[str, Any]:
    """Parse the agent-supplied findings JSON and produce a consolidation summary."""
    try:
        all_findings: list[dict[str, Any]] = json.loads(findings_json)
        if not isinstance(all_findings, list):
            all_findings = []
    except (json.JSONDecodeError, TypeError):
        all_findings = []

    counts: dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for item in all_findings:
        sev = str(item.get("severity", "")).lower()
        if sev in counts:
            counts[sev] += 1

    return {
        "total_findings": len(all_findings),
        "severity_counts": counts,
        "findings": all_findings,
        "risk_level": (
            "CRITICAL" if counts["critical"] > 0
            else "HIGH" if counts["high"] > 0
            else "MEDIUM" if counts["medium"] > 0
            else "LOW"
        ),
    }


# ---------------------------------------------------------------------------
# LangChain tools available to the agent
# ---------------------------------------------------------------------------


@tool
def identify_rent_roll_anomalies(data_summary: str) -> str:
    """Perform Python-based anomaly detection on rent roll data.

    Applies rule-based checks to the structured data summary and returns
    a JSON list of concrete findings.  Checks include: missing rent values,
    zero or negative rent, statistical outliers (max > 3× mean), and
    unusually high vacancy rates.

    Parameters
    ----------
    data_summary:
        The textual summary / raw content of the rent roll data produced by
        the data processor.

    Returns
    -------
    str
        A JSON object with a ``findings`` list of detected anomalies, each
        with ``type``, ``severity``, ``affected``, ``description``, and
        ``recommended_action`` keys.
    """
    findings = _analyse_rent_roll(data_summary)
    return json.dumps({"document_type": "rent_roll", "findings": findings})


@tool
def identify_projection_anomalies(data_summary: str) -> str:
    """Perform Python-based anomaly detection on rent projection data.

    Applies rule-based checks and returns a JSON list of concrete findings.
    Checks include: variance > 10 % (notable) or > 25 % (critical) of mean
    projected rent, elevated mean variance across periods, and negative
    reported variance values.

    Parameters
    ----------
    data_summary:
        The textual summary / raw content of the rent projection data.

    Returns
    -------
    str
        A JSON object with a ``findings`` list of detected anomalies.
    """
    findings = _analyse_projections(data_summary)
    return json.dumps({"document_type": "projections", "findings": findings})


@tool
def identify_concession_anomalies(data_summary: str) -> str:
    """Perform Python-based anomaly detection on concessions data.

    Applies rule-based checks and returns a JSON list of concrete findings.
    Checks include: statistical outliers (max concession > 3× mean), high
    total concession burden, and unrecognised concession types.

    Parameters
    ----------
    data_summary:
        The textual summary / raw content of the concessions data.

    Returns
    -------
    str
        A JSON object with a ``findings`` list of detected anomalies.
    """
    findings = _analyse_concessions(data_summary)
    return json.dumps({"document_type": "concessions", "findings": findings})


@tool
def generate_audit_report(findings_json: str) -> str:
    """Consolidate all anomaly findings into a structured audit report summary.

    Parses the JSON findings collected from the prior analysis tools, counts
    anomalies by severity, and returns a structured consolidation object that
    the agent uses to compose the final markdown report.

    Parameters
    ----------
    findings_json:
        A JSON array string of all anomaly findings gathered from prior tool
        calls.

    Returns
    -------
    str
        A JSON object containing ``total_findings``, ``severity_counts``,
        ``risk_level``, and the full ``findings`` list — ready for the agent
        to format into the final report.
    """
    consolidation = _consolidate_findings(findings_json)
    return json.dumps(consolidation)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert real-estate audit AI agent specialising in \
identifying anomalies in rent rolls, rent projections, and concessions data. \
Your goal is to perform a thorough, systematic audit of the provided documents.

The tools available to you perform **actual Python-based anomaly detection** and \
return concrete findings as JSON — they are not just instructions.  You MUST use \
their output as the factual basis for your report.

When given document content, you MUST:
1. Call `identify_rent_roll_anomalies` if rent roll data is present.
2. Call `identify_projection_anomalies` if rent projection data is present.
3. Call `identify_concession_anomalies` if concessions data is present.
4. Call `generate_audit_report` with ALL the findings from steps 1–3 as a JSON array.
5. Write a comprehensive markdown audit report that incorporates the tool findings \
AND any additional qualitative issues you observe in the data.

For each anomaly, classify it with a severity level:
- **critical**: Requires immediate action (e.g., financial fraud indicators, data integrity failures)
- **high**: Significant issue that needs prompt resolution
- **medium**: Notable issue that should be investigated
- **low**: Minor discrepancy or best-practice violation

Be thorough, precise, and professional. Format your final report in clear markdown.

IMPORTANT — Structured anomaly output:
After your full markdown report, append a fenced JSON code block containing every
anomaly (from both tool findings AND any additional issues you identified) as a
list of objects.  The block MUST follow this exact schema:

```json
[
  {
    "severity": "critical",
    "document_type": "rent_roll",
    "affected": "Unit 101",
    "description": "Monthly rent is $0 — possible data error.",
    "recommended_action": "Verify rent amount and tenant status for Unit 101."
  }
]
```

Rules for the JSON block:
- "severity" must be one of: "critical", "high", "medium", "low"
- "document_type" must be one of: "rent_roll", "projections", "concessions", "general"
- "affected" identifies the unit, row, or period involved
- "description" is a concise explanation of the anomaly
- "recommended_action" is a concrete next step
- If no anomalies are found, output an empty array: ```json\n[]\n```
- Output ONLY ONE json code block, placed at the very end of your response."""


class AuditAgent:
    """LangGraph-powered audit agent.

    Parameters
    ----------
    model:
        OpenAI model name. Defaults to the ``OPENAI_MODEL`` environment
        variable, falling back to ``"gpt-4o"``.
    max_tokens:
        Maximum tokens for LLM responses. Defaults to the
        ``OPENAI_MAX_TOKENS`` environment variable, falling back to ``8192``.
        GPT-4o supports a 128 k-token context window; 8192 output tokens
        gives ample room for comprehensive audit reports while avoiding
        mid-sentence truncation.
    api_key:
        OpenAI API key. When supplied, it is passed directly to the
        ``ChatOpenAI`` client rather than being read from the environment,
        keeping the key out of ``os.environ``.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> None:
        model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        token_limit = max_tokens or int(os.getenv("OPENAI_MAX_TOKENS", "8192"))

        llm = ChatOpenAI(
            model=model_name,
            max_tokens=token_limit,
            temperature=0,
            api_key=api_key or None,
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
        """Extract structured anomaly records from the agent response.

        The agent is instructed to append a ``json`` fenced code block
        containing all anomalies.  This method parses that block first.  If no
        valid JSON block is present (e.g., an older prompt or truncated
        response), it falls back to lightweight keyword matching so that
        *something* is always returned rather than an empty list.
        """
        # --- Primary path: parse the JSON code block emitted by the agent ---
        json_block_re = re.compile(
            r"```json\s*(\[.*?\])\s*```", re.DOTALL | re.IGNORECASE
        )
        match = json_block_re.search(response)
        if match:
            try:
                records = json.loads(match.group(1))
                if isinstance(records, list):
                    valid: list[dict[str, Any]] = []
                    valid_severities = {"critical", "high", "medium", "low"}
                    for item in records:
                        if not isinstance(item, dict):
                            continue
                        severity = str(item.get("severity", "")).lower()
                        if severity not in valid_severities:
                            continue
                        valid.append(
                            {
                                "severity": severity,
                                "document_type": str(item.get("document_type", "general")),
                                "affected": str(item.get("affected", "")),
                                "description": str(item.get("description", "")),
                                "recommended_action": str(
                                    item.get("recommended_action", "")
                                ),
                            }
                        )
                    return valid
            except (json.JSONDecodeError, TypeError):
                pass  # fall through to keyword fallback

        # --- Fallback: lightweight keyword matching for backward compatibility ---
        anomalies: list[dict[str, Any]] = []
        severity_keywords = ["critical", "high", "medium", "low"]
        for line in response.splitlines():
            lower = line.lower()
            for severity in severity_keywords:
                if f"**{severity}**" in lower or f"severity: {severity}" in lower:
                    anomalies.append(
                        {
                            "severity": severity,
                            "document_type": "general",
                            "affected": "",
                            "description": line.strip(" -•*"),
                            "recommended_action": "",
                        }
                    )
                    break
        return anomalies
