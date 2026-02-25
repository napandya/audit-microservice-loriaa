"""
Audit Microservice — Streamlit Application.

Allows users to upload CSV, Excel, PDF, and Word documents containing rent
roll, rent projections, and concessions data.  An AI audit agent (powered by
LangGraph + OpenAI) analyses the content and reports anomalies.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

from agents.audit_agent import AuditAgent, AuditResult
from parsers.document_parser import DocumentParser, ParsedDocument
from utils.data_processor import DataProcessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

load_dotenv()

_ACCEPTED_TYPES = ["csv", "xlsx", "xls", "pdf", "docx"]
_DOC_TYPE_OPTIONS = ["Rent Roll", "Rent Projections", "Concessions"]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Audit Agent — Rent Roll & Concessions",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _parse_uploaded_file(
    uploaded_file,
    parser: DocumentParser,
) -> Optional[ParsedDocument]:
    """Parse a Streamlit ``UploadedFile`` and return a ``ParsedDocument``."""
    try:
        raw_bytes = uploaded_file.read()
        return parser.parse(raw_bytes, uploaded_file.name)
    except (ValueError, IOError, KeyError, RuntimeError) as exc:
        logger.exception("Failed to parse file '%s'", uploaded_file.name)
        st.error(f"❌ Failed to parse **{uploaded_file.name}**: {exc}")
        return None


def _get_content_for_type(
    doc_type: str,
    parsed: ParsedDocument,
    processor: DataProcessor,
) -> str:
    """Return a processed text summary for the given document type."""
    df = parsed.dataframe
    if doc_type == "Rent Roll":
        structured = processor.summarise_rent_roll(df)
        return f"[File: {parsed.filename}]\n{structured}\n\nRaw text:\n{parsed.text_content}"
    if doc_type == "Rent Projections":
        structured = processor.summarise_rent_projections(df)
        return f"[File: {parsed.filename}]\n{structured}\n\nRaw text:\n{parsed.text_content}"
    if doc_type == "Concessions":
        structured = processor.summarise_concessions(df)
        return f"[File: {parsed.filename}]\n{structured}\n\nRaw text:\n{parsed.text_content}"
    return parsed.text_content


def _severity_badge(severity: str) -> str:
    """Return a coloured emoji badge for a severity level."""
    return {
        "critical": "🔴 **CRITICAL**",
        "high": "🟠 **HIGH**",
        "medium": "🟡 **MEDIUM**",
        "low": "🟢 **LOW**",
    }.get(severity.lower(), f"⚪ {severity.upper()}")


def _render_results(result: AuditResult) -> None:
    """Render audit results in the Streamlit UI."""
    st.success("✅ Audit complete!")

    tab_report, tab_anomalies, tab_raw = st.tabs(
        ["📋 Full Report", "⚠️ Anomaly List", "🔍 Raw Agent Output"]
    )

    with tab_report:
        st.markdown(result.raw_response)

    with tab_anomalies:
        if result.anomalies:
            st.write(f"**{len(result.anomalies)} anomalies detected**")
            for i, anomaly in enumerate(result.anomalies, start=1):
                badge = _severity_badge(anomaly.get("severity", "unknown"))
                doc_type = anomaly.get("document_type", "")
                affected = anomaly.get("affected", "")
                label_parts = [f"Anomaly #{i}", badge]
                if doc_type:
                    label_parts.append(f"· {doc_type.replace('_', ' ').title()}")
                if affected:
                    label_parts.append(f"· {affected}")
                with st.expander(" ".join(label_parts), expanded=i <= 5):
                    st.write(anomaly.get("description", "No description available."))
                    action = anomaly.get("recommended_action", "")
                    if action:
                        st.info(f"**Recommended action:** {action}")
        else:
            st.info("No structured anomaly records were extracted from the report. See the Full Report tab.")

    with tab_raw:
        st.text_area(
            "Raw agent response",
            value=result.raw_response,
            height=500,
            disabled=True,
        )


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------


def _render_sidebar() -> str:
    """Render the sidebar and return the user-provided OpenAI API key."""
    st.sidebar.title("⚙️ Configuration")

    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Your OpenAI API key. You can also set it via the OPENAI_API_KEY environment variable.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Supported File Types")
    st.sidebar.markdown(
        "- 📄 CSV (`.csv`)\n"
        "- 📊 Excel (`.xlsx`, `.xls`)\n"
        "- 📕 PDF (`.pdf`)\n"
        "- 📝 Word (`.docx`)"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown(
        "This application uses a **LangGraph** ReAct agent backed by **GPT-4o** "
        "to audit real-estate documents for anomalies across rent rolls, rent "
        "projections, and concessions."
    )
    return api_key


# ---------------------------------------------------------------------------
# Main upload / audit section
# ---------------------------------------------------------------------------


def _render_upload_section(
    parser: DocumentParser,
    processor: DataProcessor,
) -> tuple[str, str, str]:
    """Render the three document-upload panels and return processed text content."""
    st.markdown("## 📁 Upload Documents")
    st.info(
        "Upload one or more files for each document type. "
        "Accepted formats: CSV, Excel (.xlsx/.xls), PDF, Word (.docx)."
    )

    col1, col2, col3 = st.columns(3)

    rent_roll_content = ""
    projections_content = ""
    concessions_content = ""

    with col1:
        st.markdown("### 🏢 Rent Roll")
        rent_roll_files = st.file_uploader(
            "Upload rent roll file(s)",
            type=_ACCEPTED_TYPES,
            accept_multiple_files=True,
            key="rent_roll_uploader",
        )
        if rent_roll_files:
            parts: list[str] = []
            for uf in rent_roll_files:
                parsed = _parse_uploaded_file(uf, parser)
                if parsed:
                    st.success(f"✔ Parsed: **{parsed.filename}** ({parsed.file_type})")
                    if parsed.dataframe is not None:
                        with st.expander(f"Preview — {parsed.filename}"):
                            st.dataframe(parsed.dataframe.head(10), use_container_width=True)
                    parts.append(_get_content_for_type("Rent Roll", parsed, processor))
            rent_roll_content = "\n\n".join(parts)

    with col2:
        st.markdown("### 📈 Rent Projections")
        projections_files = st.file_uploader(
            "Upload rent projection file(s)",
            type=_ACCEPTED_TYPES,
            accept_multiple_files=True,
            key="projections_uploader",
        )
        if projections_files:
            parts = []
            for uf in projections_files:
                parsed = _parse_uploaded_file(uf, parser)
                if parsed:
                    st.success(f"✔ Parsed: **{parsed.filename}** ({parsed.file_type})")
                    if parsed.dataframe is not None:
                        with st.expander(f"Preview — {parsed.filename}"):
                            st.dataframe(parsed.dataframe.head(10), use_container_width=True)
                    parts.append(_get_content_for_type("Rent Projections", parsed, processor))
            projections_content = "\n\n".join(parts)

    with col3:
        st.markdown("### 🎁 Concessions")
        concessions_files = st.file_uploader(
            "Upload concessions file(s)",
            type=_ACCEPTED_TYPES,
            accept_multiple_files=True,
            key="concessions_uploader",
        )
        if concessions_files:
            parts = []
            for uf in concessions_files:
                parsed = _parse_uploaded_file(uf, parser)
                if parsed:
                    st.success(f"✔ Parsed: **{parsed.filename}** ({parsed.file_type})")
                    if parsed.dataframe is not None:
                        with st.expander(f"Preview — {parsed.filename}"):
                            st.dataframe(parsed.dataframe.head(10), use_container_width=True)
                    parts.append(_get_content_for_type("Concessions", parsed, processor))
            concessions_content = "\n\n".join(parts)

    return rent_roll_content, projections_content, concessions_content


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main Streamlit application entry point."""
    st.title("🔍 AI Audit Agent — Rent Roll, Projections & Concessions")
    st.markdown(
        "Upload your real-estate documents and let the AI agent identify anomalies, "
        "discrepancies, and compliance issues automatically."
    )

    api_key = _render_sidebar()

    parser = DocumentParser()
    processor = DataProcessor()

    rent_roll_content, projections_content, concessions_content = _render_upload_section(
        parser, processor
    )

    st.markdown("---")
    st.markdown("## 🚀 Run Audit")

    has_content = any([rent_roll_content, projections_content, concessions_content])
    audit_disabled = not (api_key and has_content)

    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to enable the audit.")
    if not has_content:
        st.warning("⚠️ Please upload at least one document above before running the audit.")

    if st.button(
        "🔍 Run AI Audit",
        disabled=audit_disabled,
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("🤖 AI agent is analysing your documents… this may take a moment."):
            try:
                # Pass the API key directly to the agent to avoid storing it
                # in os.environ where it could be read by other code.
                agent = AuditAgent(api_key=api_key)
                result = agent.run(
                    rent_roll_content=rent_roll_content,
                    projections_content=projections_content,
                    concessions_content=concessions_content,
                )
                st.session_state["audit_result"] = result
            except (ValueError, RuntimeError, ConnectionError) as exc:
                logger.exception("Audit run failed")
                st.error(f"❌ Audit failed: {exc}")
                st.session_state.pop("audit_result", None)

    # Persist results across re-runs (e.g. tab switching)
    if "audit_result" in st.session_state:
        st.markdown("---")
        st.markdown("## 📊 Audit Results")
        _render_results(st.session_state["audit_result"])


if __name__ == "__main__":
    main()
