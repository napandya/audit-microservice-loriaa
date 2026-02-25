"""
Tests for parsers.document_parser.

These tests exercise the DocumentParser class using in-memory file bytes so
that no external dependencies (files on disk, network, LLM) are required.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from parsers.document_parser import DocumentParser, ParsedDocument


@pytest.fixture()
def parser() -> DocumentParser:
    return DocumentParser()


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


class TestCSVParser:
    def test_parse_returns_parsed_document(self, parser: DocumentParser) -> None:
        csv_content = b"unit_id,tenant_name,monthly_rent\n101,Alice,1500\n102,Bob,1600\n"
        result = parser.parse(csv_content, "rent_roll.csv")

        assert isinstance(result, ParsedDocument)
        assert result.filename == "rent_roll.csv"
        assert result.file_type == "csv"

    def test_parse_csv_dataframe_shape(self, parser: DocumentParser) -> None:
        csv_content = b"unit_id,tenant_name,monthly_rent\n101,Alice,1500\n102,Bob,1600\n"
        result = parser.parse(csv_content, "rent_roll.csv")

        assert result.dataframe is not None
        assert result.dataframe.shape == (2, 3)
        assert list(result.dataframe.columns) == ["unit_id", "tenant_name", "monthly_rent"]

    def test_parse_csv_text_content(self, parser: DocumentParser) -> None:
        csv_content = b"a,b\n1,2\n"
        result = parser.parse(csv_content, "data.csv")

        assert "a" in result.text_content
        assert "b" in result.text_content


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------


class TestExcelParser:
    @staticmethod
    def _make_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
        return buf.getvalue()

    def test_parse_excel_returns_parsed_document(self, parser: DocumentParser) -> None:
        df = pd.DataFrame({"unit": ["A1", "A2"], "rent": [1200, 1300]})
        xl_bytes = self._make_excel_bytes(df)
        result = parser.parse(xl_bytes, "data.xlsx")

        assert isinstance(result, ParsedDocument)
        assert result.file_type == "excel"

    def test_parse_excel_dataframe_not_empty(self, parser: DocumentParser) -> None:
        df = pd.DataFrame({"unit": ["A1"], "rent": [1200]})
        xl_bytes = self._make_excel_bytes(df)
        result = parser.parse(xl_bytes, "data.xlsx")

        assert result.dataframe is not None
        assert not result.dataframe.empty

    def test_parse_excel_sheet_name_in_text(self, parser: DocumentParser) -> None:
        df = pd.DataFrame({"col": [1]})
        xl_bytes = self._make_excel_bytes(df, sheet_name="RentRoll")
        result = parser.parse(xl_bytes, "multi.xlsx")

        assert "RentRoll" in result.text_content

    def test_parse_xls_extension_accepted(self, parser: DocumentParser) -> None:
        # .xls extension should map to "excel" type even though we create .xlsx bytes
        df = pd.DataFrame({"x": [1]})
        xl_bytes = self._make_excel_bytes(df)
        result = parser.parse(xl_bytes, "legacy.xls")

        assert result.file_type == "excel"


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


class TestPDFParser:
    @staticmethod
    def _make_simple_pdf_bytes(text: str) -> bytes:
        """Create a minimal valid PDF containing *text* without external deps."""
        # Build a minimal hand-crafted PDF
        stream_content = f"BT /F1 12 Tf 100 700 Td ({text}) Tj ET"
        stream_bytes = stream_content.encode("latin-1")
        stream_len = len(stream_bytes)

        objects: list[str] = []
        objects.append(
            "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj"
        )
        objects.append(
            "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj"
        )
        objects.append(
            "3 0 obj\n<< /Type /Page /Parent 2 0 R "
            "/MediaBox [0 0 612 792] "
            "/Contents 4 0 R "
            "/Resources << /Font << /F1 5 0 R >> >> >>\nendobj"
        )
        objects.append(
            f"4 0 obj\n<< /Length {stream_len} >>\nstream\n"
            + stream_content
            + "\nendstream\nendobj"
        )
        objects.append(
            "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj"
        )

        body = "%PDF-1.4\n"
        offsets: list[int] = []
        for obj in objects:
            offsets.append(len(body))
            body += obj + "\n"

        xref_offset = len(body)
        body += f"xref\n0 {len(objects) + 1}\n"
        body += "0000000000 65535 f \n"
        for off in offsets:
            body += f"{off:010d} 00000 n \n"

        body += (
            "trailer\n"
            f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            "startxref\n"
            f"{xref_offset}\n"
            "%%EOF"
        )
        return body.encode("latin-1")

    def test_parse_pdf_returns_parsed_document(self, parser: DocumentParser) -> None:
        pdf_bytes = self._make_simple_pdf_bytes("Hello Audit")
        result = parser.parse(pdf_bytes, "report.pdf")

        assert isinstance(result, ParsedDocument)
        assert result.file_type == "pdf"
        assert result.dataframe is None

    def test_parse_pdf_page_markers_present(self, parser: DocumentParser) -> None:
        pdf_bytes = self._make_simple_pdf_bytes("Test content")
        result = parser.parse(pdf_bytes, "report.pdf")

        assert "Page 1" in result.text_content


# ---------------------------------------------------------------------------
# Word
# ---------------------------------------------------------------------------


class TestWordParser:
    @staticmethod
    def _make_docx_bytes(paragraphs: list[str]) -> bytes:
        from docx import Document as DocxDocument

        doc = DocxDocument()
        for para in paragraphs:
            doc.add_paragraph(para)
        buf = io.BytesIO()
        doc.save(buf)
        return buf.getvalue()

    def test_parse_word_returns_parsed_document(self, parser: DocumentParser) -> None:
        docx_bytes = self._make_docx_bytes(["Unit A1: $1,200/month", "Tenant: Alice"])
        result = parser.parse(docx_bytes, "lease.docx")

        assert isinstance(result, ParsedDocument)
        assert result.file_type == "word"
        assert result.dataframe is None

    def test_parse_word_text_content(self, parser: DocumentParser) -> None:
        paragraphs = ["Hello", "World"]
        docx_bytes = self._make_docx_bytes(paragraphs)
        result = parser.parse(docx_bytes, "notes.docx")

        assert "Hello" in result.text_content
        assert "World" in result.text_content


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestUnsupportedFormat:
    def test_unsupported_extension_raises_value_error(self, parser: DocumentParser) -> None:
        with pytest.raises(ValueError, match="Unsupported file type"):
            parser.parse(b"data", "file.txt")

    def test_error_message_lists_supported_types(self, parser: DocumentParser) -> None:
        with pytest.raises(ValueError, match=r"\.csv"):
            parser.parse(b"data", "file.unknown")
