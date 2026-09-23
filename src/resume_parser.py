"""Resume text extraction for PDF and DOCX files.

V1 supports exactly two input formats:

- PDF, read with PyMuPDF (``fitz``), using ``page.get_text("text", sort=True)``.
- DOCX, read with ``python-docx``.

OCR is explicitly out of scope for V1. If a PDF contains no extractable
text (e.g. it is a scanned image), ``extract_text`` raises a
``ResumeParsingError`` that says so clearly instead of silently returning
an empty string.

The heavy PDF dependency (``fitz``) is imported lazily, inside
``_extract_pdf_text``, so that importing this module — and parsing DOCX
files — does not require PyMuPDF to be installed.
"""

from __future__ import annotations

import re
from pathlib import Path

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx")


class ResumeParsingError(Exception):
    """Raised for any resume-parsing failure that the CLI should surface
    to the user as a clear, actionable message (missing file, unsupported
    format, image-only PDF, missing dependency, corrupted file, etc.)."""


def extract_text(path: Path | str) -> str:
    """Extract raw text from a resume file.

    Args:
        path: Path to a ``.pdf`` or ``.docx`` resume file.

    Returns:
        The extracted, but not yet cleaned, text of the resume.

    Raises:
        ResumeParsingError: if the file does not exist, has an unsupported
            extension, cannot be parsed, or yields no extractable text.
    """
    resume_path = Path(path)

    if not resume_path.exists():
        raise ResumeParsingError(f"Resume file not found: {resume_path}")

    if not resume_path.is_file():
        raise ResumeParsingError(f"Resume path is not a file: {resume_path}")

    extension = resume_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise ResumeParsingError(
            f"Unsupported resume format '{extension}'. "
            f"Supported formats in V1: {supported}. OCR/image-only PDFs "
            f"and other formats (e.g. .doc, .txt, .rtf) are not supported."
        )

    if extension == ".pdf":
        text = _extract_pdf_text(resume_path)
    else:  # ".docx"
        text = _extract_docx_text(resume_path)

    if not text or not text.strip():
        raise ResumeParsingError(
            f"No extractable text was found in '{resume_path.name}'. "
            "If this is a scanned/image-only PDF, note that OCR is not "
            "supported in V1 — please provide a text-based PDF or DOCX."
        )

    return text


def _extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF using PyMuPDF (fitz)."""
    try:
        import pymupdf as fitz  # PyMuPDF — imported lazily on purpose (see module docstring)
    except ImportError as exc:
        raise ResumeParsingError(
            "PyMuPDF (the 'pymupdf' module) is required to parse PDF resumes "
            "but is not installed. Install it with: pip install pymupdf"
        ) from exc

    try:
        document = fitz.open(path)
    except Exception as exc:  # PyMuPDF raises its own exception types
        raise ResumeParsingError(f"Failed to open PDF '{path.name}': {exc}") from exc

    try:
        page_texts = [page.get_text("text", sort=True) for page in document]
    except Exception as exc:
        raise ResumeParsingError(
            f"Failed to extract text from PDF '{path.name}': {exc}"
        ) from exc
    finally:
        document.close()

    return "\n".join(page_texts)


def _extract_docx_text(path: Path) -> str:
    """Extract text from a DOCX using python-docx.

    Paragraphs (including blank ones, which help section detection) and
    table cell text are both included, in document order.
    """
    try:
        import docx
    except ImportError as exc:
        raise ResumeParsingError(
            "python-docx is required to parse DOCX resumes but is not "
            "installed. Install it with: pip install python-docx"
        ) from exc

    try:
        document = docx.Document(str(path))
    except Exception as exc:
        raise ResumeParsingError(f"Failed to open DOCX '{path.name}': {exc}") from exc

    lines: list[str] = [paragraph.text for paragraph in document.paragraphs]

    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    lines.append(cell.text)

    return "\n".join(lines)


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving meaningful line breaks.

    - Normalizes Windows/Mac line endings to ``\n``.
    - Expands tabs to single spaces.
    - Strips trailing whitespace from each line.
    - Collapses runs of 3+ blank lines down to a single blank line.
    - Collapses runs of horizontal whitespace within a line to one space.
    - Strips leading/trailing blank lines from the whole document.
    """
    if not text:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\t", " ")

    lines = [re.sub(r"[ \u00a0]+", " ", line).strip() for line in normalized.split("\n")]

    cleaned_lines: list[str] = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run <= 1:
                cleaned_lines.append(line)
        else:
            blank_run = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip("\n")
