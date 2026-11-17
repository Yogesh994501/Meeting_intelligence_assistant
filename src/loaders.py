"""Step 1-2: document loading and text extraction (PDF / TXT / DOCX)."""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PageText:
    text: str
    page: Optional[int]  # 1-based page number, None when the format has no pages


class UnsupportedFileType(ValueError):
    pass


def load_document(filename: str, data: bytes) -> list[PageText]:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _load_pdf(data)
    if ext in (".txt", ".md"):
        return _load_txt(data)
    if ext == ".docx":
        return _load_docx(data)
    raise UnsupportedFileType(f"Unsupported file type: {ext or 'unknown'}")


def _load_pdf(data: bytes) -> list[PageText]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(PageText(text=text, page=i))
    if not pages:
        raise ValueError("No extractable text found (scanned PDF? try OCR first).")
    return pages


def _load_txt(data: bytes) -> list[PageText]:
    text = None
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    text = (text or "").strip()
    if not text:
        raise ValueError("The text file is empty.")
    return [PageText(text=text, page=None)]


def _load_docx(data: bytes) -> list[PageText]:
    from docx import Document

    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    text = "\n".join(parts).strip()
    if not text:
        raise ValueError("The Word document has no text.")
    return [PageText(text=text, page=None)]
