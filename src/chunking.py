"""Step 3: text chunking with LangChain's RecursiveCharacterTextSplitter.

Each chunk keeps provenance metadata: source file, page, nearest section heading,
transcript timestamp and line range -> used later for source references.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .loaders import PageText

_TIME_RE = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b")


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 80 or s.endswith((".", "?", "!")):
        return False
    if s.startswith("#"):
        return True
    if re.fullmatch(r"(\d+[.)]\s+)?[A-Z][A-Z0-9 &/,\-()']{3,}:?", s):  # ALL CAPS
        return True
    if re.match(r"(agenda item|topic|item|section)\s*\d*\s*[:\-]", s, re.I):
        return True
    if re.fullmatch(r"\d+[.)]\s+[A-Z][\w &/,\-()']{2,60}", s):  # "1. Budget review"
        return True
    if re.fullmatch(r"[A-Za-z0-9 &/,\-()']{3,50}:", s) and len(s.split()) <= 6:  # "Action Items:"
        return True
    return False


def _heading_index(text: str) -> list[tuple[int, str]]:
    out, offset = [], 0
    for line in text.splitlines(keepends=True):
        if is_heading(line):
            out.append((offset, line.strip().lstrip("#").strip().rstrip(":").strip()))
        offset += len(line)
    return out


def chunk_document(
    source: str,
    pages: list[PageText],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks: list[Chunk] = []
    carried_heading = ""
    for page in pages:
        headings = _heading_index(page.text)
        for doc in splitter.create_documents([page.text]):
            start = doc.metadata.get("start_index", 0)
            section = carried_heading
            for off, title in headings:
                if off <= start:
                    section = title
                else:
                    break
            if not section and headings:
                section = headings[0][1]
            ts = _TIME_RE.search(doc.page_content)
            line_start = page.text.count("\n", 0, start) + 1
            idx = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{source}::{idx}",
                    text=doc.page_content,
                    metadata={
                        "source": source,
                        "page": page.page or 0,  # 0 = not applicable (TXT/DOCX)
                        "chunk_index": idx,
                        "section": section,
                        "timestamp": ts.group(1) if ts else "",
                        "line_start": line_start,
                        "line_end": line_start + doc.page_content.count("\n"),
                    },
                )
            )
        if headings:
            carried_heading = headings[-1][1]
    return chunks
