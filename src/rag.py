"""Steps 7-10: retrieval, answer generation, source references, not-found handling,
plus meeting-specific features (summaries, action items, deadlines, decisions)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config, prompts
from .chunking import chunk_document
from .llm import LLMClient
from .loaders import load_document
from .vector_store import Hit, VectorStore

NOT_FOUND_MESSAGE = (
    "I couldn't find that in the uploaded meeting documents. "
    "Try rephrasing, selecting more meetings, or lowering the minimum-similarity setting."
)

STATUS_MAP = {"pending": "Pending", "in progress": "In progress", "in-progress": "In progress",
              "done": "Done", "completed": "Done", "complete": "Done", "blocked": "Blocked"}


# --------------------------------------------------------------------------- helpers
@dataclass
class Answer:
    text: str
    found: bool
    sources: list[dict] = field(default_factory=list)
    related: list[dict] = field(default_factory=list)  # closest chunks when nothing relevant was found
    llm_error: bool = False


def hit_to_dict(label: str, hit: Hit) -> dict:
    m = hit.meta
    return {
        "label": label, "source": m["source"], "page": m.get("page", 0), "section": m.get("section", ""),
        "timestamp": m.get("timestamp", ""), "line_start": m.get("line_start", 0),
        "line_end": m.get("line_end", 0), "score": round(hit.score, 3), "text": hit.text,
    }


def location_label(d: dict) -> str:
    parts = [d["source"]]
    if d.get("page"):
        parts.append(f"p.{d['page']}")
    if d.get("section"):
        parts.append(f"§ {d['section']}")
    if d.get("timestamp"):
        parts.append(f"⏱ {d['timestamp']}")
    if not d.get("page") and d.get("line_start"):
        parts.append(f"lines {d['line_start']}–{d['line_end']}")
    return " · ".join(parts)


def group_chunks(chunks: list[Hit], max_chars: int = 9000) -> list[list[Hit]]:
    groups, cur, size = [], [], 0
    for c in chunks:
        if cur and size + len(c.text) > max_chars:
            groups.append(cur)
            cur, size = [], 0
        cur.append(c)
        size += len(c.text)
    if cur:
        groups.append(cur)
    return groups


def parse_json_array(raw: str) -> list[dict]:
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict)]


def _s(v) -> str:
    return "" if v is None else str(v).strip()


# --------------------------------------------------------------------------- main class
class MeetingRAG:
    def __init__(self, store: VectorStore, llm: LLMClient):
        self.store = store
        self.llm = llm

    # ---- ingestion (upload -> extract -> chunk -> embed -> store) ------------
    def ingest(self, filename: str, data: bytes) -> int:
        pages = load_document(filename, data)
        chunks = chunk_document(filename, pages)
        return self.store.add_document(chunks, file_type=Path(filename).suffix.lower().lstrip("."))

    # ---- question answering --------------------------------------------------
    def ask(self, question: str, sources: list[str] | None = None, k: int = 6,
            min_similarity: float = config.DEFAULT_MIN_SIMILARITY, history: list[dict] | None = None) -> Answer:
        broad = re.search(r"\b(all|every|list|summar\w*|overall|across|pending|outstanding|decisions?|"
                          r"action items?|deadlines?|tasks?|who)\b", question, re.I)
        hits = self.store.search(question, k=max(k, 12) if broad else k, sources=sources,
                                 min_similarity=min_similarity)
        if not hits:
            closest = self.store.search(question, k=3, sources=sources, min_similarity=0.0)
            return Answer(NOT_FOUND_MESSAGE, False, [], [hit_to_dict(f"R{i}", h) for i, h in enumerate(closest, 1)])

        context = "\n\n".join(
            f"[S{i}] ({location_label(hit_to_dict('', h))})\n{h.text}" for i, h in enumerate(hits, 1)
        )
        convo = ""
        if history:
            convo = "Previous conversation (for follow-up context only):\n" + "\n".join(
                f"{m['role'].title()}: {m['content'][:500]}" for m in history[-4:]
            ) + "\n\n"
        user = f"{convo}Meeting excerpts:\n{context}\n\nQuestion: {question}"
        try:
            text = self.llm.generate(prompts.QA_SYSTEM, user, max_tokens=1200)
        except Exception as e:  # noqa: BLE001 - never leave the user with nothing
            msg = (f"The LLM call failed ({type(e).__name__}: {e}). "
                   "Showing the most relevant excerpts from your documents instead.")
            return Answer(msg, True, [hit_to_dict(f"S{i}", h) for i, h in enumerate(hits, 1)], llm_error=True)

        if text.strip().upper().startswith("NOT_FOUND"):
            return Answer(NOT_FOUND_MESSAGE, False, [], [hit_to_dict(f"R{i}", h) for i, h in enumerate(hits[:3], 1)])

        cited = sorted({int(n) for n in re.findall(r"\[S(\d+)\]", text) if 1 <= int(n) <= len(hits)})
        chosen = [(i, hits[i - 1]) for i in cited] or list(enumerate(hits, 1))
        return Answer(text, True, [hit_to_dict(f"S{i}", h) for i, h in chosen])

    # ---- summaries ----------------------------------------------------------------
    def summarize(self, source: str, force: bool = False) -> str:
        chunks = self.store.get_chunks(source)
        if not chunks:
            return "_No content found for this document._"
        key = self._cache_key("summary", source, chunks)
        if not force and (cached := self._cache_get(key)) is not None:
            return cached
        parts = []
        for g in group_chunks(chunks):
            body = "\n\n".join(c.text for c in g)
            parts.append(self.llm.generate(prompts.SUMMARY_SYSTEM, f"Meeting: {source}\n\n{body}", max_tokens=1500))
        result = parts[0] if len(parts) == 1 else self.llm.generate(
            prompts.SUMMARY_REDUCE_SYSTEM, "\n\n---\n\n".join(parts), max_tokens=1800)
        self._cache_set(key, result)
        return result

    # ---- structured extraction (action items / decisions) ----------------------
    def extract(self, source: str, kind: str, force: bool = False) -> list[dict]:
        assert kind in ("actions", "decisions")
        chunks = self.store.get_chunks(source)
        if not chunks:
            return []
        key = self._cache_key(kind, source, chunks)
        if not force and (cached := self._cache_get(key)) is not None:
            return cached
        system = prompts.ACTIONS_SYSTEM if kind == "actions" else prompts.DECISIONS_SYSTEM
        by_idx = {c.meta["chunk_index"]: c.meta for c in chunks}
        items: list[dict] = []
        for g in group_chunks(chunks):
            body = "\n\n".join(f"[C {c.meta['chunk_index']}]\n{c.text}" for c in g)
            raw = self.llm.generate(system, f"Meeting: {source}\n\n{body}", max_tokens=2500, temperature=0.0)
            for it in parse_json_array(raw):
                items.append(self._normalize(it, kind, source, by_idx))
        items = self._dedupe(items, kind)
        self._cache_set(key, items)
        return items

    @staticmethod
    def _normalize(it: dict, kind: str, source: str, by_idx: dict) -> dict:
        try:
            meta = by_idx.get(int(it.get("chunk")), {})
        except (TypeError, ValueError):
            meta = {}
        base = {"meeting": source, "page": meta.get("page", 0), "section": meta.get("section", ""),
                "timestamp": meta.get("timestamp", ""), "line_start": meta.get("line_start", 0),
                "line_end": meta.get("line_end", 0)}
        if kind == "decisions":
            return {**base, "decision": _s(it.get("decision")), "context": _s(it.get("context")),
                    "made_by": _s(it.get("made_by"))}
        date = _s(it.get("deadline_date"))
        try:
            dt.date.fromisoformat(date)
        except ValueError:
            date = ""
        return {**base, "task": _s(it.get("task")), "owner": _s(it.get("owner")) or "Unassigned",
                "deadline": _s(it.get("deadline")), "deadline_date": date,
                "status": STATUS_MAP.get(_s(it.get("status")).lower(), "Pending"),
                "priority": _s(it.get("priority")), "evidence": _s(it.get("evidence"))}

    @staticmethod
    def _dedupe(items: list[dict], kind: str) -> list[dict]:
        field_ = "task" if kind == "actions" else "decision"
        seen, out = set(), []
        for it in items:
            if not it[field_]:
                continue
            sig = re.sub(r"\W+", " ", it[field_].lower()).strip()[:60]
            if sig not in seen:
                seen.add(sig)
                out.append(it)
        return out

    # ---- disk cache (avoid paying for the same LLM call twice) -------------------
    def _cache_key(self, kind: str, source: str, chunks: list[Hit]) -> str:
        h = hashlib.sha1("".join(c.text for c in chunks).encode()).hexdigest()
        raw = f"{kind}|{source}|{h}|{self.llm.provider}|{self.llm.model}"
        return hashlib.sha1(raw.encode()).hexdigest()

    @staticmethod
    def _cache_get(key: str):
        p = config.CACHE_DIR / f"{key}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    @staticmethod
    def _cache_set(key: str, value) -> None:
        (config.CACHE_DIR / f"{key}.json").write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
