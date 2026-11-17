"""Offline pipeline tests: no model download or API key needed (fake embedder + fake LLM)."""
import json
import math
import re
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src import config
from src.chunking import chunk_document
from src.loaders import load_document
from src.rag import MeetingRAG, parse_json_array
from src.vector_store import VectorStore

SAMPLES = config.SAMPLE_DIR


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder (similar words -> similar vectors)."""
    dim = 512

    def _vec(self, text):
        v = [0.0] * self.dim
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[zlib.crc32(w.encode()) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    provider, model = "fake", "fake"
    is_configured = True

    def __init__(self):
        self.last_user = ""

    def generate(self, system, user, max_tokens=0, temperature=0):
        self.last_user = user
        if "ACTION ITEMS" in system:
            return '```json\n[{"task":"Finalize API spec","owner":"Rahul Verma","deadline":"November 6",' \
                   '"deadline_date":"2026-11-06","status":"done","chunk":0,"evidence":"x"}]\n```'
        if "DECISIONS" in system:
            return '[{"decision":"Use Razorpay","context":"cheapest","made_by":"Team","chunk":0}]'
        if "Meeting Intelligence Assistant" in system:
            return "Razorpay was chosen [S1]."
        return "## Overview\nsummary"


@pytest.fixture()
def rag(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    store = VectorStore(tmp_path / "chroma", FakeEmbedder())
    r = MeetingRAG(store, FakeLLM())
    for f in SAMPLES.iterdir():
        if f.suffix in (".txt", ".pdf"):
            r.ingest(f.name, f.read_bytes())
    return r


def test_pdf_pages_and_chunk_metadata():
    pages = load_document("m.pdf", (SAMPLES / "2026-11-12_Budget_Review_Minutes.pdf").read_bytes())
    assert [p.page for p in pages] == [1, 2]
    chunks = chunk_document("m.pdf", pages)
    assert {c.metadata["page"] for c in chunks} == {1, 2}
    assert any("Marketing" in c.metadata["section"] for c in chunks)


def test_txt_sections_and_timestamps():
    pages = load_document("a.txt", (SAMPLES / "2026-11-02_Q4_Product_Planning.txt").read_bytes())
    chunks = chunk_document("a.txt", pages)
    assert all(c.metadata["page"] == 0 for c in chunks)
    assert any(c.metadata["timestamp"] for c in chunks)
    assert any(c.metadata["section"] for c in chunks)


def test_ingest_and_cross_document_search(rag):
    assert len(rag.store.list_documents()) == 3
    hits = rag.store.search("payment SDK Android 14 issue", k=3)
    assert any("Weekly_Engineering" in h.meta["source"] for h in hits)
    hits = rag.store.search("legal review contract", k=3, sources=["2026-11-12_Budget_Review_Minutes.pdf"])
    assert all(h.meta["source"].endswith(".pdf") for h in hits)


def test_ask_returns_sources(rag):
    ans = rag.ask("Which payment vendor was chosen?", k=4, min_similarity=0.05)
    assert ans.found and ans.sources and ans.sources[0]["source"]


def test_not_found_handling(rag):
    ans = rag.ask("zebra volcano submarine quantum", k=4, min_similarity=0.3)
    assert not ans.found and ans.sources == [] and "couldn't find" in ans.text


def test_llm_not_found_marker(rag):
    rag.llm.generate = lambda *a, **k: "NOT_FOUND"
    ans = rag.ask("Which payment vendor was chosen?", k=4, min_similarity=0.05)
    assert not ans.found


def test_extract_actions_normalised_and_cached(rag):
    src = "2026-11-02_Q4_Product_Planning.txt"
    items = rag.extract(src, "actions")
    assert items[0]["status"] == "Done" and items[0]["owner"] == "Rahul Verma"
    rag.llm.generate = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cache miss"))
    assert rag.extract(src, "actions") == items  # served from cache


def test_parse_json_robustness():
    assert parse_json_array("nothing here") == []
    assert parse_json_array('Sure!\n```json\n[{"a":1}]\n```') == [{"a": 1}]
    assert parse_json_array("[broken") == []


def test_stale_index_reset_and_info(tmp_path):
    class A(FakeEmbedder):
        model_name = "model-A"

    class B(FakeEmbedder):
        model_name = "model-B"

    pages = load_document("a.txt", (SAMPLES / "2026-11-02_Q4_Product_Planning.txt").read_bytes())
    s1 = VectorStore(tmp_path / "db", A())
    s1.add_document(chunk_document("a.txt", pages), "txt")
    info = s1.info()
    assert info["dimensions"] == 512 and info["vectors stored"] > 0 and len(s1.get_vector("a.txt::0")) == 512
    s2 = VectorStore(tmp_path / "db", B())  # different embedding model -> index reset
    assert s2.stale_reset and s2.col.count() == 0
    s3 = VectorStore(tmp_path / "db", B())  # same model -> data kept
    assert not s3.stale_reset


def test_scores_are_true_cosine(rag):
    chunk = rag.store.get_chunks("2026-11-02_Q4_Product_Planning.txt")[0]
    hits = rag.store.search(chunk.text, k=3)
    assert hits[0].score > 0.99 and all(-1.0 <= h.score <= 1.0 for h in hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_llm_failure_falls_back_to_excerpts(rag):
    def boom(*a, **k):
        raise RuntimeError("404 model not found")
    rag.llm.generate = boom
    ans = rag.ask("Which payment vendor was chosen?", k=4, min_similarity=0.05)
    assert ans.found and ans.llm_error and ans.sources and "404" in ans.text
