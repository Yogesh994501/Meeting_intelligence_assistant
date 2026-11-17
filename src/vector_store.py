"""Steps 5-7: ChromaDB vector storage, semantic similarity search, chunk retrieval."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings

from .chunking import Chunk


@dataclass
class Hit:
    text: str
    meta: dict
    score: float  # cosine similarity (1.0 = identical)


class VectorStore:
    def __init__(self, path: Path | str, embedder, collection: str = "meetings"):
        self.embedder = embedder
        self.path = str(path)
        self.client = chromadb.PersistentClient(path=self.path, settings=Settings(anonymized_telemetry=False))
        self.collection_name = collection
        self.model_name = getattr(embedder, "model_name", "unknown")
        self.stale_reset = False  # True if an index built with another embedding model was discarded
        self.col = self._open()

    def _open(self):
        # A vector index is only valid for the embedding model that produced it. If the stored model
        # differs (or the collection is unreadable) the old vectors are useless -> start fresh.
        try:
            old = self.client.get_collection(self.collection_name)
            if (old.metadata or {}).get("embedding_model", self.model_name) != self.model_name:
                raise ValueError("embedding model changed")
        except ValueError:
            try:
                self.client.delete_collection(self.collection_name)
                self.stale_reset = True
            except Exception:  # noqa: BLE001  (collection did not exist)
                pass
        except Exception:  # noqa: BLE001
            pass
        meta = {"embedding_model": self.model_name}
        try:  # newer Chroma: configuration; older: metadata key
            return self.client.get_or_create_collection(
                self.collection_name, metadata=meta, configuration={"hnsw": {"space": "cosine"}})
        except Exception:  # noqa: BLE001
            return self.client.get_or_create_collection(
                self.collection_name, metadata={**meta, "hnsw:space": "cosine"})

    def info(self) -> dict:
        n = self.col.count()
        dim = 0
        if n:
            dim = len(self.col.get(limit=1, include=["embeddings"])["embeddings"][0])
        return {"location": self.path, "collection": self.collection_name, "vectors stored": n,
                "dimensions": dim, "distance metric": "cosine", "embedding model": self.model_name}

    def get_vector(self, chunk_id: str) -> list[float]:
        r = self.col.get(ids=[chunk_id], include=["embeddings"])
        return [float(x) for x in r["embeddings"][0]] if len(r["ids"]) else []

    # ---- write -------------------------------------------------------------
    def add_document(self, chunks: list[Chunk], file_type: str = "") -> int:
        if not chunks:
            return 0
        source = chunks[0].metadata["source"]
        self.delete_document(source)  # re-uploading replaces the old version
        now = dt.datetime.now().isoformat(timespec="seconds")
        for i in range(0, len(chunks), 64):
            batch = chunks[i : i + 64]
            texts = [c.text for c in batch]
            metas = [{**c.metadata, "file_type": file_type, "uploaded_at": now} for c in batch]
            self.col.add(
                ids=[c.id for c in batch],
                documents=texts,
                embeddings=self.embedder.embed_documents(texts),
                metadatas=metas,
            )
        return len(chunks)

    def delete_document(self, source: str) -> None:
        self.col.delete(where={"source": source})

    def reset(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.col = self._open()

    # ---- read --------------------------------------------------------------
    def search(self, query: str, k: int = 6, sources: list[str] | None = None, min_similarity: float = 0.0) -> list[Hit]:
        total = self.col.count()
        if total == 0:
            return []
        where = None
        if sources:
            where = {"source": sources[0]} if len(sources) == 1 else {"source": {"$in": list(sources)}}
        qv = self.embedder.embed_query(query)
        res = self.col.query(
            query_embeddings=[qv],
            n_results=min(k, total),
            where=where,
            include=["documents", "metadatas", "embeddings"],
        )
        q = np.asarray(qv, dtype=float)
        q = q / (np.linalg.norm(q) or 1.0)
        hits = []
        for doc, meta, emb in zip(res["documents"][0], res["metadatas"][0], res["embeddings"][0]):
            e = np.asarray(emb, dtype=float)
            # cosine similarity computed here, so scores are correct whatever distance metric Chroma uses
            score = float(q @ e / (np.linalg.norm(e) or 1.0))
            if score >= min_similarity:
                hits.append(Hit(doc, meta, score))
        return sorted(hits, key=lambda h: h.score, reverse=True)

    def get_chunks(self, source: str) -> list[Hit]:
        res = self.col.get(where={"source": source}, include=["documents", "metadatas"])
        hits = [Hit(d, m, 1.0) for d, m in zip(res["documents"], res["metadatas"])]
        return sorted(hits, key=lambda h: h.meta.get("chunk_index", 0))

    def list_documents(self) -> dict[str, dict]:
        res = self.col.get(include=["metadatas"])
        docs: dict[str, dict] = {}
        for m in res["metadatas"]:
            d = docs.setdefault(
                m["source"],
                {"chunks": 0, "pages": 0, "file_type": m.get("file_type", ""), "uploaded_at": m.get("uploaded_at", "")},
            )
            d["chunks"] += 1
            d["pages"] = max(d["pages"], m.get("page", 0))
        return dict(sorted(docs.items()))
