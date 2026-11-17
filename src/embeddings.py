"""Step 4: embedding generation (local sentence-transformers model, no API key needed)."""
from __future__ import annotations


class Embedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
