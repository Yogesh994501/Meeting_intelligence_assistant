"""Central configuration (override anything via environment variables or a .env file)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("MIA_DATA_DIR", BASE_DIR / "data"))
CHROMA_DIR = DATA_DIR / "chroma"
CACHE_DIR = DATA_DIR / "cache"
UPLOAD_DIR = DATA_DIR / "uploads"
SAMPLE_DIR = BASE_DIR / "sample_data"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 900))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

DEFAULT_TOP_K = 8
DEFAULT_MIN_SIMILARITY = 0.15  # cosine similarity; below this a chunk is considered irrelevant

SUPPORTED_TYPES = ["pdf", "txt", "docx", "md"]

for _d in (CHROMA_DIR, CACHE_DIR, UPLOAD_DIR):
    _d.mkdir(parents=True, exist_ok=True)
