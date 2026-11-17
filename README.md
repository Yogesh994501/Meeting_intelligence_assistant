# 🗂️ Meeting Intelligence Assistant

A RAG (Retrieval-Augmented Generation) app that lets you upload meeting transcripts / minutes and then
**summarize meetings, ask questions across meetings, and track decisions, action items, owners and deadlines** - with
every answer linked back to the source document, page and section.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # add GOOGLE_API_KEY for Gemini (or use Ollama - no key)
streamlit run app.py
```

Then click **Load samples** in the sidebar (2 TXT + 1 PDF demo meetings) or upload your own files.
The first run downloads the embedding model (~90 MB, one time).

> **Gemini:** free key from https://aistudio.google.com/apikey -> put it in `.env` as `GOOGLE_API_KEY`
> (or paste it in the sidebar). Model name is editable (default `gemini-3.5-flash`; the app auto-falls back to `gemini-flash-latest` if a model is retired).
>
> **Ollama (no key, fully offline):** install [Ollama](https://ollama.com), run `ollama pull llama3.1`,
> keep it running, and choose *Ollama (local, free)* in the sidebar. Embeddings always run locally.

## Features

| Tab | What it does |
|---|---|
| 💬 **Ask** | Chat across all meetings (or a chosen subset). Cited answers `[S1]`, source cards with document / page / section / timestamp / lines / relevance score. Follow-up questions supported. |
| 📝 **Summaries** | Overview, key discussion points, decisions, action items, open questions - per meeting; long meetings use map-reduce. Download as Markdown. |
| ✅ **Action items & deadlines** | Extracts tasks with **owner, deadline, status, priority**. Views: table (CSV export), by owner, deadline timeline with overdue flags. |
| 🧭 **Decisions** | Extracts decisions with context, who decided, and source location. |
| 📚 **Library** | Indexed documents, stored chunks, delete / reset index. |

## Requirement checklist

| # | Requirement | Where |
|---|---|---|
| 1 | Document upload (PDF/TXT/DOCX/MD) | `app.py` sidebar uploader (files also saved to `data/uploads/`) |
| 2 | Loading & text extraction | `src/loaders.py` (pypdf per page, python-docx, TXT with encoding fallback) |
| 3 | Text chunking | `src/chunking.py` - LangChain `RecursiveCharacterTextSplitter` (900 chars / 150 overlap) |
| 4 | Embeddings | `src/embeddings.py` - `sentence-transformers/all-MiniLM-L6-v2` (local) |
| 5 | Vector storage | `src/vector_store.py` - persistent **ChromaDB** collection (cosine space) |
| 6 | Semantic similarity search | `VectorStore.search()` |
| 7 | Retrieval of relevant chunks | top-k + min-similarity filter + optional per-meeting filter |
| 8 | LLM answer generation | `src/llm.py` (Gemini / Ollama) + `src/prompts.py` + `src/rag.py` |
| 9 | Source references | document name, page (PDF), section heading, timestamp, line range |
| 10 | Not-found handling | two layers: similarity threshold (no LLM call if nothing relevant) **and** the LLM must return `NOT_FOUND` if excerpts don't answer it |
| 11 | User-friendly UI | Streamlit (`app.py`) |

## Architecture

```
Upload ─► loaders ─► chunking (+page/section/timestamp metadata) ─► embeddings ─► ChromaDB
                                                                                      │
Question ─► embed query ─► similarity search (top-k, min-sim, meeting filter) ◄────────┘
                                   │
                     numbered excerpts [S1..Sn] ─► LLM (answer only from excerpts, cite, or NOT_FOUND)
                                   │
                       answer + source cards (doc · page · section · lines · score)

Summaries / action items / decisions: all chunks of a meeting ─► LLM (map-reduce / JSON extraction) ─► cached
```

## How "not found" works
1. If no chunk reaches the **min. similarity** (sidebar, default 0.25) the app answers *"couldn't find that"* **without calling the LLM** and shows the closest excerpts for transparency.
2. If chunks pass but don't actually answer the question, the LLM is instructed to reply `NOT_FOUND`, which the app turns into the same message.
3. Partial answers state clearly what is not mentioned.

## Project structure
```
app.py                  Streamlit UI
src/config.py           settings
src/loaders.py          PDF / TXT / DOCX extraction
src/chunking.py         splitter + provenance metadata
src/embeddings.py       embedding model
src/vector_store.py     ChromaDB wrapper
src/llm.py              Gemini / Ollama
src/prompts.py          prompts
src/rag.py              ask / summarize / extract (+ disk cache)
sample_data/            3 demo meetings (TXT + PDF)
tests/test_pipeline.py  offline tests (fake embedder + fake LLM):  pytest -q
```

## Tips & notes
- Re-uploading a file with the same name replaces its old chunks.
- Summaries/extractions are cached in `data/cache/`; tick *Regenerate / Re-extract* to refresh.
- TXT/DOCX have no real pages, so references show section heading + timestamp + line range instead.
- Scanned (image-only) PDFs need OCR first; the app tells you when no text can be extracted.
- Good transcript formats: speaker names, `[hh:mm:ss]` timestamps, headings like `AGENDA ITEM 1: ...` or `ACTION ITEMS:` improve section detection.
- Try: *"Why was the launch date changed?"*, *"Who owns the payment integration and by when?"*, *"What did Finance approve?"*, *"What was decided about Kubernetes?"* (not in the docs → not-found demo).

## Troubleshooting: "it doesn't answer"
Click **Test LLM connection** in the sidebar first - it shows the real error.
- *model not found / 404 / 503* -> the model was retired; use `gemini-3.5-flash` (or `gemini-flash-latest`).
- *API key not valid / 403 / 429* -> check `GOOGLE_API_KEY` / quota.
- *Cannot reach Ollama* -> run `ollama serve` and `ollama pull llama3.1`.
- *"I couldn't find that..." for everything* -> lower **Min. similarity** (sidebar > Retrieval settings) to 0.05-0.1;
  the "closest excerpts" shown under the message list the best matches with their scores.
- *Nothing indexed* -> after choosing files you must click **Index files** (or **Load samples**).

## Troubleshooting
- **Every question fails** -> click *Test LLM connection* in the sidebar. A 404/503 usually means the Gemini model name is retired; use a current one (default `gemini-3.5-flash`).
- **"I couldn't find that" for everything** -> open *Library -> Retrieval debugger* to see real similarity scores; lower *Min. similarity* if scores are just under the threshold. Also make sure you clicked **Index files** after choosing files.
- **Index built with another embedding model** -> the app detects this, resets the ChromaDB index and asks you to re-index.
- Delete the `data/` folder at any time for a completely clean start.
