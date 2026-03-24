# Granite Vision Embedding Pipeline

Streamlit + FastAPI app for generating vector embeddings from PDF documents and images using IBM Granite's [Vision Embedding](https://huggingface.co/ibm-granite/granite-vision-3.3-2b-embedding) model, with batch processing via a SQLite job queue and RAG answer generation via an external VLM.

## Features

- Tabbed UI (Upload, Jobs, Query) with sidebar health info and connection check
- Upload PDFs and images (PNG, JPG, JPEG, WebP) in bulk
- PDF page rendering at configurable DPI (72, 150, 300) via [PyMuPDF](https://pymupdf.readthedocs.io/)
- Multi-vector embeddings with [Granite Vision 3.3 2B Embedding](https://huggingface.co/ibm-granite/granite-vision-3.3-2b-embedding)
- Batch processing of thousands of documents via background worker thread
- Cross-document text search with top-K and score threshold filtering
- RAG answer generation via external OpenAI-compatible VLM (OpenAI, Ollama, vLLM, etc.)
- Per-document and combined JSON embedding downloads
- Auto-refreshing job dashboard with dataframe, status badges, and detail panel
- Combined Search + Ask query tab with spinners and help tooltips
- Automatic device selection (MPS > CUDA > CPU)

## Setup

```bash
uv sync
```

Start the API server and Streamlit UI in separate terminals:

```bash
# Terminal 1: API server
uv run uvicorn api.app:create_app --factory --port 8000

# Terminal 2: Streamlit UI
uv run streamlit run streamlit_app.py
```

## Development

```bash
uv run ruff check .   # lint
uv run ruff format .  # format
uv run ty check       # typecheck
uv run pytest         # test
```

## Architecture

```
streamlit_app.py          Streamlit UI (tabs, sidebar, auto-refresh via @st.fragment)
api/
  app.py                  FastAPI routes, _cleanup_job_files(), _run_search()
  models.py               Pydantic request/response schemas
  database.py             SQLite connection, schema, job CRUD (indexed)
  worker.py               Background embedding worker thread
core/
  constants.py            MODEL_ID, DPI_OPTIONS, limits
  types.py                EmbeddingProcessor protocol
  embedding.py            Model loading, image loading, embed()
  rendering.py            PDF page rendering (PyMuPDF)
  search.py               Embedding search + filtering
  generation.py           Image encoding, VLM message building
tests/
  conftest.py             Shared fixtures (db, dirs)
  factories.py            Shared mock factories
  test_core.py            Core function tests
  test_generation.py      Generation function tests
  test_database.py        SQLite job management tests
  test_worker.py          Worker thread tests
  test_api.py             API helper and route tests
  test_streamlit_app.py   Streamlit UI tests (AppTest integration)
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `API_URL` | `http://localhost:8000` | API base URL (Streamlit) |
| `UPLOAD_DIR` | `uploads/` | Uploaded file storage |
| `RESULT_DIR` | `results/` | Embedding output storage |
| `DATABASE_PATH` | `data/jobs.db` | SQLite database path |
| `GENERATION_API_URL` | None | VLM endpoint for answer generation |
| `GENERATION_API_KEY` | `""` | VLM auth token |
| `GENERATION_MODEL` | None | VLM model ID |
| `GENERATION_MAX_TOKENS` | `1024` | Max VLM response tokens |
| `GENERATION_TIMEOUT` | `120` | VLM request timeout (seconds) |
