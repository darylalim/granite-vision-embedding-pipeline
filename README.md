# Granite Vision Embedding Pipeline

Streamlit + FastAPI app for generating vector embeddings from PDF documents and images using IBM Granite's [Vision Embedding](https://huggingface.co/ibm-granite/granite-vision-3.3-2b-embedding) model, with batch processing via a SQLite job queue and RAG answer generation via an external VLM.

## Features

- Upload PDFs and images (PNG, JPG, JPEG, WebP) in bulk
- PDF page rendering at configurable DPI (72, 150, 300) via [PyMuPDF](https://pymupdf.readthedocs.io/)
- Multi-vector embeddings with [Granite Vision 3.3 2B Embedding](https://huggingface.co/ibm-granite/granite-vision-3.3-2b-embedding)
- Batch processing of thousands of documents via background worker thread
- Cross-document text search with top-K and score threshold filtering
- RAG answer generation via external OpenAI-compatible VLM (OpenAI, Ollama, vLLM, etc.)
- Per-document and combined JSON embedding downloads
- Real-time progress bar with auto-refresh on job completion
- Automatic device selection (MPS > CUDA > CPU)
- Job dashboard with status tracking, filtering, bulk delete, and per-job deletion

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
