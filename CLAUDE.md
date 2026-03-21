# Granite Vision Embedding Pipeline

## Project Overview

Streamlit web app + FastAPI backend for generating vector embeddings from PDF documents and images and searching over them using IBM Granite's [Vision Embedding](https://huggingface.co/ibm-granite/granite-vision-3.3-2b-embedding) model. Supports batch processing of thousands of documents via a job queue.

## Setup

```bash
uv sync

# Terminal 1: API server
uv run uvicorn api.app:create_app --factory --port 8000

# Terminal 2: Streamlit UI
uv run streamlit run streamlit_app.py
```

## Commands

- **API server**: `uv run uvicorn api.app:create_app --factory --port 8000`
- **Streamlit UI**: `uv run streamlit run streamlit_app.py`
- **Lint**: `uv run ruff check .`
- **Format**: `uv run ruff format .`
- **Typecheck**: `uv run ty check`
- **Test**: `uv run pytest`

## Code Style

- snake_case for functions/variables, PascalCase for classes
- Type annotations on all parameters and returns
- isort with combine-as-imports (configured in `pyproject.toml`)

## Dependencies

- `transformers` — Hugging Face model loading (`AutoModel`, `AutoProcessor`)
- `pymupdf` — PDF page rendering
- `torch` — tensor operations
- `streamlit` — web user interface
- `fastapi` — REST API framework
- `uvicorn` — ASGI server
- `httpx` — HTTP client for Streamlit → API communication
- `python-multipart` — file upload handling
- `ruff` — linting/formatting (dev)
- `ty` — type checking (dev)
- `pytest` — testing (dev)

## Configuration

- `pyproject.toml` — project metadata, dependencies, dev dependency group, ruff lint isort (`combine-as-imports`), ty (`python-version = "3.12"`)
- `API_URL` env var — API base URL (default: `http://localhost:8000`)
- `UPLOAD_DIR` env var — upload directory (default: `uploads/`)
- `RESULT_DIR` env var — results directory (default: `results/`)
- `DATABASE_PATH` env var — SQLite path (default: `data/jobs.db`)

## Architecture

### Overview

```
Streamlit UI  →  FastAPI Backend  →  Embedding Worker (background thread)
                      ↕
                   SQLite DB (WAL mode)
                      ↕
                 File Storage (uploads/ , results/)
```

### Entry Points

- `streamlit_app.py` — thin API client (Streamlit UI)
- `api/app.py` — FastAPI backend with `create_app()` factory

### Core Module (`core/`)

Pure logic with no Streamlit or FastAPI dependencies:

- `core/constants.py` — `MODEL_ID`, `DPI_OPTIONS`, `IMAGE_EXTENSIONS`, `MAX_UPLOAD_BYTES`
- `core/types.py` — `EmbeddingProcessor` Protocol
- `core/embedding.py` — `get_device`, `load_model`, `load_image`, `embed`
- `core/rendering.py` — `render_pages`
- `core/search.py` — `search_multi`, `filter_results`

### API Module (`api/`)

- `api/app.py` — FastAPI routes (`create_app()` factory)
- `api/database.py` — SQLite connection, schema, job CRUD queries
- `api/worker.py` — `EmbeddingWorker` background thread with job processing and search
- `api/models.py` — Pydantic request/response models

### Embedding Model

[Granite Vision 3.3 2B Embedding](https://huggingface.co/ibm-granite/granite-vision-3.3-2b-embedding) — multi-vector vision-language embedding model (loaded via `trust_remote_code=True`)

### Pipeline

Upload files → API saves to `uploads/` and creates SQLite job → Worker thread picks up pending jobs → render PDF pages as images at configurable DPI or load images directly → embed with model → save JSON + `.pt` tensor to `results/` → mark completed → Streamlit polls for status

### Worker

- Background thread started during FastAPI lifespan
- Loads model once at startup, reuses for all jobs and search
- Polls SQLite for pending jobs (FIFO by `created_at`)
- Search requests serialized via `queue.Queue` + `concurrent.futures.Future`
- LRU tensor cache (max 500 entries) for fast search

### Performance

- Best available device: MPS > CUDA > CPU
- `torch.inference_mode()` for inference
- `torch.float16` for model precision
- `time.perf_counter_ns()` for timing (nanoseconds)
- SQLite WAL mode + busy_timeout for concurrent access

### Constants

- `MODEL_ID = "ibm-granite/granite-vision-3.3-2b-embedding"`
- `DPI_OPTIONS = {"Low (72)": 72, "Medium (150)": 150, "High (300)": 300}`
- `IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}`
- `MAX_UPLOAD_BYTES = 50 * 1024 * 1024`

### API Routes

```
POST   /jobs              Upload file + DPI → save file, create job, return job ID
GET    /jobs              List all jobs (optional ?status= filter)
GET    /jobs/{id}         Get single job status and metadata
DELETE /jobs/{id}         Delete job and files (409 if processing)
GET    /jobs/{id}/result  Download embedding JSON
POST   /search            Text query → ranked results via worker
GET    /health            Device, queue depth, worker status
```

### Database

SQLite `jobs` table with fields: `id`, `status` (pending/processing/completed/failed), `created_at`, `updated_at`, `file_name`, `file_stem`, `file_path`, `file_type`, `dpi`, `page_count`, `duration_ns`, `result_path`, `tensor_path`, `error`.

### JSON Download

Per-document download via API, plus "Download All" in Streamlit. Fields per document:

- `file_name` (string) — file stem without extension
- `model` (string) — model that produced the embeddings
- `dpi` (integer) — render resolution in dots per inch (72–300)
- `embeddings` (number[][][]) — per-page multi-vector embeddings (page → patches → 128-dim vectors)
- `total_duration` (integer) — total duration in nanoseconds
- `page_count` (integer) — number of PDF pages processed

### Search

Text query submitted to `POST /search`, dispatched to worker thread which scores against page embeddings across all completed documents via `search_multi`. Results filtered by `filter_results` (min score threshold + top-K). Optional document filter via `filter_file_id`.

### Error Handling

- Invalid file types rejected with 400
- Files over 50 MB rejected with 400
- Processing jobs cannot be deleted (409)
- Failed jobs store sanitized error message (no stack traces)
- Streamlit shows API connection errors gracefully

## Tests

- `tests/test_core.py` — unit tests for core functions: `TestDpiOptions`, `TestImageExtensions`, `TestLoadImage`, `TestGetDevice`, `TestRenderPages`, `TestEmbed`, `TestFilterResults`, `TestSearchMulti`
- `tests/test_database.py` — SQLite job management: `TestInitDb`, `TestCreateJob`, `TestGetJob`, `TestListJobs`, `TestUpdateJob`, `TestDeleteJob`, `TestResetProcessingJobs`, `TestNextPendingJob`
- `tests/test_worker.py` — embedding worker: `TestProcessJob`, `TestStartupRecovery`, `TestTensorCache`, `TestSearchDispatch`
- `tests/test_api.py` — FastAPI routes: `TestHealth`, `TestUploadJob`, `TestListJobs`, `TestGetJob`, `TestDeleteJob`, `TestGetResult`, `TestSearch`
- `tests/data/pdf/single_page.pdf` — single-page PDF fixture
- `tests/data/pdf/multi_page.pdf` — multi-page PDF fixture (3 pages)
- `tests/data/images/red.png` — PNG image fixture
- `tests/data/images/blue.jpg` — JPG image fixture
- `tests/data/images/green.webp` — WebP image fixture

## Resources

- [Model Card](https://huggingface.co/ibm-granite/granite-vision-3.3-2b-embedding)
