# Granite Vision Embedding Pipeline

## Project Overview

Streamlit + FastAPI app for generating vector embeddings from PDF documents and images using IBM Granite's [Vision Embedding](https://huggingface.co/ibm-granite/granite-vision-3.3-2b-embedding) model, with batch processing via a SQLite job queue and RAG answer generation via an external VLM.

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

- `fastapi` — REST API framework
- `uvicorn` — ASGI server
- `httpx` — HTTP client (Streamlit → API, API → VLM)
- `python-multipart` — file upload handling
- `transformers` — Hugging Face model loading (`AutoModel`, `AutoProcessor`)
- `pymupdf` — PDF page rendering
- `torch` — tensor operations
- `torchvision` — required by Granite model remote code
- `streamlit` — web user interface
- `ruff` — linting/formatting (dev)
- `ty` — type checking (dev)
- `pytest` — testing (dev)

## Configuration

`pyproject.toml` — project metadata, dependencies, dev dependency group, ruff lint isort (`combine-as-imports`), ty (`python-version = "3.12"`)

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

## Architecture

### Overview

```
Streamlit UI  →  FastAPI Backend  →  Embedding Worker (background thread)
                      ↕                     ↕
                   SQLite DB           External VLM API
                  (WAL mode)          (answer generation)
                      ↕
                 File Storage (uploads/ , results/)
```

### Entry Points

- `streamlit_app.py` — thin API client (Streamlit UI)
- `api/app.py` — FastAPI backend (`create_app()` factory)

### Core Module (`core/`)

Pure logic with no Streamlit or FastAPI dependencies:

- `core/constants.py` — `MODEL_ID`, `DPI_OPTIONS`, `IMAGE_EXTENSIONS`, `MAX_UPLOAD_BYTES`, `GENERATION_MAX_TOKENS`
- `core/types.py` — `EmbeddingProcessor` protocol
- `core/embedding.py` — `get_device`, `load_model`, `load_image`, `embed`
- `core/rendering.py` — `render_pages`, `render_page`
- `core/search.py` — `search_multi`, `filter_results`
- `core/generation.py` — `encode_image`, `build_messages`

### API Module (`api/`)

- `api/app.py` — FastAPI routes (`create_app()` factory)
- `api/models.py` — Pydantic request/response models
- `api/database.py` — SQLite connection, schema, job CRUD
- `api/worker.py` — `EmbeddingWorker` background thread

### Embedding Model

[Granite Vision 3.3 2B Embedding](https://huggingface.co/ibm-granite/granite-vision-3.3-2b-embedding) — multi-vector vision-language embedding model (loaded via `trust_remote_code=True`)

### Pipeline

Upload files → API saves to `uploads/` and creates SQLite job → worker picks up pending jobs (FIFO) → renders PDF pages or loads images → embeds with model → saves JSON + `.pt` to `results/` → marks completed → Streamlit polls for status

### Answer Generation

`POST /ask` retrieves relevant pages via the embedding search pipeline, re-renders only the matched pages, builds an OpenAI-compatible request using `build_messages()`, and calls an external VLM via `httpx.AsyncClient`. Returns a grounded answer with source citations. Requires `GENERATION_API_URL` and `GENERATION_MODEL` to be configured; returns 503 otherwise.

### Worker

- Background thread started during FastAPI lifespan
- Loads model once, reuses for all jobs and search
- Polls SQLite for pending jobs (FIFO by `created_at`)
- Search serialized via `queue.Queue` + `concurrent.futures.Future`
- LRU tensor cache (max 500 entries) for search

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
- `GENERATION_MAX_TOKENS = 1024`

### API Routes

```
POST   /jobs              Upload file + DPI, returns job ID (201)
GET    /jobs              List jobs, optional ?status= filter
GET    /jobs/{id}         Single job status and metadata
DELETE /jobs/{id}         Delete job and files (409 if processing)
GET    /jobs/{id}/result  Download embedding JSON
POST   /search            Text query, returns ranked results
POST   /ask              Text query + retrieval, returns VLM-generated answer (503 if not configured)
GET    /health            Device, queue depth, worker status
```

### Database

SQLite `jobs` table: `id`, `status` (pending/processing/completed/failed), `created_at`, `updated_at`, `file_name`, `file_stem`, `file_path`, `file_type`, `dpi`, `page_count`, `duration_ns`, `result_path`, `tensor_path`, `error`

### JSON Output

Fields per document:

- `file_name` (string) — file stem without extension
- `model` (string) — model ID
- `dpi` (integer) — render DPI (72–300)
- `embeddings` (number[][][]) — per-page multi-vector embeddings (page → patches → 128-dim)
- `total_duration` (integer) — nanoseconds
- `page_count` (integer) — pages processed

### Search

`POST /search` dispatches to worker thread, which scores query against page embeddings via `search_multi`, then applies `filter_results` (min score + top-K). Optional `filter_file_id` restricts to one document.

### Error Handling

- Invalid file types → 400
- Files over 50 MB → 400
- Delete processing job → 409
- Failed jobs store sanitized error (no stack traces)
- Empty/corrupt PDFs → job marked failed
- Streamlit shows API connection errors gracefully
- VLM not configured → 503
- VLM unreachable/timeout → 502
- VLM error response → 502

## Tests

- `tests/test_core.py` — core functions: `TestDpiOptions`, `TestImageExtensions`, `TestMaxUploadBytes`, `TestGenerationMaxTokens`, `TestLoadImage`, `TestLoadModel`, `TestGetDevice`, `TestRenderPages`, `TestRenderPage`, `TestEmbed`, `TestFilterResults`, `TestSearchMulti`
- `tests/test_generation.py` — generation functions: `TestEncodeImage`, `TestBuildMessages`
- `tests/test_database.py` — SQLite job management: `TestInitDb`, `TestCreateJob`, `TestGetJob`, `TestListJobs`, `TestUpdateJob`, `TestDeleteJob`, `TestResetProcessingJobs`, `TestNextPendingJob`
- `tests/test_worker.py` — embedding worker: `TestProcessJob`, `TestStartupRecovery`, `TestTensorCache`, `TestSearchDispatch`
- `tests/test_api.py` — FastAPI routes: `TestHealth`, `TestUploadJob`, `TestListJobs`, `TestGetJob`, `TestDeleteJob`, `TestGetResult`, `TestSearch`, `TestAsk`
- `tests/data/pdf/single_page.pdf` — single-page PDF fixture
- `tests/data/pdf/multi_page.pdf` — multi-page PDF fixture (3 pages)
- `tests/data/images/red.png` — PNG image fixture
- `tests/data/images/blue.jpg` — JPG image fixture
- `tests/data/images/green.webp` — WebP image fixture

## Resources

- [Model Card](https://huggingface.co/ibm-granite/granite-vision-3.3-2b-embedding)
