# Batch Processing Design

## Goal

Add batch processing support for thousands of documents by introducing a FastAPI backend with a job queue, replacing the current synchronous in-process embedding flow.

## Architecture

```
Streamlit UI  →  FastAPI Backend  →  Embedding Worker (background thread)
                      ↕
                   SQLite DB (WAL mode)
                      ↕
                 File Storage (uploads/ , results/)
```

- **Streamlit** handles file upload, job submission, progress display, search, and download — communicates with FastAPI via `httpx.Client` (sync).
- **FastAPI** exposes a REST API for job CRUD, file upload, and search — manages SQLite for job state and coordinates the worker thread.
- **Embedding worker** runs as a background thread (not a child process) to avoid GPU tensor serialization issues. The GIL is released during `torch` operations and I/O, so a single background thread keeps the model loaded in-process while processing jobs sequentially.
- **SQLite** tracks job metadata (status, timestamps, page counts, durations, errors). Opened with `journal_mode=WAL` and `busy_timeout=5000` for safe concurrent reads/writes.
- **File storage** — uploaded files saved to `uploads/`, embedding result JSON written to `results/`, embedding tensors saved as `.pt` files alongside the JSON for fast search reloading.

The existing core functions (`render_pages`, `embed`, `search_multi`, `filter_results`) move into a shared `core/` module used by both the worker and the API. The API runs locally without authentication.

## Project Structure

```
granite-vision-embedding-pipeline/
├── streamlit_app.py              # Streamlit UI (rewritten as API client)
├── api/
│   ├── __init__.py
│   ├── app.py                    # FastAPI application and routes
│   ├── models.py                 # Pydantic request/response models
│   ├── database.py               # SQLite connection, schema, queries
│   └── worker.py                 # Background thread, job processing loop
├── core/
│   ├── __init__.py
│   ├── constants.py              # MODEL_ID, DPI_OPTIONS, IMAGE_EXTENSIONS
│   ├── types.py                  # EmbeddingProcessor protocol, EmbedResults TypedDict
│   ├── embedding.py              # get_device, load_model, embed, load_image
│   ├── rendering.py              # render_pages
│   └── search.py                 # search_multi, filter_results
├── tests/
│   ├── test_core.py              # Unit tests for core functions (migrated)
│   ├── test_api.py               # API route tests (FastAPI TestClient)
│   ├── test_worker.py            # Worker/job processing tests
│   ├── test_database.py          # SQLite query tests
│   └── data/                     # Existing fixtures (unchanged)
├── uploads/                      # Uploaded files (gitignored)
├── results/                      # Embedding JSON + .pt output (gitignored)
├── data/                         # SQLite database (gitignored)
├── pyproject.toml
└── CLAUDE.md
```

- `core/` holds pure logic extracted from the current `streamlit_app.py` — no Streamlit or FastAPI dependencies, testable in isolation.
- `core/constants.py` holds `MODEL_ID`, `DPI_OPTIONS`, `IMAGE_EXTENSIONS`.
- `core/types.py` holds `EmbeddingProcessor` protocol. `EmbedResults` TypedDict is removed — replaced by Pydantic models in `api/models.py` and database rows. `cleanup_stale_results` is also removed — job lifecycle is managed by the API.
- `core/embedding.py` holds `get_device`, `load_model` (without `@st.cache_resource`), `embed`, and `load_image`. `load_image(path: Path) -> Image.Image` opens a file, converts to RGB, and lets `UnidentifiedImageError`/`OSError` propagate (the worker catches them and marks the job as failed).
- `core/search.py` holds `search_multi` and `filter_results`. The single-document `search` function is dropped — `search_multi` covers that case with `filter_file_id`.
- `api/` holds everything backend — routes, database, worker.
- `streamlit_app.py` stays as the entry point but becomes a thin API client.
- Existing test logic migrates to `test_core.py` with updated import paths and patch targets (e.g., `@patch("core.embedding.torch")` instead of `@patch("streamlit_app.torch")`).

## Database Schema

Single `jobs` table:

```sql
CREATE TABLE jobs (
    id            TEXT PRIMARY KEY,   -- UUID
    status        TEXT NOT NULL,      -- pending, processing, completed, failed
    created_at    TEXT NOT NULL,      -- ISO 8601 timestamp
    updated_at    TEXT NOT NULL,      -- ISO 8601 timestamp
    file_name     TEXT NOT NULL,      -- original filename
    file_stem     TEXT NOT NULL,      -- filename without extension
    file_path     TEXT NOT NULL,      -- path in uploads/
    file_type     TEXT NOT NULL,      -- "pdf" or "image"
    dpi           INTEGER NOT NULL,   -- render DPI (72/150/300)
    page_count    INTEGER,           -- populated after processing
    duration_ns   INTEGER,           -- total embedding time in nanoseconds
    result_path   TEXT,              -- path to JSON in results/
    tensor_path   TEXT,              -- path to .pt file in results/
    error         TEXT               -- error message if failed
);
```

SQLite opened with `journal_mode=WAL` and `busy_timeout=5000` for safe concurrent access between the main thread (API inserts/deletes) and worker thread (status updates). Schema created via `CREATE TABLE IF NOT EXISTS` at connection initialization during FastAPI lifespan startup. Schema migrations are out of scope for v1.

### Job Lifecycle

```
pending → processing → completed
                    ↘ failed
```

1. **Upload** — Streamlit sends file to FastAPI, file saved to `uploads/`, job row inserted as `pending`.
2. **Queue** — Worker thread polls SQLite for the next `pending` job (oldest by `created_at`), sets to `processing`. Only one job is picked up at a time — the worker does not poll for the next job until the current one finishes.
3. **Process** — Worker renders pages (if PDF) or loads image, embeds, writes JSON to `results/{id}.json` and tensor to `results/{id}.pt`, sets to `completed` with `page_count`, `duration_ns`, `result_path`, `tensor_path`.
4. **Failure** — Any exception during processing sets status to `failed` with a sanitized error message (no stack traces).
5. **Poll** — Streamlit polls `GET /jobs` to update progress display.

No retry logic — failed jobs can be re-submitted.

## API Routes

```
POST   /jobs              Upload file + DPI → save file, create job, return job ID
GET    /jobs              List all jobs (with status filter query param)
GET    /jobs/{id}         Get single job status and metadata
DELETE /jobs/{id}         Cancel pending job or delete completed job (removes files)
GET    /jobs/{id}/result  Download embedding JSON from results/
POST   /search            Text query + top_k + min_score + optional file filter → ranked results
GET    /health            Worker status, job queue depth, device info
```

- `POST /jobs` accepts `multipart/form-data` (file + DPI param). Validates file type against `IMAGE_EXTENSIONS` + `pdf`. Rejects files larger than 50 MB. Returns `{job_id, status: "pending"}`.
- `GET /jobs` supports `?status=pending,processing` for filtering. Returns list sorted by `created_at`.
- `POST /search` enqueues a search request to the worker thread (see Worker Design) and awaits the result. Returns ranked results as JSON.
- `DELETE /jobs/{id}` removes the upload file, result files (JSON + .pt) using `Path.unlink(missing_ok=True)` for paths that may be NULL (pending/failed jobs), database row, and cache entry. Only allowed for `pending`, `completed`, or `failed` jobs — not `processing` (returns 409).
- Error responses use FastAPI's default `HTTPException` format: `{"detail": "message"}` with status codes 400 (validation), 404 (not found), 409 (state conflict).

## Worker Design

The worker runs as a **background thread** started during FastAPI's lifespan. It shares the same process as the API server, which avoids GPU tensor serialization issues that would occur with `ProcessPoolExecutor` (CUDA/MPS tensors cannot be pickled across process boundaries).

**Why a background thread:** The GIL is released during `torch` GPU operations and file I/O, so a single background thread can process jobs without blocking the API's async event loop. Only one model instance is loaded, and jobs process sequentially — no GPU contention.

**Model loading:** `load_model` moves to `core/embedding.py` without the `@st.cache_resource` decorator. The worker thread loads the model once at startup via `load_model(device)` and stores the model and processor as instance attributes on the worker object. These are reused for all jobs and search requests.

**Job processing loop:**
1. Poll SQLite for the next `pending` job (oldest `created_at`), sleeping 1 second between polls when idle.
2. Between polls, check the search request queue and process any pending search requests before polling for the next job.
3. Set job to `processing`.
4. Read file from `uploads/`, render pages if PDF or load image via `load_image`, embed, write JSON + `.pt` to `results/`.
5. Update job to `completed` with metadata, or `failed` with error.
6. Loop back to step 1.

**Search threading model:** Search requests are enqueued to the worker thread via a `queue.Queue` of `(search_params, concurrent.futures.Future)` tuples. The API route handler creates a `Future`, puts the request on the queue, and awaits the result (via `asyncio.wrap_future`). The worker thread checks this queue between job polling cycles and during idle waits (step 2 above). This ensures the model is never accessed concurrently from two threads — all model operations (embedding and search) are serialized on the worker thread. Trade-off: search requests block while a job is processing (potentially minutes for large documents). This is acceptable for a local tool; for lower latency, a future version could checkpoint between pages.

**Tensor cache:** The worker maintains an in-memory LRU cache of `{job_id: torch.Tensor}` for completed job embeddings, with a maximum of 500 entries. Tensors are loaded from `.pt` files on first search access and cached. Entries are evicted LRU-first when the cache is full, and explicitly removed when jobs are deleted. Evicted entries can be reloaded from `.pt` files on demand.

**Startup recovery:** The first operation in the worker thread's startup, before the first poll cycle, is resetting any jobs left as `processing` to `pending`.

**Shutdown:** FastAPI lifespan sets a stop event; the worker thread finishes its current job and exits.

## Streamlit UI Changes

The Streamlit app becomes a thin client that talks to FastAPI via `httpx.Client` (sync).

**Upload & Submit:** File uploader remains multi-file (PDFs + images), DPI selector stays. "Embed" button sends each file to `POST /jobs` individually, collects job IDs stored in `st.session_state.job_ids`.

**Progress Dashboard:** Replaces the current single progress bar. Uses a manual "Refresh" button that calls `st.rerun()` after fetching `GET /jobs`. Displays a table of jobs with filename, status (color indicators), page count, and duration. Processing jobs show a spinner.

**Results & Download:** Per-document expanders for completed jobs — fetch result JSON via `GET /jobs/{id}/result`. "Download All" concatenates results from all completed jobs.

**Search:** Search form unchanged (query, document filter, top-K, min score). Submits to `POST /search`, displays ranked results as before.

**Job Management:** Delete button per job (calls `DELETE /jobs/{id}`). Status filter dropdown (all / pending / processing / completed / failed).

No embedding logic runs in the Streamlit process. All heavy work goes through the API.

## Testing Strategy

**`test_core.py`** — migrated from current `test_app.py`:
- All existing test classes move here with updated imports (`core.embedding`, `core.rendering`, `core.search`) and patch targets (`@patch("core.embedding.torch")` etc.).
- `TestSearch` class is removed (single-doc `search` function dropped in favor of `search_multi`).
- New tests for `load_image` function.

**`test_api.py`** — FastAPI route tests using `TestClient`:
- Upload valid PDF/image, invalid file type rejection
- List jobs with status filter, get single job
- Delete job (allowed for pending/completed/failed, rejected for processing)
- Download result for completed job, 404 for pending/failed
- Search across completed jobs
- Health endpoint returns device and queue depth

**`test_worker.py`** — worker logic tests with mocked model:
- `process_job` with valid PDF/image → result JSON + .pt written, status completed
- `process_job` with corrupt file → status failed with error
- Startup recovery resets stale `processing` jobs to `pending`
- Search dispatched through worker returns ranked results
- Tensor cache populated on first search, evicted on delete

**`test_database.py`** — SQLite query tests:
- Job CRUD operations, status transitions, filtering, ordering
- WAL mode enabled on connection

All tests use existing mock helpers (with updated imports) and test fixtures.

## Dependencies

**New runtime:**
- `fastapi` — API framework (includes `python-multipart` for file uploads)
- `uvicorn` — ASGI server
- `httpx` — sync HTTP client for Streamlit → API communication

**Running:**
```bash
# Terminal 1: API server
uv run uvicorn api.app:app --port 8000

# Terminal 2: Streamlit UI
uv run streamlit run streamlit_app.py
```

## Configuration

- `API_URL` env var — API base URL (default: `http://localhost:8000`)
- `UPLOAD_DIR` env var — upload directory (default: `uploads/`)
- `RESULT_DIR` env var — results directory (default: `results/`)
- `DATABASE_PATH` env var — SQLite path (default: `data/jobs.db`)

**`.gitignore` additions:** `uploads/`, `results/`, `data/`

## Documentation Updates

After implementation, update:
- `CLAUDE.md` — new architecture section, new commands for running both servers, updated test file list, new dependencies.
- `pyproject.toml` — add `fastapi`, `uvicorn`, `httpx` to dependencies.
- `README.md` — updated setup instructions for two-process startup.

## Migration Notes

Implementation should proceed in phases to keep the app working throughout:
1. Extract `core/` module from `streamlit_app.py` and migrate tests — existing Streamlit app continues to work by importing from `core/`.
2. Build `api/` (database, worker, routes) with tests.
3. Rewrite `streamlit_app.py` as API client.
4. Update documentation and configuration.
