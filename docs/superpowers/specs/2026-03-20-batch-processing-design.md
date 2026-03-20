# Batch Processing Design

## Goal

Add batch processing support for thousands of documents by introducing a FastAPI backend with a job queue, replacing the current synchronous in-process embedding flow.

## Architecture

```
Streamlit UI  →  FastAPI Backend  →  Embedding Worker (ProcessPoolExecutor)
                      ↕
                   SQLite DB
                      ↕
                 File Storage (uploads/ , results/)
```

- **Streamlit** handles file upload, job submission, progress display, search, and download — communicates with FastAPI via HTTP.
- **FastAPI** exposes a REST API for job CRUD, file upload, and search — manages SQLite for job state and coordinates the process pool.
- **Embedding worker** runs in a child process via `ProcessPoolExecutor(max_workers=1)` to keep one model loaded on GPU, processes jobs sequentially from a queue.
- **SQLite** tracks job metadata (status, timestamps, page counts, durations, errors).
- **File storage** — uploaded files saved to `uploads/`, embedding results written as JSON to `results/`.

The existing core functions (`render_pages`, `embed`, `search_multi`, `filter_results`) move into a shared `core/` module used by both the worker and the API.

## Project Structure

```
granite-vision-embedding-pipeline/
├── streamlit_app.py              # Streamlit UI (rewritten as API client)
├── api/
│   ├── __init__.py
│   ├── app.py                    # FastAPI application and routes
│   ├── models.py                 # Pydantic request/response models
│   ├── database.py               # SQLite connection, schema, queries
│   └── worker.py                 # ProcessPoolExecutor, job processing loop
├── core/
│   ├── __init__.py
│   ├── embedding.py              # get_device, load_model, embed
│   ├── rendering.py              # render_pages
│   └── search.py                 # search, search_multi, filter_results
├── tests/
│   ├── test_core.py              # Unit tests for core functions (migrated)
│   ├── test_api.py               # API route tests (FastAPI TestClient)
│   ├── test_worker.py            # Worker/job processing tests
│   ├── test_database.py          # SQLite query tests
│   └── data/                     # Existing fixtures (unchanged)
├── uploads/                      # Uploaded files (gitignored)
├── results/                      # Embedding JSON output (gitignored)
├── data/                         # SQLite database (gitignored)
├── pyproject.toml
└── CLAUDE.md
```

- `core/` holds pure logic extracted from the current `streamlit_app.py` — no Streamlit or FastAPI dependencies, testable in isolation.
- `api/` holds everything backend — routes, database, worker.
- `streamlit_app.py` stays as the entry point but becomes a thin API client.
- Existing test logic migrates to `test_core.py`; new test files cover the API, worker, and database layers.

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
    result_path   TEXT,              -- path in results/ (JSON)
    error         TEXT               -- error message if failed
);
```

### Job Lifecycle

```
pending → processing → completed
                    ↘ failed
```

1. **Upload** — Streamlit sends file to FastAPI, file saved to `uploads/`, job row inserted as `pending`.
2. **Queue** — Worker picks up `pending` jobs in FIFO order (by `created_at`), sets to `processing`.
3. **Process** — Worker renders pages (if PDF), embeds, writes JSON to `results/`, sets to `completed` with `page_count`, `duration_ns`, `result_path`.
4. **Failure** — Any exception during processing sets status to `failed` with `error` message.
5. **Poll** — Streamlit polls `GET /jobs` to update progress display.

No retry logic — failed jobs can be re-submitted.

## API Routes

```
POST   /jobs/upload      Upload file + DPI → save file, create job, return job ID
GET    /jobs              List all jobs (with status filter query param)
GET    /jobs/{id}         Get single job status and metadata
DELETE /jobs/{id}         Cancel pending job or delete completed job (removes files)
GET    /jobs/{id}/result  Download embedding JSON from results/
POST   /search            Text query + top_k + min_score + optional file filter → ranked results
GET    /health            Worker status, job queue depth, device info
```

- `POST /jobs/upload` accepts `multipart/form-data` (file + DPI param). Validates file type against `IMAGE_EXTENSIONS` + `pdf`. Returns `{job_id, status: "pending"}`.
- `GET /jobs` supports `?status=pending,processing` for filtering. Returns list sorted by `created_at`.
- `POST /search` loads result JSONs for completed jobs, runs `search_multi` across them, applies `filter_results`. Search requests are dispatched to the worker process pool to reuse the loaded model.
- `DELETE /jobs/{id}` removes the upload file, result file, and database row. Only allowed for `pending`, `completed`, or `failed` jobs — not `processing`.

## Worker Design

The worker runs as a `ProcessPoolExecutor(max_workers=1)` managed by the FastAPI app's lifespan.

**Why a single worker process:** The embedding model consumes significant GPU memory — only one instance should be loaded. Jobs process sequentially, which is predictable and avoids GPU contention. The child process loads the model once at startup and reuses it for all jobs.

**Processing loop:** FastAPI submits jobs to the executor via `executor.submit(process_job, job_id)`. The worker function `process_job` handles the full pipeline: read file from `uploads/`, render pages if PDF, embed, write JSON to `results/`, update SQLite row.

**Search through the worker:** Search also runs in the worker process so it can reuse the loaded model. `executor.submit(run_search, query, params)` returns results via the future.

**Startup recovery:** On API startup, any jobs left as `processing` (from a previous crash) are reset to `pending` so they get reprocessed.

**Shutdown:** FastAPI lifespan `shutdown` calls `executor.shutdown(wait=True)` to let the current job finish.

## Streamlit UI Changes

The Streamlit app becomes a thin client that talks to FastAPI.

**Upload & Submit:** File uploader remains multi-file (PDFs + images), DPI selector stays. "Embed" button sends each file to `POST /jobs/upload` individually, collects job IDs stored in `st.session_state.job_ids`.

**Progress Dashboard:** Replaces the current single progress bar. Polls `GET /jobs` on a short interval or via manual refresh. Displays a table of jobs with filename, status (color indicators), page count, and duration.

**Results & Download:** Per-document expanders remain — fetch result JSON via `GET /jobs/{id}/result`. "Download All" concatenates results from all completed jobs.

**Search:** Search form unchanged (query, document filter, top-K, min score). Submits to `POST /search`, displays ranked results as before.

**Job Management:** Delete button per job (calls `DELETE /jobs/{id}`). Status filter (all / pending / processing / completed / failed).

No embedding logic runs in the Streamlit process. All heavy work goes through the API.

## Testing Strategy

**`test_core.py`** — migrated from current `test_app.py`. All existing test classes move here unchanged. Imports shift from `streamlit_app` to `core.embedding`, `core.rendering`, `core.search`.

**`test_api.py`** — FastAPI route tests using `TestClient`:
- Upload valid PDF/image, invalid file type rejection
- List jobs with status filter, get single job
- Delete job (allowed for pending/completed/failed, rejected for processing)
- Download result for completed job, 404 for pending/failed
- Search across completed jobs
- Health endpoint

**`test_worker.py`** — worker logic tests with mocked model:
- `process_job` with valid PDF/image → result written, status completed
- `process_job` with corrupt file → status failed with error
- Startup recovery resets stale `processing` jobs
- Search dispatched through worker returns ranked results

**`test_database.py`** — SQLite query tests:
- Job CRUD operations, status transitions, filtering, ordering

All tests use existing mock helpers and test fixtures.

## Dependencies

**New runtime:**
- `fastapi` — API framework
- `uvicorn` — ASGI server
- `httpx` — HTTP client for Streamlit → API communication
- `python-multipart` — file upload handling

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
