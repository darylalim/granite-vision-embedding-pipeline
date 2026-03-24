# Streamlit UI Tests Design

## Overview

Add unit tests and integration tests for `streamlit_app.py`. Unit tests cover pure helpers (`STATUS_ICONS`, `document_filter`). Integration tests use `streamlit.testing.v1.AppTest` to verify app structure, connection handling, and tab content with mocked HTTP responses.

## Test File

- `tests/test_streamlit_app.py` — all Streamlit UI tests in one file, following the existing pattern of one test file per module.

## Unit Tests

### `TestStatusIcons`
- All 4 statuses present as keys (`completed`, `failed`, `pending`, `processing`)
- Each value is a non-empty string

### `TestDocumentFilter`
- Mock `st.selectbox` to return index 0 → returns `None` (All documents)
- Mock `st.selectbox` to return index 1 → returns the first job's ID
- Empty completed list → options are just `["All documents"]`, returns `None`

## Integration Tests (AppTest)

All integration tests mock `httpx.Client` to avoid real network calls. The mock is applied via `unittest.mock.patch` on `streamlit_app.api_client` or `httpx.Client`.

### `TestConnectionCheck`
- Mock `httpx.Client` so that `GET /health` raises `httpx.ConnectError` → app renders error text containing "Cannot connect", no tabs render

### `TestHealthyAppStructure`
- Mock `httpx.Client` to return healthy `/health` response (`{"device": "cpu", "queue_depth": 0, "worker_running": true}`) and empty `GET /jobs` → verify:
  - Sidebar contains model ID, device (`CPU`), queue depth
  - Three tabs render (Upload, Jobs, Query)

### `TestUploadTab`
- Mock healthy API → verify file uploader widget exists with descriptive label text
- Verify DPI radio widget exists with 3 options

### `TestJobsTab`
- Mock `/jobs` returning empty list → "No jobs found" info message appears
- Mock `/jobs` returning a list of jobs → dataframe widget renders

### `TestQueryTab`
- Mock `/jobs?status=completed` returning empty → "No documents available" info message appears
- Mock `/jobs?status=completed` returning completed jobs → document filter selectbox and query text input render

## What Does Not Change

- Existing test files (`test_core.py`, `test_generation.py`, `test_database.py`, `test_worker.py`, `test_api.py`) — no modifications.
- No changes to `api/`, `core/`, or `streamlit_app.py`.

## File Changes

- Create: `tests/test_streamlit_app.py`
- Modify: `CLAUDE.md` (add test file to Tests section)
