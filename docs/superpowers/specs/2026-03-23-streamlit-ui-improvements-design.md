# Streamlit UI Improvements Design

## Overview

Rewrite `streamlit_app.py` to improve usability and friendliness. The app remains a single file. All 16 improvements from the proposal are included.

## Layout & Navigation

- Change page config to `layout="wide"`.
- Move API health status (device, queue depth, connection state) and model ID into `st.sidebar` so they are always visible.
- Replace the single-scroll layout with `st.tabs(["Upload", "Jobs", "Query"])`.
- Combine the existing Search and Ask sections into one "Query" tab with shared controls.

## Upload Tab

- File uploader label: `"Drop PDFs or images here (PNG, JPG, WebP) — up to 50 MB each"`.
- Wrap DPI radio in `st.expander("Advanced options")` with default Medium (150).
- Show file count + total size caption when files are selected.
- Disable Submit button via `st.session_state` while jobs are processing to prevent double-submission.
- Keep the existing `st.status` + progress bar polling loop for batch progress.
- Use `st.toast` for successful submission confirmation.

## Jobs Tab

- Auto-refresh the job list using `@st.fragment(run_every=5)` instead of a manual Refresh button.
- Top row: status filter selectbox + Delete All button.
- Display jobs in `st.dataframe` with columns: Name, Status, Type, DPI, Pages, Duration.
- Status column uses colored emoji prefixes: green circle (completed), red circle (failed), yellow circle (pending), blue circle (processing).
- Row selection on the dataframe reveals a detail panel below with: error message (if failed), download JSON button, delete button.
- Download All button shown when 2+ completed jobs exist.
- Metrics row (total duration, total pages, document count) above the table when completed jobs exist.
- `st.toast` for deletion confirmations.
- Existing `st.dialog` for Delete All confirmation stays.

## Query Tab

- Document filter selectbox at the top, shared between search and ask.
- Advanced options in `st.expander`: Top-K (`help="Number of results to return"`) and Min Score (`help="Filter out low-confidence results"`) side by side.
- Single `st.text_input` for the query, persisted in `st.session_state` across reruns.
- Two buttons side by side: "Search" and "Ask", both using the same query, filter, and advanced settings.
- `st.spinner("Searching...")` / `st.spinner("Generating answer...")` wrapping API calls.
- Search results displayed as structured rows: rank, document name, page number, score.
- Ask results in a bordered `st.container`: answer markdown on top, sources listed below with "Sources:" caption.
- 503 from `/ask` shows a friendly warning that generation is not configured.

## Error Handling & Connection

- On page load, call `/health`. If unreachable, show a persistent `st.error` banner and skip all tab content (early return).
- `st.toast` for transient success feedback (uploads, deletions).
- Inline `st.error` for actual failures (API errors, validation) that need to persist.

## What Does Not Change

- `api_client()` cached via `@st.cache_resource` (single httpx connection pool).
- `document_filter()` helper function (reused in Query tab).
- All API interactions go through the existing REST endpoints.
- No changes to `api/`, `core/`, or tests.

## File Changes

- `streamlit_app.py` — full rewrite (single file, ~500-600 lines).
