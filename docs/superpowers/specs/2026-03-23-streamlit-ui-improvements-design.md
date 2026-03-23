# Streamlit UI Improvements Design

## Overview

Rewrite `streamlit_app.py` to improve usability and friendliness. The app remains a single file. All 16 improvements from the proposal are included.

## Layout & Navigation

- Keep `page_title="Granite Vision Embedding Pipeline"`, change to `layout="wide"`.
- Move API health status (device, queue depth, connection state) and model ID into `st.sidebar` using `st.sidebar.caption` lines. Sidebar content loads once on page load (not auto-refreshed).
- Replace the single-scroll layout with `st.tabs(["Upload", "Jobs", "Query"])`.
- Combine the existing Search and Ask sections into one "Query" tab with shared controls.
- On page load, call `/health`. If unreachable, show a persistent `st.error` banner below the title and `st.stop()` — tabs are not rendered at all.

## Upload Tab

- File uploader label: `"Drop PDFs or images here (PNG, JPG, WebP) — up to 50 MB each"`.
- Wrap DPI radio in `st.expander("Advanced options")` with default Medium (150).
- Show file count + total size caption when files are selected.
- Disable Submit button via `st.session_state["uploading"]` flag during the local polling loop to prevent double-submission. This is local state only — not gated on global job status.
- Keep the existing `st.status` + progress bar polling loop for batch progress.
- Use `st.toast` for successful submission confirmation.

## Jobs Tab

- The `@st.fragment(run_every=5)` wraps only the data fetch, dataframe render, and metrics — not the Delete All dialog or detail panel. This prevents auto-refresh from interrupting user interactions.
- Top row: status filter selectbox + Delete All button (outside the fragment).
- Display jobs in `st.dataframe` with columns: Name, Status, Type, DPI, Pages, Duration. Use `selection_mode="single-rows"` and `on_select="rerun"`. When no row is selected, the detail panel is hidden.
- Selected row ID is persisted in `st.session_state["selected_job_id"]` so it survives fragment reruns. If the selected job no longer exists (deleted), the selection is cleared.
- Status column uses colored emoji prefixes: green circle (completed), red circle (failed), yellow circle (pending), blue circle (processing).
- Detail panel below the dataframe (outside the fragment): shows error message (if failed), download JSON button, delete button.
- Metrics row (total duration, total pages, document count) above the table. Metrics always reflect all completed jobs regardless of the status filter.
- Download All button below the metrics, shown when 2+ completed jobs exist.
- `st.toast` for both individual and bulk deletion confirmations.
- Existing `st.dialog` for Delete All confirmation stays.
- Session state cleanup: `search_results` and `ask_result` are cleared on both individual and bulk deletes (preserving current behavior).

## Query Tab

- When no completed jobs exist, show `st.info("No documents available — upload and process files first.")` and skip the rest of the tab.
- Document filter selectbox at the top (single instance, no `key_prefix` needed since Search and Ask share it).
- Advanced options in `st.expander`: Top-K (`help="Number of results to return"`) and Min Score (`help="Filter out low-confidence results"`) side by side.
- Single `st.text_input` for the query, persisted in `st.session_state` across reruns.
- Two buttons side by side: "Search" and "Ask", both using the same query, filter, and advanced settings.
- `st.spinner("Searching...")` / `st.spinner("Generating answer...")` wrapping API calls.
- Search results displayed as structured rows: rank, document name, page number, score.
- Ask results in a bordered `st.container`: answer markdown on top, sources listed below with "Sources:" caption.
- 503 from `/ask` shows a friendly warning that generation is not configured.

## What Does Not Change

- `api_client()` cached via `@st.cache_resource` (single httpx connection pool).
- `document_filter()` helper function (adapted for Query tab — `key_prefix` parameter removed since only one instance is needed).
- All API interactions go through the existing REST endpoints.
- No changes to `api/`, `core/`, or tests.

## File Changes

- `streamlit_app.py` — full rewrite (single file, ~500-600 lines).
