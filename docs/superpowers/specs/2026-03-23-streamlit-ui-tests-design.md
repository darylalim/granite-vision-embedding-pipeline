# Streamlit UI Tests Design

## Overview

Add tests for `streamlit_app.py`. Unit tests cover pure constants (`STATUS_ICONS`) as self-contained assertions. All other tests use `streamlit.testing.v1.AppTest` to verify app structure, connection handling, and tab content with mocked HTTP responses.

**Note:** Importing `streamlit_app` directly outside the Streamlit runtime is not reliable — module-level code calls `st.set_page_config`, `st.tabs`, `st.radio`, etc. that require the Streamlit context. All tests that exercise `streamlit_app.py` must use `AppTest.from_file()`.

## Test File

- `tests/test_streamlit_app.py` — all Streamlit UI tests in one file, following the existing pattern of one test file per module.

## Mocking Strategy

All AppTest tests mock `httpx.Client` via `unittest.mock.patch("httpx.Client")` before calling `AppTest.from_file()`. A single `AppTest.run()` of `streamlit_app.py` makes multiple HTTP calls:

1. `GET /health` — connection check
2. `GET /jobs` — inside `jobs_fragment()` (no params)
3. `GET /jobs?status=completed` — Download All section in Jobs tab
4. `GET /jobs?status=completed` — Query tab completed jobs check

The mock client's `.get()` method should use `side_effect` with a routing function that inspects the URL path and params to return appropriate responses for each call. Example pattern:

```python
def mock_get(url, **kwargs):
    if "health" in url:
        return mock_health_response
    if kwargs.get("params", {}).get("status") == "completed":
        return mock_completed_response
    return mock_all_jobs_response
```

## Unit Tests

### `TestStatusIcons`
- Self-contained assertions (no import of `streamlit_app`) — test against known literal values
- All 4 statuses present as keys (`completed`, `failed`, `pending`, `processing`)
- Each value is a non-empty string

## Integration Tests (AppTest)

### `TestConnectionCheck`
- Mock `httpx.Client` so that `GET /health` raises `httpx.ConnectError` → app renders error text containing "Cannot connect", no tabs render

### `TestHealthyAppStructure`
- Mock healthy `/health` response + empty `/jobs` → verify:
  - Sidebar contains model ID, device (`CPU`), queue depth
  - Three tabs render (Upload, Jobs, Query)

### `TestUploadTab`
- Mock healthy API → verify file uploader widget exists. **Note:** `AppTest` in Streamlit 1.52.2 does not expose `file_uploader` as a named accessor — verify via element tree or `at.get("file_uploader")` returning an element whose label contains the descriptive text.
- Verify DPI radio widget exists with 3 options

### `TestJobsTab`
- Mock `/jobs` returning empty list → "No jobs found" info message appears
- Mock `/jobs` returning a list of jobs → dataframe widget renders

### `TestQueryTab`
- Mock `/jobs?status=completed` returning empty → "No documents available" info message appears
- Mock `/jobs?status=completed` returning completed jobs → document filter selectbox and query text input render

## Out of Scope

- `confirm_delete_all` dialog interaction (requires multi-step AppTest interaction)
- Search/ask result display state (would require pre-setting session state)
- `@st.fragment` auto-refresh behavior (fragment content runs on initial `at.run()` which is sufficient)
- Existing test files — no modifications

## File Changes

- Create: `tests/test_streamlit_app.py`
- Modify: `CLAUDE.md` (add test file to Tests section)
