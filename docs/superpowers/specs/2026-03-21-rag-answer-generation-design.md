# RAG Answer Generation Design

## Overview

Add a `POST /ask` endpoint that retrieves relevant document pages using the existing embedding search pipeline, sends those page images to an external OpenAI-compatible VLM API, and returns a grounded answer with source citations. This closes the RAG loop — the project currently stops at retrieval.

## Data Flow

```
POST /ask { query, top_k, min_score, filter_file_id }
    ↓
Enqueue retrieval to worker (reuse existing _execute_search)
    ↓
Top-K results: [(file_id, page_index, score), ...]
    ↓
Look up each result's job in DB → file_path, file_type, dpi
    ↓
Render only matched pages: render_page() / load_image()
    ↓
Encode page images as base64 PNG
    ↓
Build VLM request: system prompt + query + page images
    ↓
POST to OpenAI-compatible /chat/completions endpoint
    ↓
Return AskResponse { answer, sources }
```

Retrieval reuses the existing worker search pipeline unchanged. Everything after retrieval (re-rendering, VLM call) runs directly in the async endpoint handler.

## New Function in `core/rendering.py`

### `render_page(data: bytes, page_index: int, dpi: int = 150) -> Image.Image`

Renders a single page from a PDF by index. The existing `render_pages()` renders all pages, which is appropriate for the embedding pipeline but wasteful for `/ask` where only 3-5 specific pages are needed from potentially large documents. This function opens the PDF, renders only the requested page, and returns it as a PIL Image.

## New Module: `core/generation.py`

Pure functions for building VLM request payloads. No HTTP calls — consistent with the existing `core/` convention of pure logic with no I/O or framework dependencies. The actual HTTP call to the VLM lives in the `/ask` endpoint handler in `api/app.py`.

### `encode_image(image: Image.Image) -> str`

Takes a PIL Image, encodes it as a base64 PNG string for the VLM request payload.

### `build_messages(query: str, images: list[Image.Image]) -> list[dict]`

Constructs the OpenAI-compatible messages array:

- System message with grounding instructions: answer using only the provided pages, state when information is not present.
- User message containing the text query and page images as base64 `image_url` content parts.

## API Changes

### New Pydantic Models (`api/models.py`)

```python
class AskRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=10)
    min_score: float = Field(default=0.0, ge=0.0)
    filter_file_id: str | None = None

class AskResponse(BaseModel):
    answer: str
    sources: list[SearchResult]
```

`top_k` defaults to 3 (not 5 like search) to keep VLM context focused and reduce latency/cost. Capped at 10 because each result page is sent as a base64 image to the VLM — higher values risk exceeding context limits and causing excessive latency.

`AskRequest` intentionally duplicates `SearchRequest` fields rather than sharing a base model, since the two may diverge (different defaults, different constraints like the `top_k` cap).

### New Endpoint (`api/app.py`): `POST /ask`

1. Check if VLM is configured (`GENERATION_API_URL` and `GENERATION_MODEL` env vars). If either is missing, return 503. Both are required — some providers accept empty model strings, but this is fragile and leads to opaque errors.
2. Run retrieval: enqueue search to worker, await results (same as `/search`).
3. If no results, return an answer stating no relevant pages were found, with empty sources.
4. Look up each result's job in DB to get `file_path`, `file_type`, `dpi`.
5. Render only the matched pages: `render_page(data, page_index, dpi)` for PDFs, `load_image()` for images. Skip pages that fail to render (missing/corrupt file) and proceed with remaining pages.
6. Build VLM messages using `build_messages()` from `core/generation.py`.
7. Call the VLM endpoint via `httpx.AsyncClient` directly in the async handler (no `asyncio.to_thread` needed — `httpx.AsyncClient` is natively async and appropriate for I/O-bound HTTP calls in FastAPI).
8. Return `AskResponse` with the answer and the retrieval sources.

`max_tokens` is server-side only (from env var / constant), not exposed per-request in `AskRequest`. This keeps the API simple — callers control what pages to search, not how the VLM generates.

### New Constant (`core/constants.py`)

```python
GENERATION_MAX_TOKENS = 1024
```

Default max response tokens. API URL, key, and model ID come from env vars since they are deployment-specific.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GENERATION_API_URL` | None | VLM endpoint (e.g., `http://localhost:11434/v1`) |
| `GENERATION_API_KEY` | `""` | Auth token (empty string for local models like Ollama) |
| `GENERATION_MODEL` | None | Model ID for the generative VLM (required alongside `GENERATION_API_URL`) |
| `GENERATION_MAX_TOKENS` | `1024` | Max response tokens |

## Streamlit UI Changes

Add an "Ask" section below the existing "Search" section in `streamlit_app.py`. Only shown when there are completed jobs (same condition as Search).

Controls mirror the Search section: document filter, top K (default 3, max 10), min score. The difference is the response — a generated answer followed by source pages, rather than just a list of scores.

Sources display uses the same pattern as Search: the `completed` jobs list is already available in the Streamlit context, so `file_id` is mapped to `file_stem` via a lookup dict (same as `streamlit_app.py` line 198).

If the API returns 503 (VLM not configured), show a warning: "Answer generation is not configured. Set GENERATION_API_URL and GENERATION_MODEL to enable."

The existing Search section stays unchanged.

## Error Handling

**VLM API errors:**

- Connection refused / timeout: return 502 with "Unable to reach generation service."
- Non-200 response: return 502 with "Generation service error."
- Malformed JSON / missing `choices`: return 502 with "Unexpected response from generation service."

**Page re-rendering errors:**

- Uploaded file deleted from disk: skip that page, proceed with remaining pages. If no pages can be loaded, return a "no relevant pages" answer.
- Corrupt file during re-render: same behavior, skip and proceed.

**Timeouts:**

- `httpx.AsyncClient` in the `/ask` endpoint uses a configurable timeout (default 120s). Long enough for large context windows with slow models.

**Streamlit:**

- VLM call failures shown via `st.error()` with the error detail, consistent with existing error display patterns.

No retries. Surface errors and let the user retry manually.

## Testing

### `tests/test_generation.py` (new file)

- `TestEncodeImage` — verifies base64 output is valid PNG data, handles different image modes.
- `TestBuildMessages` — checks message structure: system message present, user message contains query text and image content parts, correct number of images.
- `TestGenerate` — tests are not needed in `test_generation.py` since the HTTP call moved to `api/app.py`. Generation-related HTTP tests live in `TestAsk` in `test_api.py`.

### `tests/test_api.py` (additions)

- `TestAsk`:
  - `test_returns_503_when_not_configured` — no `GENERATION_API_URL` set.
  - `test_returns_answer_with_sources` — mock VLM HTTP call, create a completed job, verify response structure.
  - `test_returns_answer_when_no_results` — query with no matching pages returns answer with empty sources.
  - `test_rejects_invalid_top_k` — validation (ge=1, le=10).
  - `test_returns_502_on_vlm_timeout` — VLM call times out, returns 502.
  - `test_returns_502_on_vlm_error` — VLM returns non-200, returns 502.

All tests mock the VLM HTTP call. No real API calls in tests.

### `tests/test_core.py` (additions)

- `TestRenderPage` — tests for the new `render_page()` function: renders correct page by index, raises for out-of-bounds index, raises for corrupt PDF.

## Files Changed

**New files:**

- `core/generation.py` — pure functions: `encode_image`, `build_messages`
- `tests/test_generation.py` — unit tests for generation module

**Modified files:**

- `core/constants.py` — add `GENERATION_MAX_TOKENS`
- `core/rendering.py` — add `render_page()` for single-page rendering
- `api/models.py` — add `AskRequest`, `AskResponse`
- `api/app.py` — add `POST /ask` endpoint (includes VLM HTTP call via `httpx.AsyncClient`)
- `streamlit_app.py` — add "Ask" UI section
- `tests/test_core.py` — add `TestRenderPage` class
- `tests/test_api.py` — add `TestAsk` class
- `CLAUDE.md` — document new endpoint, env vars, module

**No changes to:**

- Embedding pipeline, worker thread, database schema, existing search endpoint
- `pyproject.toml` — no new dependencies (httpx already present)

## Key Decisions

- **OpenAI-compatible API format** — works with OpenAI, Ollama, vLLM, LiteLLM, and most hosted providers.
- **Non-streaming responses** — simpler implementation, easier to test. Streaming can be added later.
- **Render only matched pages** — new `render_page()` function renders individual pages by index, avoiding the cost of rendering all pages from large PDFs.
- **Sources from retrieval results** — no parsed citations from VLM output. Retrieval results already identify exactly which pages informed the answer.
- **503 when VLM not configured** — endpoint always registered, returns 503 if `GENERATION_API_URL` or `GENERATION_MODEL` is missing.
- **Generation inline in endpoint** — VLM call is an HTTP request (not GPU-bound), so it runs via `httpx.AsyncClient` directly in the async handler rather than going through the worker queue.
- **`core/generation.py` stays pure** — only `encode_image` and `build_messages` (no I/O). The HTTP call lives in `api/app.py`, preserving the `core/` convention of framework-agnostic pure logic.
