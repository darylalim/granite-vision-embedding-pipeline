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
Re-render matched pages: render_pages() / load_image()
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

## New Module: `core/generation.py`

Framework-agnostic module (no FastAPI imports), consistent with existing `core/` convention. Three functions:

### `encode_image(image: Image.Image) -> str`

Takes a PIL Image, encodes it as a base64 PNG string for the VLM request payload.

### `build_messages(query: str, images: list[Image.Image]) -> list[dict]`

Constructs the OpenAI-compatible messages array:

- System message with grounding instructions: answer using only the provided pages, state when information is not present.
- User message containing the text query and page images as base64 `image_url` content parts.

### `generate(query: str, images: list[Image.Image], *, api_url: str, api_key: str, model: str, max_tokens: int) -> str`

Calls the VLM endpoint using `httpx.Client` (synchronous — called from the async endpoint via `asyncio.to_thread` to avoid blocking the event loop). Returns the generated answer text. Raises on HTTP errors or missing config.

## API Changes

### New Pydantic Models (`api/models.py`)

```python
class AskRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1)
    min_score: float = Field(default=0.0, ge=0.0)
    filter_file_id: str | None = None

class AskResponse(BaseModel):
    answer: str
    sources: list[SearchResult]
```

`top_k` defaults to 3 (not 5 like search) to keep VLM context focused and reduce latency/cost.

### New Endpoint (`api/app.py`): `POST /ask`

1. Check if VLM is configured (`GENERATION_API_URL` env var). If not, return 503.
2. Run retrieval: enqueue search to worker, await results (same as `/search`).
3. If no results, return an answer stating no relevant pages were found, with empty sources.
4. Look up each result's job in DB to get `file_path`, `file_type`, `dpi`.
5. Re-render matched pages using `render_pages()` / `load_image()`.
6. Call `generate()` via `asyncio.to_thread()`.
7. Return `AskResponse` with the answer and the retrieval sources.

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
| `GENERATION_MODEL` | `""` | Model ID for the generative VLM |
| `GENERATION_MAX_TOKENS` | `1024` | Max response tokens |

## Streamlit UI Changes

Add an "Ask" section below the existing "Search" section in `streamlit_app.py`. Only shown when there are completed jobs (same condition as Search).

Controls mirror the Search section: document filter, top K (default 3), min score. The difference is the response — a generated answer followed by source pages, rather than just a list of scores.

If the API returns 503 (VLM not configured), show a warning: "Answer generation is not configured. Set GENERATION_API_URL to enable."

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

- `httpx.Client` in `generate()` uses a configurable timeout (default 120s). Long enough for large context windows with slow models.

**Streamlit:**

- VLM call failures shown via `st.error()` with the error detail, consistent with existing error display patterns.

No retries. Surface errors and let the user retry manually.

## Testing

### `tests/test_generation.py` (new file)

- `TestEncodeImage` — verifies base64 output is valid PNG data, handles different image modes.
- `TestBuildMessages` — checks message structure: system message present, user message contains query text and image content parts, correct number of images.
- `TestGenerate` — mocks `httpx.Client.post` to verify correct request payload, answer extraction from response, HTTP error handling, and missing API URL behavior.

### `tests/test_api.py` (additions)

- `TestAsk`:
  - `test_returns_503_when_not_configured` — no `GENERATION_API_URL` set.
  - `test_returns_answer_with_sources` — mock VLM HTTP call, create a completed job, verify response structure.
  - `test_returns_answer_when_no_results` — query with no matching pages returns answer with empty sources.
  - `test_rejects_invalid_top_k` — validation.

All tests mock the VLM HTTP call. No real API calls in tests.

## Files Changed

**New files:**

- `core/generation.py` — VLM client: `encode_image`, `build_messages`, `generate`
- `tests/test_generation.py` — unit tests for generation module

**Modified files:**

- `core/constants.py` — add `GENERATION_MAX_TOKENS`
- `api/models.py` — add `AskRequest`, `AskResponse`
- `api/app.py` — add `POST /ask` endpoint
- `streamlit_app.py` — add "Ask" UI section
- `tests/test_api.py` — add `TestAsk` class
- `CLAUDE.md` — document new endpoint, env vars, module

**No changes to:**

- Embedding pipeline, worker thread, database schema, existing search endpoint
- `pyproject.toml` — no new dependencies (httpx already present)

## Key Decisions

- **OpenAI-compatible API format** — works with OpenAI, Ollama, vLLM, LiteLLM, and most hosted providers.
- **Non-streaming responses** — simpler implementation, easier to test. Streaming can be added later.
- **Re-render pages on demand** — no cached page images, no storage overhead, minimal latency for 3-5 pages.
- **Sources from retrieval results** — no parsed citations from VLM output. Retrieval results already identify exactly which pages informed the answer.
- **503 when VLM not configured** — endpoint always registered, clear error when generation is unavailable.
- **Generation inline in endpoint** — VLM call is an HTTP request (not GPU-bound), so it runs via `asyncio.to_thread` rather than going through the worker queue.
