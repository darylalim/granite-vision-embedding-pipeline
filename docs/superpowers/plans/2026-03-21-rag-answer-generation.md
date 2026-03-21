# RAG Answer Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `POST /ask` endpoint that retrieves relevant document pages, sends them to an external VLM, and returns a grounded answer with source citations.

**Architecture:** Retrieval reuses the existing worker search pipeline. After retrieval, the `/ask` endpoint re-renders only matched pages, builds an OpenAI-compatible VLM request using pure functions in `core/generation.py`, calls the VLM via `httpx.AsyncClient`, and returns the answer with sources.

**Tech Stack:** FastAPI, httpx (AsyncClient), Pydantic, PIL, base64, Streamlit, pytest

**Spec:** `docs/superpowers/specs/2026-03-21-rag-answer-generation-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `core/constants.py` | Modify | Add `GENERATION_MAX_TOKENS` |
| `core/rendering.py` | Modify | Add `render_page()` for single-page rendering |
| `core/generation.py` | Create | Pure functions: `encode_image`, `build_messages` |
| `api/models.py` | Modify | Add `AskRequest`, `AskResponse` |
| `api/app.py` | Modify | Add `POST /ask` endpoint with VLM HTTP call |
| `streamlit_app.py` | Modify | Add "Ask" UI section |
| `tests/test_core.py` | Modify | Add `TestRenderPage` |
| `tests/test_generation.py` | Create | `TestEncodeImage`, `TestBuildMessages` |
| `tests/test_api.py` | Modify | Add `TestAsk` |
| `CLAUDE.md` | Modify | Document new endpoint, env vars, module |

---

### Task 1: Add `GENERATION_MAX_TOKENS` constant with test

**Files:**
- Modify: `core/constants.py:1-4`
- Modify: `tests/test_core.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_core.py`. First, add `GENERATION_MAX_TOKENS` to the import from `core.constants`:

```python
from core.constants import DPI_OPTIONS, GENERATION_MAX_TOKENS, IMAGE_EXTENSIONS, MAX_UPLOAD_BYTES
```

Then add this test class after `TestMaxUploadBytes`:

```python
class TestGenerationMaxTokens:
    def test_equals_1024(self) -> None:
        assert GENERATION_MAX_TOKENS == 1024
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core.py::TestGenerationMaxTokens -v`
Expected: FAIL with "cannot import name 'GENERATION_MAX_TOKENS'"

- [ ] **Step 3: Add the constant**

Append to `core/constants.py`:

```python
GENERATION_MAX_TOKENS = 1024
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_core.py::TestGenerationMaxTokens -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/constants.py tests/test_core.py
git commit -m "feat: add GENERATION_MAX_TOKENS constant"
```

---

### Task 2: Add `render_page()` with tests

**Files:**
- Modify: `core/rendering.py:1-22`
- Modify: `tests/test_core.py`

- [ ] **Step 1: Write failing tests for `render_page`**

Add to `tests/test_core.py`. First, add `render_page` to the import from `core.rendering`:

```python
from core.rendering import render_page, render_pages
```

Then add this test class after `TestRenderPages`:

```python
class TestRenderPage:
    def test_renders_first_page(self) -> None:
        data = (PDF_DATA_DIR / "multi_page.pdf").read_bytes()
        page = render_page(data, page_index=0)
        assert isinstance(page, Image.Image)
        assert page.mode == "RGB"

    def test_renders_last_page(self) -> None:
        data = (PDF_DATA_DIR / "multi_page.pdf").read_bytes()
        page = render_page(data, page_index=2)
        assert isinstance(page, Image.Image)
        assert page.mode == "RGB"

    def test_raises_for_out_of_bounds_index(self) -> None:
        data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        with pytest.raises(IndexError):
            render_page(data, page_index=1)

    def test_raises_for_corrupt_pdf(self) -> None:
        with pytest.raises(ValueError, match="Corrupt or unreadable PDF"):
            render_page(b"not a pdf", page_index=0)

    def test_respects_dpi(self) -> None:
        data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        page_72 = render_page(data, page_index=0, dpi=72)
        page_300 = render_page(data, page_index=0, dpi=300)
        assert page_300.width > page_72.width
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_core.py::TestRenderPage -v`
Expected: FAIL with "cannot import name 'render_page'"

- [ ] **Step 3: Implement `render_page`**

Add to `core/rendering.py` after `render_pages`:

```python
def render_page(data: bytes, page_index: int, dpi: int = 150) -> Image.Image:
    """Render a single PDF page by index as a PIL Image.

    Raises ValueError for corrupt PDF data.
    Raises IndexError if page_index is out of range.
    """
    try:
        scale = dpi / 72
        matrix = fitz.Matrix(scale, scale)
        with fitz.open(stream=data, filetype="pdf") as doc:
            if page_index < 0 or page_index >= len(doc):
                raise IndexError(
                    f"Page index {page_index} out of range for {len(doc)}-page PDF"
                )
            page = doc[page_index]
            pix = page.get_pixmap(matrix=matrix)
            return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    except (fitz.FileDataError, fitz.EmptyFileError):
        raise ValueError("Corrupt or unreadable PDF")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_core.py::TestRenderPage -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `uv run pytest tests/test_core.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add core/rendering.py tests/test_core.py
git commit -m "feat: add render_page() for single-page PDF rendering"
```

---

### Task 3: Create `core/generation.py` with tests

**Files:**
- Create: `core/generation.py`
- Create: `tests/test_generation.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_generation.py`:

```python
import base64
import io

from PIL import Image

from core.generation import build_messages, encode_image


class TestEncodeImage:
    def test_returns_valid_base64_png(self) -> None:
        img = Image.new("RGB", (64, 64), color="red")
        result = encode_image(img)
        decoded = base64.b64decode(result)
        # Verify it's valid PNG by re-loading
        reloaded = Image.open(io.BytesIO(decoded))
        assert reloaded.format == "PNG"

    def test_converts_rgba_to_rgb(self) -> None:
        img = Image.new("RGBA", (64, 64), color=(255, 0, 0, 128))
        result = encode_image(img)
        decoded = base64.b64decode(result)
        reloaded = Image.open(io.BytesIO(decoded))
        assert reloaded.mode == "RGB"

    def test_converts_l_to_rgb(self) -> None:
        img = Image.new("L", (64, 64), color=128)
        result = encode_image(img)
        decoded = base64.b64decode(result)
        reloaded = Image.open(io.BytesIO(decoded))
        assert reloaded.mode == "RGB"


class TestBuildMessages:
    def test_contains_system_and_user_messages(self) -> None:
        img = Image.new("RGB", (64, 64))
        messages = build_messages("What is this?", [img])
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_message_has_grounding_instructions(self) -> None:
        img = Image.new("RGB", (64, 64))
        messages = build_messages("test", [img])
        system_content = messages[0]["content"]
        assert "provided pages" in system_content.lower()

    def test_user_message_contains_query_text(self) -> None:
        img = Image.new("RGB", (64, 64))
        messages = build_messages("What color is this?", [img])
        user_content = messages[1]["content"]
        text_parts = [p for p in user_content if p["type"] == "text"]
        assert any("What color is this?" in p["text"] for p in text_parts)

    def test_user_message_contains_image_parts(self) -> None:
        images = [Image.new("RGB", (64, 64)) for _ in range(3)]
        messages = build_messages("test", images)
        user_content = messages[1]["content"]
        image_parts = [p for p in user_content if p["type"] == "image_url"]
        assert len(image_parts) == 3

    def test_image_urls_are_base64_data_uris(self) -> None:
        img = Image.new("RGB", (64, 64))
        messages = build_messages("test", [img])
        user_content = messages[1]["content"]
        image_part = [p for p in user_content if p["type"] == "image_url"][0]
        url = image_part["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")

    def test_empty_images_list(self) -> None:
        messages = build_messages("test query", [])
        user_content = messages[1]["content"]
        image_parts = [p for p in user_content if p["type"] == "image_url"]
        assert len(image_parts) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_generation.py -v`
Expected: FAIL with "No module named 'core.generation'"

- [ ] **Step 3: Implement `core/generation.py`**

Create `core/generation.py`:

```python
import base64
import io

from PIL import Image


def encode_image(image: Image.Image) -> str:
    """Encode a PIL Image as a base64 PNG string."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_messages(query: str, images: list[Image.Image]) -> list[dict]:
    """Build OpenAI-compatible messages for a VLM request.

    Returns a system message with grounding instructions and a user message
    containing the query text and page images as base64 data URIs.
    """
    system_message = {
        "role": "system",
        "content": (
            "You are answering questions about documents. You are given page "
            "images from the most relevant pages. Answer the question using "
            "only information visible in the provided pages. Cite which "
            "page(s) support your answer. If the pages do not contain enough "
            "information to answer, say so."
        ),
    }

    user_content: list[dict] = [{"type": "text", "text": query}]
    for image in images:
        b64 = encode_image(image)
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )

    user_message = {"role": "user", "content": user_content}
    return [system_message, user_message]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_generation.py -v`
Expected: all 9 tests PASS

- [ ] **Step 5: Lint**

Run: `uv run ruff check core/generation.py tests/test_generation.py`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add core/generation.py tests/test_generation.py
git commit -m "feat: add core/generation.py with encode_image and build_messages"
```

---

### Task 4: Add `AskRequest` and `AskResponse` models

**Files:**
- Modify: `api/models.py:1-43`

- [ ] **Step 1: Add models**

Append to `api/models.py` after `HealthResponse`:

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

- [ ] **Step 2: Run existing tests to verify no regressions**

Run: `uv run pytest tests/test_api.py -v`
Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add api/models.py
git commit -m "feat: add AskRequest and AskResponse Pydantic models"
```

---

### Task 5: Add `POST /ask` endpoint with tests

**Files:**
- Modify: `api/app.py:1-193`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_api.py`. First, update the imports at the top of the file to add `AsyncMock` and `httpx`:

```python
from unittest.mock import AsyncMock, MagicMock, patch
```

```python
import httpx
```

Then add the `TestAsk` class at the end of the file:

```python
class TestAsk:
    def test_returns_503_when_not_configured(self, api: ApiFixture) -> None:
        # Explicitly ensure generation env vars are absent
        with patch.dict("os.environ", {}, clear=False):
            resp = api.client.post("/ask", json={"query": "test"})
        assert resp.status_code == 503

    def test_returns_answer_with_sources(self, api: ApiFixture) -> None:
        from concurrent.futures import Future

        from api.database import update_job

        # Create a completed job with an uploaded file
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = api.client.post(
            "/jobs",
            files={"file": ("test.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )
        job_id = create_resp.json()["job_id"]
        db = api.client.app.state.db
        update_job(
            db,
            job_id,
            status="completed",
            page_count=1,
            duration_ns=100,
            result_path="r.json",
            tensor_path="r.pt",
        )

        # Mock worker search to return this job
        search_future: Future = Future()
        search_future.set_result([(job_id, 0, 0.9)])
        api.mock_worker.enqueue_search.return_value = search_future

        # Mock the VLM HTTP call
        mock_vlm_response = httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "The document shows a test page."}}
                ]
            },
        )

        with (
            patch.dict(
                "os.environ",
                {
                    "GENERATION_API_URL": "http://fake-vlm:8000/v1",
                    "GENERATION_MODEL": "test-model",
                },
            ),
            patch("httpx.AsyncClient") as MockClient,
        ):
            mock_client_instance = AsyncMock()
            mock_client_instance.post.return_value = mock_vlm_response
            MockClient.return_value = mock_client_instance

            resp = api.client.post("/ask", json={"query": "What is this?"})

        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert data["answer"] == "The document shows a test page."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["file_id"] == job_id

    def test_returns_answer_when_no_results(self, api: ApiFixture) -> None:
        from concurrent.futures import Future

        from api.database import update_job

        # Create a completed job, but search returns empty
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = api.client.post(
            "/jobs",
            files={"file": ("test.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )
        job_id = create_resp.json()["job_id"]
        db = api.client.app.state.db
        update_job(
            db,
            job_id,
            status="completed",
            page_count=1,
            duration_ns=100,
            result_path="r.json",
            tensor_path="r.pt",
        )

        # Search returns empty results (no relevant pages)
        search_future: Future = Future()
        search_future.set_result([])
        api.mock_worker.enqueue_search.return_value = search_future

        with patch.dict(
            "os.environ",
            {
                "GENERATION_API_URL": "http://fake-vlm:8000/v1",
                "GENERATION_MODEL": "test-model",
            },
        ):
            resp = api.client.post("/ask", json={"query": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["sources"] == []
        assert len(data["answer"]) > 0

    def test_rejects_invalid_top_k(self, api: ApiFixture) -> None:
        with patch.dict(
            "os.environ",
            {
                "GENERATION_API_URL": "http://fake-vlm:8000/v1",
                "GENERATION_MODEL": "test-model",
            },
        ):
            resp = api.client.post("/ask", json={"query": "test", "top_k": 0})
            assert resp.status_code == 422
            resp = api.client.post("/ask", json={"query": "test", "top_k": 11})
            assert resp.status_code == 422

    def test_returns_502_on_vlm_timeout(self, api: ApiFixture) -> None:
        from concurrent.futures import Future

        from api.database import update_job

        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = api.client.post(
            "/jobs",
            files={"file": ("test.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )
        job_id = create_resp.json()["job_id"]
        db = api.client.app.state.db
        update_job(
            db,
            job_id,
            status="completed",
            page_count=1,
            duration_ns=100,
            result_path="r.json",
            tensor_path="r.pt",
        )

        search_future: Future = Future()
        search_future.set_result([(job_id, 0, 0.9)])
        api.mock_worker.enqueue_search.return_value = search_future

        with (
            patch.dict(
                "os.environ",
                {
                    "GENERATION_API_URL": "http://fake-vlm:8000/v1",
                    "GENERATION_MODEL": "test-model",
                },
            ),
            patch("httpx.AsyncClient") as MockClient,
        ):
            mock_client_instance = AsyncMock()
            mock_client_instance.post.side_effect = httpx.TimeoutException(
                "Connection timed out"
            )
            MockClient.return_value = mock_client_instance

            resp = api.client.post("/ask", json={"query": "What is this?"})

        assert resp.status_code == 502
        assert "Unable to reach" in resp.json()["detail"]

    def test_returns_502_on_vlm_error(self, api: ApiFixture) -> None:
        from concurrent.futures import Future

        from api.database import update_job

        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = api.client.post(
            "/jobs",
            files={"file": ("test.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )
        job_id = create_resp.json()["job_id"]
        db = api.client.app.state.db
        update_job(
            db,
            job_id,
            status="completed",
            page_count=1,
            duration_ns=100,
            result_path="r.json",
            tensor_path="r.pt",
        )

        search_future: Future = Future()
        search_future.set_result([(job_id, 0, 0.9)])
        api.mock_worker.enqueue_search.return_value = search_future

        mock_vlm_response = httpx.Response(500, json={"error": "internal error"})

        with (
            patch.dict(
                "os.environ",
                {
                    "GENERATION_API_URL": "http://fake-vlm:8000/v1",
                    "GENERATION_MODEL": "test-model",
                },
            ),
            patch("httpx.AsyncClient") as MockClient,
        ):
            mock_client_instance = AsyncMock()
            mock_client_instance.post.return_value = mock_vlm_response
            MockClient.return_value = mock_client_instance

            resp = api.client.post("/ask", json={"query": "What is this?"})

        assert resp.status_code == 502
        assert "Generation service error" in resp.json()["detail"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api.py::TestAsk -v`
Expected: FAIL (404 — endpoint does not exist yet)

- [ ] **Step 3: Implement the `POST /ask` endpoint**

In `api/app.py`, add `import httpx` after the existing `import os` (line 2).

Update the existing `from api.models` import (lines 18-25) to also include `AskRequest` and `AskResponse`:

```python
from api.models import (
    AskRequest,
    AskResponse,
    HealthResponse,
    JobCreateResponse,
    JobResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
```

Update the existing `from core.constants` import (line 27) to also include `GENERATION_MAX_TOKENS`:

```python
from core.constants import DPI_OPTIONS, GENERATION_MAX_TOKENS, IMAGE_EXTENSIONS, MAX_UPLOAD_BYTES
```

Update the existing `from core.embedding` import (line 28) to also include `load_image`:

```python
from core.embedding import get_device, load_image
```

Add two new imports after the `core.embedding` line:

```python
from core.generation import build_messages
from core.rendering import render_page
```

Then add the endpoint inside `create_app()`, after the `search_embeddings` route and before `return app`:

```python
    @app.post("/ask", response_model=AskResponse)
    async def ask(req: AskRequest):
        generation_api_url = os.environ.get("GENERATION_API_URL")
        generation_model = os.environ.get("GENERATION_MODEL")
        if not generation_api_url or not generation_model:
            raise HTTPException(
                503,
                detail="Answer generation is not configured. "
                "Set GENERATION_API_URL and GENERATION_MODEL.",
            )

        worker = app.state.worker
        if not worker:
            raise HTTPException(503, detail="Worker not running")

        # Retrieval (reuse search pipeline)
        db = app.state.db
        completed_jobs = list_jobs(db, status="completed")
        job_ids = [j["id"] for j in completed_jobs]

        if not job_ids:
            return AskResponse(
                answer="No documents have been processed yet.", sources=[]
            )

        if req.filter_file_id and req.filter_file_id not in job_ids:
            raise HTTPException(
                400,
                detail=f"filter_file_id '{req.filter_file_id}' is not a completed job",
            )

        params = {
            "query": req.query,
            "top_k": req.top_k,
            "min_score": req.min_score,
            "filter_file_id": req.filter_file_id,
            "job_ids": job_ids,
        }
        future = worker.enqueue_search(params)
        search_results = await asyncio.wrap_future(future)

        if not search_results:
            return AskResponse(
                answer="No relevant pages found for your query.", sources=[]
            )

        # Re-render matched pages
        images = []
        sources = []
        for file_id, page_index, score in search_results:
            job = get_job(db, file_id)
            if not job:
                continue
            file_path = Path(job["file_path"])
            if not file_path.exists():
                continue
            try:
                if job["file_type"] == "image":
                    image = load_image(file_path)
                else:
                    pdf_data = file_path.read_bytes()
                    image = render_page(pdf_data, page_index, dpi=job["dpi"])
                images.append(image)
                sources.append(
                    SearchResult(
                        file_id=file_id, page_index=page_index, score=score
                    )
                )
            except Exception:
                continue

        if not images:
            return AskResponse(
                answer="Could not load any of the matched pages.", sources=[]
            )

        # Call VLM
        messages = build_messages(req.query, images)
        generation_api_key = os.environ.get("GENERATION_API_KEY", "")
        max_tokens = int(
            os.environ.get("GENERATION_MAX_TOKENS", GENERATION_MAX_TOKENS)
        )

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                vlm_resp = await client.post(
                    f"{generation_api_url}/chat/completions",
                    headers={"Authorization": f"Bearer {generation_api_key}"},
                    json={
                        "model": generation_model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                    },
                )
        except (httpx.TimeoutException, httpx.ConnectError):
            raise HTTPException(
                502, detail="Unable to reach generation service."
            )

        if vlm_resp.status_code != 200:
            raise HTTPException(502, detail="Generation service error.")

        try:
            answer = vlm_resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise HTTPException(
                502, detail="Unexpected response from generation service."
            )

        return AskResponse(answer=answer, sources=sources)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py::TestAsk -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Run full API test suite**

Run: `uv run pytest tests/test_api.py -v`
Expected: all PASS

- [ ] **Step 6: Lint**

Run: `uv run ruff check api/app.py`
Expected: clean

- [ ] **Step 7: Commit**

```bash
git add api/app.py tests/test_api.py
git commit -m "feat: add POST /ask endpoint for RAG answer generation"
```

---

### Task 6: Add "Ask" section to Streamlit UI

**Files:**
- Modify: `streamlit_app.py:153-206`

- [ ] **Step 1: Add the Ask section**

Insert the following in `streamlit_app.py` after line 206 (`st.info("No results above the score threshold.")`), still inside the `if completed:` block (which is inside `if jobs:`). The indentation must be 8 spaces (two levels: `if jobs:` → `if completed:`), matching the Search section above it:

```python
        # Ask
        st.subheader("Ask")
        ask_filter_options = ["All documents"] + [j["file_stem"] for j in completed]
        ask_filter_ids: list[str | None] = [None] + [j["id"] for j in completed]
        ask_filter_idx = st.selectbox(
            "Document filter",
            range(len(ask_filter_options)),
            format_func=lambda i: ask_filter_options[i],
            key="ask_filter",
        )

        ask_col_topk, ask_col_minscore = st.columns(2)
        ask_top_k = ask_col_topk.number_input(
            "Top K", min_value=1, max_value=10, value=3, key="ask_top_k"
        )
        ask_min_score = ask_col_minscore.number_input(
            "Min score", min_value=0.0, value=0.0, step=0.1, key="ask_min_score"
        )

        ask_query = st.text_input("Question", key="ask_query")
        if st.button("Ask", key="ask_button"):
            if not ask_query:
                st.warning("Enter a question.")
            else:
                try:
                    with api_client() as client:
                        ask_resp = client.post(
                            "/ask",
                            json={
                                "query": ask_query,
                                "top_k": ask_top_k,
                                "min_score": ask_min_score,
                                "filter_file_id": ask_filter_ids[ask_filter_idx],
                            },
                        )
                        if ask_resp.status_code == 200:
                            st.session_state.ask_result = ask_resp.json()
                        elif ask_resp.status_code == 503:
                            st.warning(
                                "Answer generation is not configured. "
                                "Set GENERATION_API_URL and GENERATION_MODEL to enable."
                            )
                        else:
                            st.error(
                                ask_resp.json().get("detail", "Ask failed")
                            )
                except httpx.HTTPError as e:
                    st.error(str(e))

        if "ask_result" in st.session_state:
            ask_result = st.session_state.ask_result
            st.markdown(ask_result["answer"])
            if ask_result["sources"]:
                st.caption("Sources:")
                job_lookup = {j["id"]: j for j in completed}
                for sr in ask_result["sources"]:
                    j = job_lookup.get(sr["file_id"])
                    if j:
                        st.caption(
                            f"  {j['file_stem']} · Page {sr['page_index'] + 1} · {sr['score']:.4f}"
                        )
```

Also update the delete button handler to clear the ask result when a job is deleted. Find line 127:

```python
                        st.session_state.pop("search_results", None)
```

Add `st.session_state.pop("ask_result", None)` immediately after it (before `st.rerun()` on line 128), at the same indentation level (24 spaces).

- [ ] **Step 2: Lint**

Run: `uv run ruff check streamlit_app.py`
Expected: clean

- [ ] **Step 3: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add Ask section to Streamlit UI"
```

---

### Task 7: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md**

Add these changes to `CLAUDE.md`:

1. In the **Configuration** table, add after the `DATABASE_PATH` row:

```
| `GENERATION_API_URL` | None | VLM endpoint for answer generation |
| `GENERATION_API_KEY` | `""` | VLM auth token |
| `GENERATION_MODEL` | None | VLM model ID |
| `GENERATION_MAX_TOKENS` | `1024` | Max VLM response tokens |
```

2. In **Core Module (`core/`)**, update the `core/rendering.py` line:

```
- `core/rendering.py` — `render_pages`, `render_page`
```

And add after `core/search.py`:

```
- `core/generation.py` — `encode_image`, `build_messages`
```

3. In **Constants**, add:

```
- `GENERATION_MAX_TOKENS = 1024`
```

4. In **API Routes**, add before the `/health` line:

```
POST   /ask              Text query + retrieval, returns VLM-generated answer (503 if not configured)
```

5. In **Error Handling**, add:

```
- VLM not configured → 503
- VLM unreachable/timeout → 502
- VLM error response → 502
```

6. In **Tests**, update the `test_core.py` line:

```
- `tests/test_core.py` — core functions: `TestDpiOptions`, `TestImageExtensions`, `TestMaxUploadBytes`, `TestGenerationMaxTokens`, `TestLoadImage`, `TestGetDevice`, `TestRenderPages`, `TestRenderPage`, `TestEmbed`, `TestFilterResults`, `TestSearchMulti`
```

Add after the `test_core.py` line:

```
- `tests/test_generation.py` — generation functions: `TestEncodeImage`, `TestBuildMessages`
```

Update the `test_api.py` line:

```
- `tests/test_api.py` — FastAPI routes: `TestHealth`, `TestUploadJob`, `TestListJobs`, `TestGetJob`, `TestDeleteJob`, `TestGetResult`, `TestSearch`, `TestAsk`
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for RAG answer generation feature"
```

---

### Task 8: Final verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: all tests PASS

- [ ] **Step 2: Lint and format**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: clean

- [ ] **Step 3: Type check**

Run: `uv run ty check`
Expected: clean (or only pre-existing issues)
