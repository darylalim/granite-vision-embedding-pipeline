# Batch Processing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add batch processing for thousands of documents via a FastAPI backend with SQLite job queue and background worker thread, keeping the Streamlit UI as a thin API client.

**Architecture:** FastAPI backend with a single background worker thread that loads the Granite Vision model once and processes jobs sequentially from SQLite. Search requests are serialized on the same thread via `queue.Queue` + `Future`. Streamlit communicates with the API via `httpx.Client`.

**Tech Stack:** FastAPI, uvicorn, httpx, SQLite (WAL mode), torch, transformers, Streamlit

**Spec:** `docs/superpowers/specs/2026-03-20-batch-processing-design.md`

---

## Phase 1: Extract Core Module

Extract pure logic from `streamlit_app.py` into `core/` with no Streamlit or FastAPI dependencies. The existing Streamlit app continues to work by importing from `core/`.

### Task 1: Create `core/constants.py`

**Files:**
- Create: `core/__init__.py`
- Create: `core/constants.py`

- [ ] **Step 1: Create `core/__init__.py`**

```python
# empty file
```

- [ ] **Step 2: Create `core/constants.py`**

```python
MODEL_ID = "ibm-granite/granite-vision-3.3-2b-embedding"
DPI_OPTIONS = {"Low (72)": 72, "Medium (150)": 150, "High (300)": 300}
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
```

- [ ] **Step 3: Verify import works**

Run: `uv run python -c "from core.constants import MODEL_ID, DPI_OPTIONS, IMAGE_EXTENSIONS, MAX_UPLOAD_BYTES; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add core/__init__.py core/constants.py
git commit -m "feat: create core/constants.py with shared constants"
```

---

### Task 2: Create `core/types.py`

**Files:**
- Create: `core/types.py`

- [ ] **Step 1: Create `core/types.py`**

```python
from typing import Any, Protocol

import torch
from PIL import Image


class EmbeddingProcessor(Protocol):
    def process_images(self, images: list[Image.Image]) -> dict[str, Any]: ...
    def process_queries(self, queries: list[str]) -> dict[str, Any]: ...
    def score(
        self, qs: torch.Tensor, ps: torch.Tensor, *, device: str
    ) -> torch.Tensor: ...
```

- [ ] **Step 2: Verify import works**

Run: `uv run python -c "from core.types import EmbeddingProcessor; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add core/types.py
git commit -m "feat: create core/types.py with EmbeddingProcessor protocol"
```

---

### Task 3: Create `core/rendering.py`

**Files:**
- Create: `core/rendering.py`

- [ ] **Step 1: Create `core/rendering.py`**

Extract `render_pages` from `streamlit_app.py:56-68`:

```python
import fitz
from PIL import Image


def render_pages(data: bytes, dpi: int = 150) -> list[Image.Image]:
    """Render PDF pages as PIL Images."""
    try:
        scale = dpi / 72
        matrix = fitz.Matrix(scale, scale)
        with fitz.open(stream=data, filetype="pdf") as doc:
            return [
                Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                for page in doc
                for pix in [page.get_pixmap(matrix=matrix)]
            ]
    except (fitz.FileDataError, fitz.EmptyFileError):
        return []
```

- [ ] **Step 2: Verify import works**

Run: `uv run python -c "from core.rendering import render_pages; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add core/rendering.py
git commit -m "feat: extract render_pages into core/rendering.py"
```

---

### Task 4: Create `core/embedding.py`

**Files:**
- Create: `core/embedding.py`

- [ ] **Step 1: Create `core/embedding.py`**

Extract `get_device`, `load_model`, `embed` from `streamlit_app.py:34-82`, and add new `load_image`. Remove `@st.cache_resource` from `load_model`.

```python
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from core.constants import MODEL_ID
from core.types import EmbeddingProcessor


def get_device() -> str:
    """Detect best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(device: str) -> tuple[Any, Any]:
    """Load embedding model and processor."""
    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model, processor


def load_image(path: Path) -> Image.Image:
    """Load an image file and convert to RGB.

    Lets UnidentifiedImageError and OSError propagate to caller.
    """
    return Image.open(path).convert("RGB")


def embed(
    images: list[Image.Image], model: torch.nn.Module, processor: EmbeddingProcessor
) -> torch.Tensor:
    """Generate per-page multi-vector embeddings from images."""
    batch = processor.process_images(images)
    batch = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    with torch.inference_mode():
        embeddings = model(**batch)
    return embeddings
```

- [ ] **Step 2: Verify import works**

Run: `uv run python -c "from core.embedding import get_device, load_model, load_image, embed; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add core/embedding.py
git commit -m "feat: extract embedding functions into core/embedding.py"
```

---

### Task 5: Create `core/search.py`

**Files:**
- Create: `core/search.py`

- [ ] **Step 1: Create `core/search.py`**

Extract `search_multi` and `filter_results` from `streamlit_app.py:118-156`. The single-document `search` function is dropped — `search_multi` with `filter_file_id` covers that case. The `results` parameter type changes from `dict[str, EmbedResults]` to `dict[str, torch.Tensor]` mapping job IDs to their page embedding tensors directly, since `EmbedResults` is being removed.

```python
import torch

from core.types import EmbeddingProcessor


def search_multi(
    query: str,
    model: torch.nn.Module,
    processor: EmbeddingProcessor,
    embeddings: dict[str, torch.Tensor],
    filter_file_id: str | None = None,
) -> list[tuple[str, int, float]]:
    """Score a text query across multiple documents and return ranked results."""
    docs = (
        {filter_file_id: embeddings[filter_file_id]}
        if filter_file_id is not None
        else embeddings
    )
    batch = processor.process_queries([query])
    batch = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    with torch.inference_mode():
        query_embedding = model(**batch)
    ranked: list[tuple[str, int, float]] = []
    for file_id, page_embeddings in docs.items():
        scores = processor.score(
            query_embedding, page_embeddings, device=str(model.device)
        )
        for page_idx in range(scores.shape[1]):
            ranked.append((file_id, page_idx, scores[0][page_idx].item()))
    ranked.sort(key=lambda x: x[2], reverse=True)
    return ranked


def filter_results(
    results: list[tuple[str, int, float]],
    top_k: int = 5,
    min_score: float = 0.0,
) -> list[tuple[str, int, float]]:
    """Apply score threshold then top-K to ranked search results."""
    filtered = [r for r in results if r[2] >= min_score]
    return filtered[:top_k]
```

- [ ] **Step 2: Verify import works**

Run: `uv run python -c "from core.search import search_multi, filter_results; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add core/search.py
git commit -m "feat: extract search functions into core/search.py"
```

---

### Task 6: Migrate tests to `tests/test_core.py`

**Files:**
- Create: `tests/test_core.py`
- Modify: `tests/test_app.py` (will be deleted in Phase 3)

- [ ] **Step 1: Create `tests/test_core.py`**

Migrate all test classes from `tests/test_app.py` with updated imports and patch targets. Drop `TestSearch` (single-doc `search` removed) and `TestCleanupStaleResults` (`cleanup_stale_results` removed). Update `_make_result` helper to return `dict[str, torch.Tensor]` for `search_multi` tests. Update `TestSearchMulti` to pass `dict[str, torch.Tensor]` instead of `dict[str, EmbedResults]`.

```python
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image, UnidentifiedImageError
from pytest import approx

from core.constants import DPI_OPTIONS, IMAGE_EXTENSIONS
from core.embedding import embed, get_device, load_image
from core.rendering import render_pages
from core.search import filter_results, search_multi

PDF_DATA_DIR = Path(__file__).parent / "data" / "pdf"
IMAGE_DATA_DIR = Path(__file__).parent / "data" / "images"


def _make_mock_model(return_value: torch.Tensor) -> MagicMock:
    model = MagicMock()
    model.device = "cpu"
    model.return_value = return_value
    return model


def _make_image_processor() -> MagicMock:
    mock_val = MagicMock(spec=torch.Tensor)
    mock_val.to.return_value = mock_val
    processor = MagicMock()
    processor.process_images.return_value = {"pixel_values": mock_val}
    return processor


def _make_query_processor() -> MagicMock:
    mock_val = MagicMock(spec=torch.Tensor)
    mock_val.to.return_value = mock_val
    processor = MagicMock()
    processor.process_queries.return_value = {"input_ids": mock_val}
    return processor


class TestDpiOptions:
    def test_contains_three_options(self) -> None:
        assert len(DPI_OPTIONS) == 3

    def test_values_are_72_150_300(self) -> None:
        assert sorted(DPI_OPTIONS.values()) == [72, 150, 300]

    def test_labels_match_values(self) -> None:
        for label, value in DPI_OPTIONS.items():
            assert str(value) in label


class TestImageExtensions:
    def test_contains_expected_types(self) -> None:
        assert IMAGE_EXTENSIONS == {"png", "jpg", "jpeg", "webp"}


class TestLoadImage:
    def test_loads_png_as_rgb(self) -> None:
        img = load_image(IMAGE_DATA_DIR / "red.png")
        assert img.mode == "RGB"
        assert img.size == (64, 64)

    def test_loads_jpg_as_rgb(self) -> None:
        img = load_image(IMAGE_DATA_DIR / "blue.jpg")
        assert img.mode == "RGB"
        assert img.size == (64, 64)

    def test_loads_webp_as_rgb(self) -> None:
        img = load_image(IMAGE_DATA_DIR / "green.webp")
        assert img.mode == "RGB"
        assert img.size == (64, 64)

    def test_embed_accepts_image_fixture(self) -> None:
        mock_processor = _make_image_processor()
        mock_model = _make_mock_model(torch.randn(1, 4, 128))

        img = load_image(IMAGE_DATA_DIR / "red.png")
        result = embed([img], mock_model, mock_processor)

        assert isinstance(result, torch.Tensor)
        mock_processor.process_images.assert_called_once_with([img])

    def test_missing_file_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_image(IMAGE_DATA_DIR / "nonexistent.png")

    def test_corrupt_data_raises_unidentified_image(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.png"
        bad_file.write_bytes(b"not an image")
        with pytest.raises(UnidentifiedImageError):
            load_image(bad_file)


class TestGetDevice:
    @patch("core.embedding.torch")
    def test_prefers_mps(self, mock_torch: MagicMock) -> None:
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.cuda.is_available.return_value = True
        assert get_device() == "mps"

    @patch("core.embedding.torch")
    def test_falls_back_to_cuda(self, mock_torch: MagicMock) -> None:
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = True
        assert get_device() == "cuda"

    @patch("core.embedding.torch")
    def test_falls_back_to_cpu(self, mock_torch: MagicMock) -> None:
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False
        assert get_device() == "cpu"


class TestRenderPages:
    def test_renders_single_page_pdf(self) -> None:
        data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        pages = render_pages(data)
        assert len(pages) == 1
        assert isinstance(pages[0], Image.Image)
        assert pages[0].mode == "RGB"

    def test_renders_multi_page_pdf(self) -> None:
        data = (PDF_DATA_DIR / "multi_page.pdf").read_bytes()
        pages = render_pages(data)
        assert len(pages) == 3
        assert all(isinstance(p, Image.Image) for p in pages)
        assert all(p.mode == "RGB" for p in pages)

    def test_returns_empty_for_no_pages(self) -> None:
        pages = render_pages(b"%PDF-1.4\n%%EOF\n")
        assert pages == []

    def test_returns_empty_for_invalid_data(self) -> None:
        pages = render_pages(b"not a pdf")
        assert pages == []

    def test_higher_dpi_produces_larger_images(self) -> None:
        data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        pages_72 = render_pages(data, dpi=72)
        pages_150 = render_pages(data, dpi=150)
        assert pages_150[0].width > pages_72[0].width
        assert pages_150[0].height > pages_72[0].height

    def test_default_dpi_is_150(self) -> None:
        data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        pages_default = render_pages(data)
        pages_150 = render_pages(data, dpi=150)
        assert pages_default[0].size == pages_150[0].size


class TestEmbed:
    def test_returns_per_page_embeddings(self) -> None:
        num_pages = 2
        num_patches = 4
        embedding_dim = 128

        mock_processor = _make_image_processor()
        mock_model = _make_mock_model(
            torch.randn(num_pages, num_patches, embedding_dim)
        )

        images = [Image.new("RGB", (64, 64)) for _ in range(num_pages)]
        embeddings = embed(images, mock_model, mock_processor)

        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (num_pages, num_patches, embedding_dim)

    def test_calls_process_images(self) -> None:
        mock_processor = _make_image_processor()
        mock_model = _make_mock_model(torch.randn(1, 4, 128))

        images = [Image.new("RGB", (64, 64))]
        embed(images, mock_model, mock_processor)

        mock_processor.process_images.assert_called_once_with(images)
        mock_processor.process_images.return_value[
            "pixel_values"
        ].to.assert_called_once_with("cpu")


class TestFilterResults:
    def test_filters_below_min_score(self) -> None:
        results: list[tuple[str, int, float]] = [
            ("id_a", 0, 0.9),
            ("id_a", 1, 0.3),
            ("id_b", 0, 0.6),
        ]
        filtered = filter_results(results, top_k=10, min_score=0.5)
        assert len(filtered) == 2
        assert filtered[0] == ("id_a", 0, approx(0.9))
        assert filtered[1] == ("id_b", 0, approx(0.6))

    def test_limits_to_top_k(self) -> None:
        results: list[tuple[str, int, float]] = [
            ("id_a", 0, 0.9),
            ("id_b", 0, 0.8),
            ("id_a", 1, 0.7),
        ]
        filtered = filter_results(results, top_k=2)
        assert len(filtered) == 2
        assert filtered[0] == ("id_a", 0, approx(0.9))
        assert filtered[1] == ("id_b", 0, approx(0.8))

    def test_threshold_applied_before_top_k(self) -> None:
        results: list[tuple[str, int, float]] = [
            ("id_a", 0, 0.9),
            ("id_b", 0, 0.8),
            ("id_a", 1, 0.3),
            ("id_b", 1, 0.1),
        ]
        filtered = filter_results(results, top_k=5, min_score=0.5)
        assert len(filtered) == 2

    def test_returns_empty_for_empty_input(self) -> None:
        assert filter_results([], top_k=5, min_score=0.0) == []

    def test_default_values(self) -> None:
        results: list[tuple[str, int, float]] = [
            (f"id_{i}", 0, 0.9 - i * 0.1) for i in range(7)
        ]
        filtered = filter_results(results)
        assert len(filtered) == 5
        assert filtered[0][2] == approx(0.9)


class TestSearchMulti:
    def test_returns_cross_document_ranked_results(self) -> None:
        mock_processor = _make_query_processor()
        mock_model = _make_mock_model(torch.tensor([[0.1, 0.2, 0.3]]))

        mock_processor.score.side_effect = [
            torch.tensor([[0.3, 0.8]]),
            torch.tensor([[0.6]]),
        ]

        embeddings: dict[str, torch.Tensor] = {
            "id_a": torch.randn(2, 4, 128),
            "id_b": torch.randn(1, 4, 128),
        }

        ranked = search_multi("test query", mock_model, mock_processor, embeddings)

        assert len(ranked) == 3
        assert ranked[0] == ("id_a", 1, approx(0.8))
        assert ranked[1] == ("id_b", 0, approx(0.6))
        assert ranked[2] == ("id_a", 0, approx(0.3))

    def test_filters_to_single_document(self) -> None:
        mock_processor = _make_query_processor()
        mock_model = _make_mock_model(torch.tensor([[0.1, 0.2, 0.3]]))
        mock_processor.score.return_value = torch.tensor([[0.5]])

        embeddings: dict[str, torch.Tensor] = {
            "id_a": torch.randn(1, 4, 128),
            "id_b": torch.randn(1, 4, 128),
        }

        ranked = search_multi(
            "test query", mock_model, mock_processor, embeddings, filter_file_id="id_b"
        )

        assert len(ranked) == 1
        assert ranked[0][0] == "id_b"
        assert mock_processor.score.call_count == 1

    def test_encodes_query_once_for_multiple_docs(self) -> None:
        mock_processor = _make_query_processor()
        mock_model = _make_mock_model(torch.tensor([[0.1, 0.2, 0.3]]))
        mock_processor.score.side_effect = [
            torch.tensor([[0.3]]),
            torch.tensor([[0.6]]),
        ]

        embeddings: dict[str, torch.Tensor] = {
            "id_a": torch.randn(1, 4, 128),
            "id_b": torch.randn(1, 4, 128),
        }

        search_multi("test", mock_model, mock_processor, embeddings)

        mock_processor.process_queries.assert_called_once_with(["test"])
        assert mock_model.call_count == 1
```

- [ ] **Step 2: Run migrated tests**

Run: `uv run pytest tests/test_core.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_core.py
git commit -m "test: migrate tests to tests/test_core.py with core imports"
```

---

### Task 7: Update `streamlit_app.py` to import from `core/`

**Files:**
- Modify: `streamlit_app.py`
- Delete: `tests/test_app.py`

- [ ] **Step 1: Rewrite `streamlit_app.py` imports to use `core/`**

Replace the top of `streamlit_app.py` to import from `core/` instead of defining functions inline. Remove the now-extracted functions and types from the file, keeping only the Streamlit UI code. The app should use `@st.cache_resource` as a wrapper around `core.embedding.load_model`.

```python
import json
import time

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from core.constants import DPI_OPTIONS, IMAGE_EXTENSIONS, MODEL_ID
from core.embedding import embed, get_device, load_model
from core.rendering import render_pages
from core.search import filter_results, search_multi


@st.cache_resource
def cached_load_model(device: str) -> tuple:
    """Cache-wrapped model loader for Streamlit."""
    return load_model(device)


# UI
st.set_page_config(page_title="Granite Vision Embedding Pipeline", layout="centered")
st.title("Granite Vision Embedding Pipeline")
st.write(
    "Generate vector embeddings from PDFs and images with Granite Vision Embedding Pipeline."
)

uploaded_files = st.file_uploader(
    "Upload files",
    type=["pdf", *IMAGE_EXTENSIONS],
    accept_multiple_files=True,
)

device = get_device()

if uploaded_files:
    total_size_mb = sum(len(f.getvalue()) for f in uploaded_files) / 1_048_576
    st.caption(f"{len(uploaded_files)} file(s) · {total_size_mb:.1f} MB")

    dpi_label = st.radio("Render DPI", DPI_OPTIONS, index=1, horizontal=True)
    dpi = DPI_OPTIONS[dpi_label]

    # Clean up stale results for removed files
    if "results" in st.session_state:
        current_ids = {f.file_id for f in uploaded_files}
        stale_ids = set(st.session_state.results.keys()) - current_ids
        for stale_id in stale_ids:
            del st.session_state.results[stale_id]
        if stale_ids:
            st.session_state.pop("search_results", None)
        if not st.session_state.results:
            del st.session_state["results"]

    existing_results = st.session_state.get("results", {})
    files_to_embed = [f for f in uploaded_files if f.file_id not in existing_results]

    col_embed, col_reembed = st.columns(2)
    embed_clicked = col_embed.button(
        "Embed", type="primary", disabled=not files_to_embed
    )
    reembed_clicked = col_reembed.button("Re-embed All")

    if embed_clicked or reembed_clicked:
        if reembed_clicked:
            st.session_state.pop("results", None)
            files_to_embed = list(uploaded_files)

        if files_to_embed:
            try:
                progress = st.progress(0.0, text="Loading model...")
                model, processor = cached_load_model(device)
                results = st.session_state.get("results", {})

                for i, f in enumerate(files_to_embed):
                    file_stem = f.name.rsplit(".", 1)[0]
                    progress.progress(
                        i / len(files_to_embed),
                        text=f"Processing {f.name}...",
                    )
                    try:
                        total_start = time.perf_counter_ns()
                        ext = f.name.rsplit(".", 1)[-1].lower()
                        if ext in IMAGE_EXTENSIONS:
                            try:
                                image = Image.open(f).convert("RGB")
                            except (UnidentifiedImageError, OSError) as e:
                                st.error(f"{f.name}: {e}")
                                continue
                            pages = [image]
                        else:
                            pages = render_pages(f.read(), dpi=dpi)
                            if not pages:
                                st.error(f"{f.name}: PDF contains no pages to embed.")
                                continue
                        page_embeddings = embed(pages, model, processor)
                        total_duration = time.perf_counter_ns() - total_start

                        results[f.file_id] = {
                            "file_id": f.file_id,
                            "pages": pages,
                            "page_embeddings": page_embeddings,
                            "total_duration": total_duration,
                            "file_stem": file_stem,
                            "dpi": dpi,
                            "json": json.dumps(
                                {
                                    "file_name": file_stem,
                                    "model": MODEL_ID,
                                    "dpi": dpi,
                                    "embeddings": page_embeddings.tolist(),
                                    "total_duration": total_duration,
                                    "page_count": len(pages),
                                }
                            ),
                        }
                    except (OSError, RuntimeError, ValueError) as e:
                        st.error(f"{f.name}: {e}")
                    except Exception as e:
                        st.exception(e)

                progress.progress(1.0, text="Complete.")
                progress.empty()
                st.session_state.results = results
                st.session_state.pop("search_results", None)

            except (OSError, RuntimeError) as e:
                st.error(str(e))
            except Exception as e:
                st.exception(e)

    if "results" in st.session_state and st.session_state.results:
        all_results = st.session_state.results

        st.success(f"Embedded {len(all_results)} document(s).")

        # Summary metrics
        total_pages = sum(len(r["pages"]) for r in all_results.values())
        total_duration_ns = sum(r["total_duration"] for r in all_results.values())

        st.subheader("Metrics")
        st.metric("Model", MODEL_ID)
        col1, col2, col3 = st.columns(3)
        col1.metric("Duration", f"{total_duration_ns / 1_000_000_000:.2f} s")
        col2.metric("Pages", total_pages)
        col3.metric("Documents", len(all_results))

        # Per-document expanders
        for file_id, r in all_results.items():
            with st.expander(f"{r['file_stem']} ({len(r['pages'])} pages)"):
                cols = st.columns(min(len(r["pages"]), 4))
                for i, page in enumerate(r["pages"]):
                    cols[i % 4].image(page, caption=f"Page {i + 1}", width="stretch")
                doc_duration = r["total_duration"] / 1_000_000_000
                st.caption(
                    f"Duration: {doc_duration:.2f} s · "
                    f"Pages: {len(r['pages'])} · DPI: {r['dpi']}"
                )
                st.download_button(
                    label=f"Download {r['file_stem']} JSON",
                    data=r["json"],
                    file_name=f"{r['file_stem']}_embedding.json",
                    mime="application/json",
                    key=f"download_{file_id}",
                )

        # Download All
        if len(all_results) > 1:
            all_json = "[" + ",".join(r["json"] for r in all_results.values()) + "]"
            st.download_button(
                label="Download All JSON",
                data=all_json,
                file_name="all_embeddings.json",
                mime="application/json",
                key="download_all",
            )

        # Search
        st.subheader("Search")
        filter_options = ["All documents"] + [
            r["file_stem"] for r in all_results.values()
        ]
        filter_file_ids: list[str | None] = [None] + list(all_results.keys())
        filter_idx = st.selectbox(
            "Document filter",
            range(len(filter_options)),
            format_func=lambda i: filter_options[i],
        )
        selected_file_id = filter_file_ids[filter_idx]

        col_topk, col_minscore = st.columns(2)
        top_k = col_topk.number_input("Top K", min_value=1, max_value=100, value=5)
        min_score = col_minscore.number_input(
            "Min score", min_value=0.0, value=0.0, step=0.1
        )

        query = st.text_input("Text query")
        if st.button("Search"):
            if not query:
                st.warning("Enter a search query.")
            else:
                try:
                    model, processor = cached_load_model(device)
                    embeddings_map = {
                        fid: r["page_embeddings"] for fid, r in all_results.items()
                    }
                    raw_results = search_multi(
                        query,
                        model,
                        processor,
                        embeddings_map,
                        filter_file_id=selected_file_id,
                    )
                    st.session_state.search_results = filter_results(
                        raw_results, top_k=top_k, min_score=min_score
                    )
                except (OSError, RuntimeError, ValueError) as e:
                    st.error(str(e))
                except Exception as e:
                    st.exception(e)

        if "search_results" in st.session_state:
            search_results = st.session_state.search_results
            if search_results:
                cols = st.columns(min(len(search_results), 4))
                for rank, (fid, page_idx, score) in enumerate(search_results):
                    r = all_results[fid]
                    cols[rank % 4].image(
                        r["pages"][page_idx],
                        caption=(
                            f"{r['file_stem']} · Page {page_idx + 1} · {score:.4f}"
                        ),
                        width="stretch",
                    )
            else:
                st.info("No results above the score threshold.")

st.caption(f"Device: {device.upper()}")
```

- [ ] **Step 2: Delete old test file**

```bash
rm tests/test_app.py
```

- [ ] **Step 3: Run all tests to verify nothing is broken**

Run: `uv run pytest tests/test_core.py -v`
Expected: All tests PASS

- [ ] **Step 4: Run lint**

Run: `uv run ruff check .`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py
git rm tests/test_app.py
git commit -m "refactor: update streamlit_app.py to import from core/, delete old tests"
```

---

## Phase 2: API Backend

Build the FastAPI backend with SQLite database, worker thread, and API routes.

### Task 8: Add dependencies to `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add fastapi, uvicorn, httpx to dependencies**

Add to the `dependencies` list in `pyproject.toml`:

```toml
dependencies = [
    "fastapi>=0.115",
    "httpx>=0.28",
    "pymupdf>=1.25.0",
    "python-multipart>=0.0.18",
    "streamlit==1.52.2",
    "torch==2.9.1",
    "transformers>=4.50,<6",
    "uvicorn>=0.34",
]
```

- [ ] **Step 2: Sync dependencies**

Run: `uv sync`
Expected: Dependencies installed successfully

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add fastapi, uvicorn, httpx, python-multipart dependencies"
```

---

### Task 9: Create `api/database.py`

**Files:**
- Create: `api/__init__.py`
- Create: `api/database.py`
- Create: `tests/test_database.py`

- [ ] **Step 1: Write failing tests for database module**

```python
# tests/test_database.py
import sqlite3
from pathlib import Path

import pytest

from api.database import (
    create_job,
    delete_job,
    get_connection,
    get_job,
    init_db,
    list_jobs,
    update_job,
)


@pytest.fixture
def db(tmp_path: Path) -> sqlite3.Connection:
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    return conn


class TestInitDb:
    def test_creates_jobs_table(self, db: sqlite3.Connection) -> None:
        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'"
        )
        assert cursor.fetchone() is not None

    def test_enables_wal_mode(self, db: sqlite3.Connection) -> None:
        cursor = db.execute("PRAGMA journal_mode")
        assert cursor.fetchone()[0] == "wal"

    def test_idempotent(self, db: sqlite3.Connection) -> None:
        init_db(db)  # second call should not raise
        cursor = db.execute("SELECT count(*) FROM jobs")
        assert cursor.fetchone()[0] == 0


class TestCreateJob:
    def test_inserts_pending_job(self, db: sqlite3.Connection) -> None:
        job_id = create_job(
            db,
            file_name="test.pdf",
            file_stem="test",
            file_path="uploads/abc.pdf",
            file_type="pdf",
            dpi=150,
        )
        job = get_job(db, job_id)
        assert job is not None
        assert job["status"] == "pending"
        assert job["file_name"] == "test.pdf"
        assert job["file_stem"] == "test"
        assert job["file_type"] == "pdf"
        assert job["dpi"] == 150

    def test_generates_unique_ids(self, db: sqlite3.Connection) -> None:
        id1 = create_job(db, file_name="a.pdf", file_stem="a", file_path="uploads/a.pdf", file_type="pdf", dpi=150)
        id2 = create_job(db, file_name="b.pdf", file_stem="b", file_path="uploads/b.pdf", file_type="pdf", dpi=150)
        assert id1 != id2


class TestGetJob:
    def test_returns_none_for_missing(self, db: sqlite3.Connection) -> None:
        assert get_job(db, "nonexistent") is None


class TestListJobs:
    def test_returns_all_jobs(self, db: sqlite3.Connection) -> None:
        create_job(db, file_name="a.pdf", file_stem="a", file_path="uploads/a.pdf", file_type="pdf", dpi=150)
        create_job(db, file_name="b.pdf", file_stem="b", file_path="uploads/b.pdf", file_type="pdf", dpi=150)
        jobs = list_jobs(db)
        assert len(jobs) == 2

    def test_filters_by_status(self, db: sqlite3.Connection) -> None:
        job_id = create_job(db, file_name="a.pdf", file_stem="a", file_path="uploads/a.pdf", file_type="pdf", dpi=150)
        update_job(db, job_id, status="completed", page_count=1, duration_ns=100, result_path="results/a.json", tensor_path="results/a.pt")
        create_job(db, file_name="b.pdf", file_stem="b", file_path="uploads/b.pdf", file_type="pdf", dpi=150)
        pending = list_jobs(db, status="pending")
        assert len(pending) == 1
        assert pending[0]["file_name"] == "b.pdf"

    def test_ordered_by_created_at(self, db: sqlite3.Connection) -> None:
        create_job(db, file_name="first.pdf", file_stem="first", file_path="uploads/first.pdf", file_type="pdf", dpi=150)
        create_job(db, file_name="second.pdf", file_stem="second", file_path="uploads/second.pdf", file_type="pdf", dpi=150)
        jobs = list_jobs(db)
        assert jobs[0]["file_name"] == "first.pdf"

    def test_returns_empty_for_no_jobs(self, db: sqlite3.Connection) -> None:
        assert list_jobs(db) == []


class TestUpdateJob:
    def test_updates_to_completed(self, db: sqlite3.Connection) -> None:
        job_id = create_job(db, file_name="a.pdf", file_stem="a", file_path="uploads/a.pdf", file_type="pdf", dpi=150)
        update_job(db, job_id, status="completed", page_count=3, duration_ns=5000, result_path="results/a.json", tensor_path="results/a.pt")
        job = get_job(db, job_id)
        assert job["status"] == "completed"
        assert job["page_count"] == 3
        assert job["duration_ns"] == 5000
        assert job["result_path"] == "results/a.json"
        assert job["tensor_path"] == "results/a.pt"

    def test_updates_to_failed(self, db: sqlite3.Connection) -> None:
        job_id = create_job(db, file_name="a.pdf", file_stem="a", file_path="uploads/a.pdf", file_type="pdf", dpi=150)
        update_job(db, job_id, status="failed", error="corrupt file")
        job = get_job(db, job_id)
        assert job["status"] == "failed"
        assert job["error"] == "corrupt file"

    def test_updates_to_processing(self, db: sqlite3.Connection) -> None:
        job_id = create_job(db, file_name="a.pdf", file_stem="a", file_path="uploads/a.pdf", file_type="pdf", dpi=150)
        update_job(db, job_id, status="processing")
        job = get_job(db, job_id)
        assert job["status"] == "processing"


class TestDeleteJob:
    def test_removes_job(self, db: sqlite3.Connection) -> None:
        job_id = create_job(db, file_name="a.pdf", file_stem="a", file_path="uploads/a.pdf", file_type="pdf", dpi=150)
        delete_job(db, job_id)
        assert get_job(db, job_id) is None

    def test_delete_nonexistent_is_noop(self, db: sqlite3.Connection) -> None:
        delete_job(db, "nonexistent")  # should not raise


class TestResetProcessingJobs:
    def test_resets_processing_to_pending(self, db: sqlite3.Connection) -> None:
        from api.database import reset_processing_jobs
        job_id = create_job(db, file_name="a.pdf", file_stem="a", file_path="uploads/a.pdf", file_type="pdf", dpi=150)
        update_job(db, job_id, status="processing")
        count = reset_processing_jobs(db)
        assert count == 1
        job = get_job(db, job_id)
        assert job["status"] == "pending"


class TestNextPendingJob:
    def test_returns_oldest_pending(self, db: sqlite3.Connection) -> None:
        from api.database import next_pending_job
        create_job(db, file_name="first.pdf", file_stem="first", file_path="uploads/first.pdf", file_type="pdf", dpi=150)
        create_job(db, file_name="second.pdf", file_stem="second", file_path="uploads/second.pdf", file_type="pdf", dpi=150)
        job = next_pending_job(db)
        assert job is not None
        assert job["file_name"] == "first.pdf"

    def test_returns_none_when_empty(self, db: sqlite3.Connection) -> None:
        from api.database import next_pending_job
        assert next_pending_job(db) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_database.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api'`

- [ ] **Step 3: Create `api/__init__.py`**

```python
# empty file
```

- [ ] **Step 4: Implement `api/database.py`**

```python
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def get_connection(db_path: Path | str) -> sqlite3.Connection:
    """Create a SQLite connection with WAL mode and busy timeout."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create the jobs table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id            TEXT PRIMARY KEY,
            status        TEXT NOT NULL,
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL,
            file_name     TEXT NOT NULL,
            file_stem     TEXT NOT NULL,
            file_path     TEXT NOT NULL,
            file_type     TEXT NOT NULL,
            dpi           INTEGER NOT NULL,
            page_count    INTEGER,
            duration_ns   INTEGER,
            result_path   TEXT,
            tensor_path   TEXT,
            error         TEXT
        )
    """)
    conn.commit()


def create_job(
    conn: sqlite3.Connection,
    *,
    file_name: str,
    file_stem: str,
    file_path: str,
    file_type: str,
    dpi: int,
    job_id: str | None = None,
) -> str:
    """Insert a new pending job and return its ID."""
    if job_id is None:
        job_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO jobs (id, status, created_at, updated_at, file_name, file_stem, file_path, file_type, dpi)
           VALUES (?, 'pending', ?, ?, ?, ?, ?, ?, ?)""",
        (job_id, now, now, file_name, file_stem, file_path, file_type, dpi),
    )
    conn.commit()
    return job_id


def get_job(conn: sqlite3.Connection, job_id: str) -> dict | None:
    """Fetch a single job by ID, or None if not found."""
    cursor = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
    row = cursor.fetchone()
    return dict(row) if row else None


def list_jobs(conn: sqlite3.Connection, status: str | None = None) -> list[dict]:
    """List jobs ordered by created_at, optionally filtered by status."""
    if status:
        cursor = conn.execute(
            "SELECT * FROM jobs WHERE status = ? ORDER BY created_at", (status,)
        )
    else:
        cursor = conn.execute("SELECT * FROM jobs ORDER BY created_at")
    return [dict(row) for row in cursor.fetchall()]


def update_job(
    conn: sqlite3.Connection,
    job_id: str,
    *,
    status: str,
    page_count: int | None = None,
    duration_ns: int | None = None,
    result_path: str | None = None,
    tensor_path: str | None = None,
    error: str | None = None,
) -> None:
    """Update a job's status and optional metadata fields."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """UPDATE jobs SET status = ?, updated_at = ?, page_count = COALESCE(?, page_count),
           duration_ns = COALESCE(?, duration_ns), result_path = COALESCE(?, result_path),
           tensor_path = COALESCE(?, tensor_path), error = COALESCE(?, error)
           WHERE id = ?""",
        (status, now, page_count, duration_ns, result_path, tensor_path, error, job_id),
    )
    conn.commit()


def delete_job(conn: sqlite3.Connection, job_id: str) -> None:
    """Delete a job by ID."""
    conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    conn.commit()


def reset_processing_jobs(conn: sqlite3.Connection) -> int:
    """Reset any jobs stuck in 'processing' back to 'pending'. Returns count reset."""
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "UPDATE jobs SET status = 'pending', updated_at = ? WHERE status = 'processing'",
        (now,),
    )
    conn.commit()
    return cursor.rowcount


def next_pending_job(conn: sqlite3.Connection) -> dict | None:
    """Fetch the oldest pending job, or None."""
    cursor = conn.execute(
        "SELECT * FROM jobs WHERE status = 'pending' ORDER BY created_at LIMIT 1"
    )
    row = cursor.fetchone()
    return dict(row) if row else None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_database.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add api/__init__.py api/database.py tests/test_database.py
git commit -m "feat: add api/database.py with SQLite job management"
```

---

### Task 10: Create `api/models.py`

**Files:**
- Create: `api/models.py`

- [ ] **Step 1: Create Pydantic models**

```python
from pydantic import BaseModel


class JobResponse(BaseModel):
    id: str
    status: str
    created_at: str
    updated_at: str
    file_name: str
    file_stem: str
    file_type: str
    dpi: int
    page_count: int | None = None
    duration_ns: int | None = None
    error: str | None = None


class JobCreateResponse(BaseModel):
    job_id: str
    status: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    min_score: float = 0.0
    filter_file_id: str | None = None


class SearchResult(BaseModel):
    file_id: str
    page_index: int
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]


class HealthResponse(BaseModel):
    device: str
    queue_depth: int
    worker_running: bool
```

- [ ] **Step 2: Verify import works**

Run: `uv run python -c "from api.models import JobResponse, SearchRequest, HealthResponse; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add api/models.py
git commit -m "feat: add api/models.py with Pydantic request/response models"
```

---

### Task 11: Create `api/worker.py`

**Files:**
- Create: `api/worker.py`
- Create: `tests/test_worker.py`

- [ ] **Step 1: Write failing tests for worker**

```python
# tests/test_worker.py
import json
import sqlite3
import threading
from collections import OrderedDict
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from api.database import create_job, get_connection, get_job, init_db, update_job
from api.worker import EmbeddingWorker

PDF_DATA_DIR = Path(__file__).parent / "data" / "pdf"
IMAGE_DATA_DIR = Path(__file__).parent / "data" / "images"


def _make_mock_model(return_value: torch.Tensor) -> MagicMock:
    model = MagicMock()
    model.device = "cpu"
    model.return_value = return_value
    return model


def _make_mock_processor() -> MagicMock:
    mock_val = MagicMock(spec=torch.Tensor)
    mock_val.to.return_value = mock_val
    processor = MagicMock()
    processor.process_images.return_value = {"pixel_values": mock_val}
    processor.process_queries.return_value = {"input_ids": mock_val}
    return processor


@pytest.fixture
def db(tmp_path: Path) -> sqlite3.Connection:
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    return conn


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    uploads = tmp_path / "uploads"
    results = tmp_path / "results"
    uploads.mkdir()
    results.mkdir()
    return uploads, results


class TestProcessJob:
    @patch("api.worker.load_model")
    def test_processes_pdf_to_completed(
        self, mock_load: MagicMock, db: sqlite3.Connection, dirs: tuple[Path, Path]
    ) -> None:
        uploads, results = dirs
        mock_model = _make_mock_model(torch.randn(1, 4, 128))
        mock_processor = _make_mock_processor()
        mock_load.return_value = (mock_model, mock_processor)

        # Copy PDF fixture to uploads
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        upload_path = uploads / "test.pdf"
        upload_path.write_bytes(pdf_data)

        job_id = create_job(
            db, file_name="test.pdf", file_stem="test",
            file_path=str(upload_path), file_type="pdf", dpi=150,
        )

        worker = EmbeddingWorker(db, uploads_dir=uploads, results_dir=results)
        worker.process_job(get_job(db, job_id))

        job = get_job(db, job_id)
        assert job["status"] == "completed"
        assert job["page_count"] == 1
        assert job["duration_ns"] > 0
        assert Path(job["result_path"]).exists()
        assert Path(job["tensor_path"]).exists()

    @patch("api.worker.load_model")
    def test_processes_image_to_completed(
        self, mock_load: MagicMock, db: sqlite3.Connection, dirs: tuple[Path, Path]
    ) -> None:
        uploads, results = dirs
        mock_model = _make_mock_model(torch.randn(1, 4, 128))
        mock_processor = _make_mock_processor()
        mock_load.return_value = (mock_model, mock_processor)

        # Copy image fixture to uploads
        img_data = (IMAGE_DATA_DIR / "red.png").read_bytes()
        upload_path = uploads / "red.png"
        upload_path.write_bytes(img_data)

        job_id = create_job(
            db, file_name="red.png", file_stem="red",
            file_path=str(upload_path), file_type="image", dpi=150,
        )

        worker = EmbeddingWorker(db, uploads_dir=uploads, results_dir=results)
        worker.process_job(get_job(db, job_id))

        job = get_job(db, job_id)
        assert job["status"] == "completed"
        assert job["page_count"] == 1

    @patch("api.worker.load_model")
    def test_marks_corrupt_file_as_failed(
        self, mock_load: MagicMock, db: sqlite3.Connection, dirs: tuple[Path, Path]
    ) -> None:
        uploads, results = dirs
        mock_model = _make_mock_model(torch.randn(1, 4, 128))
        mock_processor = _make_mock_processor()
        mock_load.return_value = (mock_model, mock_processor)

        upload_path = uploads / "bad.png"
        upload_path.write_bytes(b"not an image")

        job_id = create_job(
            db, file_name="bad.png", file_stem="bad",
            file_path=str(upload_path), file_type="image", dpi=150,
        )

        worker = EmbeddingWorker(db, uploads_dir=uploads, results_dir=results)
        worker.process_job(get_job(db, job_id))

        job = get_job(db, job_id)
        assert job["status"] == "failed"
        assert job["error"] is not None


class TestStartupRecovery:
    @patch("api.worker.load_model")
    def test_resets_processing_to_pending(
        self, mock_load: MagicMock, db: sqlite3.Connection, dirs: tuple[Path, Path]
    ) -> None:
        uploads, results = dirs
        mock_model = _make_mock_model(torch.randn(1, 4, 128))
        mock_processor = _make_mock_processor()
        mock_load.return_value = (mock_model, mock_processor)

        job_id = create_job(
            db, file_name="a.pdf", file_stem="a",
            file_path="uploads/a.pdf", file_type="pdf", dpi=150,
        )
        update_job(db, job_id, status="processing")

        worker = EmbeddingWorker(db, uploads_dir=uploads, results_dir=results)
        worker.startup_recovery()

        job = get_job(db, job_id)
        assert job["status"] == "pending"


class TestTensorCache:
    @patch("api.worker.load_model")
    def test_cache_evicts_lru(
        self, mock_load: MagicMock, db: sqlite3.Connection, dirs: tuple[Path, Path]
    ) -> None:
        uploads, results = dirs
        mock_model = _make_mock_model(torch.randn(1, 4, 128))
        mock_processor = _make_mock_processor()
        mock_load.return_value = (mock_model, mock_processor)

        worker = EmbeddingWorker(db, uploads_dir=uploads, results_dir=results, cache_max=2)
        worker._tensor_cache["a"] = torch.randn(1, 4, 128)
        worker._tensor_cache["b"] = torch.randn(1, 4, 128)

        # Adding "c" should evict "a" (LRU)
        worker._tensor_cache["c"] = torch.randn(1, 4, 128)
        worker._enforce_cache_limit()

        assert "a" not in worker._tensor_cache
        assert "b" in worker._tensor_cache
        assert "c" in worker._tensor_cache

    @patch("api.worker.load_model")
    def test_cache_remove_on_delete(
        self, mock_load: MagicMock, db: sqlite3.Connection, dirs: tuple[Path, Path]
    ) -> None:
        uploads, results = dirs
        mock_model = _make_mock_model(torch.randn(1, 4, 128))
        mock_processor = _make_mock_processor()
        mock_load.return_value = (mock_model, mock_processor)

        worker = EmbeddingWorker(db, uploads_dir=uploads, results_dir=results)
        worker._tensor_cache["a"] = torch.randn(1, 4, 128)
        worker.evict_cache("a")
        assert "a" not in worker._tensor_cache


class TestSearchDispatch:
    @patch("api.worker.load_model")
    def test_enqueue_and_drain_returns_results(
        self, mock_load: MagicMock, db: sqlite3.Connection, dirs: tuple[Path, Path]
    ) -> None:
        uploads, results = dirs
        mock_model = _make_mock_model(torch.tensor([[0.1, 0.2]]))
        mock_processor = _make_mock_processor()
        mock_processor.score.return_value = torch.tensor([[0.8]])
        mock_load.return_value = (mock_model, mock_processor)

        # Create a completed job with a .pt file
        job_id = create_job(
            db, file_name="a.pdf", file_stem="a",
            file_path="uploads/a.pdf", file_type="pdf", dpi=150,
        )
        tensor = torch.randn(1, 4, 128)
        tensor_path = results / f"{job_id}.pt"
        torch.save(tensor, tensor_path)
        update_job(
            db, job_id, status="completed", page_count=1,
            duration_ns=100, result_path=str(results / f"{job_id}.json"),
            tensor_path=str(tensor_path),
        )

        worker = EmbeddingWorker(db, uploads_dir=uploads, results_dir=results)
        future = worker.enqueue_search({
            "query": "test",
            "top_k": 5,
            "min_score": 0.0,
            "job_ids": [job_id],
        })

        # Drain the queue manually (simulates worker loop)
        worker._drain_search_queue()

        result = future.result(timeout=5)
        assert len(result) == 1
        assert result[0][0] == job_id

    @patch("api.worker.load_model")
    def test_enqueue_returns_empty_for_no_jobs(
        self, mock_load: MagicMock, db: sqlite3.Connection, dirs: tuple[Path, Path]
    ) -> None:
        uploads, results = dirs
        mock_model = _make_mock_model(torch.randn(1, 4, 128))
        mock_processor = _make_mock_processor()
        mock_load.return_value = (mock_model, mock_processor)

        worker = EmbeddingWorker(db, uploads_dir=uploads, results_dir=results)
        future = worker.enqueue_search({
            "query": "test",
            "job_ids": [],
        })
        worker._drain_search_queue()

        result = future.result(timeout=5)
        assert result == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_worker.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `api/worker.py`**

```python
import json
import queue
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from pathlib import Path

import torch

from api.database import (
    get_job,
    next_pending_job,
    reset_processing_jobs,
    update_job,
)
from core.constants import IMAGE_EXTENSIONS, MODEL_ID
from core.embedding import embed, get_device, load_image, load_model
from core.rendering import render_pages
from core.search import filter_results, search_multi


class EmbeddingWorker:
    def __init__(
        self,
        db,
        *,
        uploads_dir: Path,
        results_dir: Path,
        cache_max: int = 500,
    ) -> None:
        self._db = db
        self._uploads_dir = uploads_dir
        self._results_dir = results_dir
        self._cache_max = cache_max
        self._tensor_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._search_queue: queue.Queue[
            tuple[dict, Future]
        ] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        device = get_device()
        self._model, self._processor = load_model(device)

    def startup_recovery(self) -> None:
        """Reset any jobs stuck in 'processing' to 'pending'."""
        reset_processing_jobs(self._db)

    def start(self) -> None:
        """Start the worker thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the worker to stop and wait for it."""
        self._stop_event.set()
        if self._thread:
            self._thread.join()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def enqueue_search(self, params: dict) -> Future:
        """Enqueue a search request and return a Future for the result."""
        future: Future = Future()
        self._search_queue.put((params, future))
        return future

    def evict_cache(self, job_id: str) -> None:
        """Remove a tensor from the cache."""
        self._tensor_cache.pop(job_id, None)

    def _enforce_cache_limit(self) -> None:
        """Evict LRU entries until cache is within limit."""
        while len(self._tensor_cache) > self._cache_max:
            self._tensor_cache.popitem(last=False)

    def _run(self) -> None:
        """Main worker loop."""
        self.startup_recovery()
        while not self._stop_event.is_set():
            self._drain_search_queue()
            job = next_pending_job(self._db)
            if job:
                self.process_job(job)
            else:
                self._stop_event.wait(timeout=1.0)

    def _drain_search_queue(self) -> None:
        """Process all pending search requests."""
        while not self._search_queue.empty():
            try:
                params, future = self._search_queue.get_nowait()
            except queue.Empty:
                break
            try:
                result = self._execute_search(params)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

    def _execute_search(self, params: dict) -> list[tuple[str, int, float]]:
        """Run a search query using cached tensors."""
        query = params["query"]
        top_k = params.get("top_k", 5)
        min_score = params.get("min_score", 0.0)
        filter_file_id = params.get("filter_file_id")
        job_ids = params.get("job_ids", [])

        embeddings: dict[str, torch.Tensor] = {}
        for job_id in job_ids:
            tensor = self._get_or_load_tensor(job_id)
            if tensor is not None:
                embeddings[job_id] = tensor

        if not embeddings:
            return []

        raw = search_multi(
            query, self._model, self._processor, embeddings,
            filter_file_id=filter_file_id,
        )
        return filter_results(raw, top_k=top_k, min_score=min_score)

    def _get_or_load_tensor(self, job_id: str) -> torch.Tensor | None:
        """Load tensor from cache or .pt file."""
        if job_id in self._tensor_cache:
            self._tensor_cache.move_to_end(job_id)
            return self._tensor_cache[job_id]

        job = get_job(self._db, job_id)
        if not job or not job.get("tensor_path"):
            return None

        tensor_path = Path(job["tensor_path"])
        if not tensor_path.exists():
            return None

        tensor = torch.load(tensor_path, weights_only=True)
        self._tensor_cache[job_id] = tensor
        self._enforce_cache_limit()
        return tensor

    def process_job(self, job: dict) -> None:
        """Process a single embedding job."""
        job_id = job["id"]
        update_job(self._db, job_id, status="processing")

        try:
            total_start = time.perf_counter_ns()
            file_path = Path(job["file_path"])

            if job["file_type"] == "image":
                image = load_image(file_path)
                pages = [image]
            else:
                pdf_data = file_path.read_bytes()
                pages = render_pages(pdf_data, dpi=job["dpi"])
                if not pages:
                    raise ValueError("PDF contains no pages to embed")

            page_embeddings = embed(pages, self._model, self._processor)
            total_duration = time.perf_counter_ns() - total_start

            # Save results
            result_path = self._results_dir / f"{job_id}.json"
            tensor_path = self._results_dir / f"{job_id}.pt"

            result_json = json.dumps({
                "file_name": job["file_stem"],
                "model": MODEL_ID,
                "dpi": job["dpi"],
                "embeddings": page_embeddings.tolist(),
                "total_duration": total_duration,
                "page_count": len(pages),
            })
            result_path.write_text(result_json)
            torch.save(page_embeddings, tensor_path)

            # Cache the tensor
            self._tensor_cache[job_id] = page_embeddings
            self._enforce_cache_limit()

            update_job(
                self._db, job_id,
                status="completed",
                page_count=len(pages),
                duration_ns=total_duration,
                result_path=str(result_path),
                tensor_path=str(tensor_path),
            )
        except Exception as e:
            update_job(self._db, job_id, status="failed", error=str(e))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_worker.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add api/worker.py tests/test_worker.py
git commit -m "feat: add api/worker.py with background embedding worker"
```

---

### Task 12: Create `api/app.py`

**Files:**
- Create: `api/app.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write failing tests for API routes**

```python
# tests/test_api.py
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

PDF_DATA_DIR = Path(__file__).parent / "data" / "pdf"
IMAGE_DATA_DIR = Path(__file__).parent / "data" / "images"


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Create a TestClient with temp directories and mocked worker."""
    uploads = tmp_path / "uploads"
    results = tmp_path / "results"
    db_path = tmp_path / "test.db"
    uploads.mkdir()
    results.mkdir()

    with (
        patch.dict("os.environ", {
            "UPLOAD_DIR": str(uploads),
            "RESULT_DIR": str(results),
            "DATABASE_PATH": str(db_path),
        }),
        patch("api.app.EmbeddingWorker") as MockWorker,
    ):
        mock_worker = MagicMock()
        mock_worker.is_running = True
        MockWorker.return_value = mock_worker

        from api.app import create_app
        app = create_app()
        with TestClient(app) as tc:
            tc._mock_worker = mock_worker
            yield tc


class TestHealth:
    def test_returns_health(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "device" in data
        assert "queue_depth" in data
        assert "worker_running" in data


class TestUploadJob:
    def test_upload_pdf(self, client: TestClient) -> None:
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        resp = client.post(
            "/jobs",
            files={"file": ("test.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    def test_upload_image(self, client: TestClient) -> None:
        img_data = (IMAGE_DATA_DIR / "red.png").read_bytes()
        resp = client.post(
            "/jobs",
            files={"file": ("red.png", img_data, "image/png")},
            data={"dpi": "150"},
        )
        assert resp.status_code == 201

    def test_rejects_invalid_file_type(self, client: TestClient) -> None:
        resp = client.post(
            "/jobs",
            files={"file": ("test.txt", b"hello", "text/plain")},
            data={"dpi": "150"},
        )
        assert resp.status_code == 400

    def test_rejects_oversized_file(self, client: TestClient) -> None:
        big_data = b"x" * (50 * 1024 * 1024 + 1)
        resp = client.post(
            "/jobs",
            files={"file": ("big.pdf", big_data, "application/pdf")},
            data={"dpi": "150"},
        )
        assert resp.status_code == 400


class TestListJobs:
    def test_list_empty(self, client: TestClient) -> None:
        resp = client.get("/jobs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_after_upload(self, client: TestClient) -> None:
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        client.post("/jobs", files={"file": ("test.pdf", pdf_data, "application/pdf")}, data={"dpi": "150"})
        resp = client.get("/jobs")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_filter_by_status(self, client: TestClient) -> None:
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        client.post("/jobs", files={"file": ("test.pdf", pdf_data, "application/pdf")}, data={"dpi": "150"})
        resp = client.get("/jobs?status=completed")
        assert resp.status_code == 200
        assert len(resp.json()) == 0


class TestGetJob:
    def test_get_existing(self, client: TestClient) -> None:
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = client.post("/jobs", files={"file": ("test.pdf", pdf_data, "application/pdf")}, data={"dpi": "150"})
        job_id = create_resp.json()["job_id"]
        resp = client.get(f"/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == job_id

    def test_get_nonexistent(self, client: TestClient) -> None:
        resp = client.get("/jobs/nonexistent")
        assert resp.status_code == 404


class TestDeleteJob:
    def test_delete_pending(self, client: TestClient) -> None:
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = client.post("/jobs", files={"file": ("test.pdf", pdf_data, "application/pdf")}, data={"dpi": "150"})
        job_id = create_resp.json()["job_id"]
        resp = client.delete(f"/jobs/{job_id}")
        assert resp.status_code == 204

    def test_delete_nonexistent(self, client: TestClient) -> None:
        resp = client.delete("/jobs/nonexistent")
        assert resp.status_code == 404

    def test_delete_processing_returns_409(self, client: TestClient) -> None:
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = client.post("/jobs", files={"file": ("test.pdf", pdf_data, "application/pdf")}, data={"dpi": "150"})
        job_id = create_resp.json()["job_id"]
        # Manually set to processing
        from api.database import update_job
        from api.app import _get_db
        db = _get_db()
        update_job(db, job_id, status="processing")
        resp = client.delete(f"/jobs/{job_id}")
        assert resp.status_code == 409


class TestGetResult:
    def test_returns_404_for_pending(self, client: TestClient) -> None:
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = client.post("/jobs", files={"file": ("test.pdf", pdf_data, "application/pdf")}, data={"dpi": "150"})
        job_id = create_resp.json()["job_id"]
        resp = client.get(f"/jobs/{job_id}/result")
        assert resp.status_code == 404


class TestSearch:
    def test_returns_empty_with_no_completed_jobs(self, client: TestClient) -> None:
        resp = client.post("/search", json={"query": "test"})
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    def test_enqueues_search_to_worker(self, client: TestClient) -> None:
        from concurrent.futures import Future
        future: Future = Future()
        future.set_result([("job1", 0, 0.9)])
        client._mock_worker.enqueue_search.return_value = future

        # Create and manually complete a job
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = client.post("/jobs", files={"file": ("test.pdf", pdf_data, "application/pdf")}, data={"dpi": "150"})
        job_id = create_resp.json()["job_id"]
        from api.app import _get_db
        from api.database import update_job
        db = _get_db()
        update_job(db, job_id, status="completed", page_count=1, duration_ns=100, result_path="r.json", tensor_path="r.pt")

        resp = client.post("/search", json={"query": "charts", "top_k": 5})
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 1
        assert results[0]["file_id"] == "job1"
        client._mock_worker.enqueue_search.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `api/app.py`**

```python
import os
from contextlib import asynccontextmanager
from pathlib import Path

import asyncio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from api.database import (
    create_job,
    delete_job,
    get_connection,
    get_job,
    init_db,
    list_jobs,
)
from api.models import (
    HealthResponse,
    JobCreateResponse,
    JobResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from api.worker import EmbeddingWorker
from core.constants import IMAGE_EXTENSIONS, MAX_UPLOAD_BYTES
from core.embedding import get_device

_db = None
_worker: EmbeddingWorker | None = None


def _get_db():
    global _db
    if _db is None:
        db_path = os.environ.get("DATABASE_PATH", "data/jobs.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        _db = get_connection(db_path)
        init_db(_db)
    return _db


def _get_dirs() -> tuple[Path, Path]:
    uploads = Path(os.environ.get("UPLOAD_DIR", "uploads"))
    results = Path(os.environ.get("RESULT_DIR", "results"))
    uploads.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    return uploads, results


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _worker
        db = _get_db()
        uploads_dir, results_dir = _get_dirs()
        _worker = EmbeddingWorker(db, uploads_dir=uploads_dir, results_dir=results_dir)
        _worker.start()
        yield
        _worker.stop()
        _worker = None

    app = FastAPI(title="Granite Vision Embedding API", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    async def health():
        db = _get_db()
        pending = list_jobs(db, status="pending")
        return HealthResponse(
            device=get_device(),
            queue_depth=len(pending),
            worker_running=_worker.is_running if _worker else False,
        )

    @app.post("/jobs", response_model=JobCreateResponse, status_code=201)
    async def upload_job(file: UploadFile = File(...), dpi: int = Form(150)):
        ext = file.filename.rsplit(".", 1)[-1].lower() if file.filename else ""
        valid_types = IMAGE_EXTENSIONS | {"pdf"}
        if ext not in valid_types:
            raise HTTPException(400, detail=f"Invalid file type: .{ext}")

        content = await file.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(400, detail="File exceeds 50 MB limit")

        file_stem = file.filename.rsplit(".", 1)[0] if file.filename else "upload"
        file_type = "image" if ext in IMAGE_EXTENSIONS else "pdf"

        uploads_dir, _ = _get_dirs()
        db = _get_db()

        # Generate job ID and save file before creating the DB row
        import uuid
        job_id = uuid.uuid4().hex
        upload_path = uploads_dir / f"{job_id}.{ext}"
        upload_path.write_bytes(content)

        create_job(
            db,
            file_name=file.filename or "unknown",
            file_stem=file_stem,
            file_path=str(upload_path),
            file_type=file_type,
            dpi=dpi,
            job_id=job_id,
        )

        return JobCreateResponse(job_id=job_id, status="pending")

    @app.get("/jobs", response_model=list[JobResponse])
    async def list_all_jobs(status: str | None = None):
        db = _get_db()
        jobs = list_jobs(db, status=status)
        return [JobResponse(**{k: v for k, v in j.items() if k in JobResponse.model_fields}) for j in jobs]

    @app.get("/jobs/{job_id}", response_model=JobResponse)
    async def get_single_job(job_id: str):
        db = _get_db()
        job = get_job(db, job_id)
        if not job:
            raise HTTPException(404, detail="Job not found")
        return JobResponse(**{k: v for k, v in job.items() if k in JobResponse.model_fields})

    @app.delete("/jobs/{job_id}", status_code=204)
    async def delete_single_job(job_id: str):
        db = _get_db()
        job = get_job(db, job_id)
        if not job:
            raise HTTPException(404, detail="Job not found")
        if job["status"] == "processing":
            raise HTTPException(409, detail="Cannot delete a processing job")

        # Clean up files
        for path_key in ("file_path", "result_path", "tensor_path"):
            path_str = job.get(path_key)
            if path_str:
                Path(path_str).unlink(missing_ok=True)

        if _worker:
            _worker.evict_cache(job_id)
        delete_job(db, job_id)

    @app.get("/jobs/{job_id}/result")
    async def get_result(job_id: str):
        db = _get_db()
        job = get_job(db, job_id)
        if not job or not job.get("result_path"):
            raise HTTPException(404, detail="Result not available")
        result_path = Path(job["result_path"])
        if not result_path.exists():
            raise HTTPException(404, detail="Result file not found")
        return FileResponse(result_path, media_type="application/json")

    @app.post("/search", response_model=SearchResponse)
    async def search_embeddings(req: SearchRequest):
        if not _worker:
            raise HTTPException(503, detail="Worker not running")

        db = _get_db()
        completed_jobs = list_jobs(db, status="completed")
        job_ids = [j["id"] for j in completed_jobs]

        if not job_ids:
            return SearchResponse(results=[])

        params = {
            "query": req.query,
            "top_k": req.top_k,
            "min_score": req.min_score,
            "filter_file_id": req.filter_file_id,
            "job_ids": job_ids,
        }
        future = _worker.enqueue_search(params)
        results = await asyncio.wrap_future(future)
        return SearchResponse(
            results=[
                SearchResult(file_id=fid, page_index=pidx, score=score)
                for fid, pidx, score in results
            ]
        )

    return app
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run all tests**

Run: `uv run pytest -v`
Expected: All tests in `test_core.py`, `test_database.py`, `test_worker.py`, `test_api.py` PASS

- [ ] **Step 6: Run lint**

Run: `uv run ruff check .`
Expected: No errors

- [ ] **Step 7: Commit**

```bash
git add api/app.py tests/test_api.py
git commit -m "feat: add api/app.py with FastAPI routes and tests"
```

---

## Phase 3: Streamlit API Client

### Task 13: Rewrite `streamlit_app.py` as API client

**Files:**
- Modify: `streamlit_app.py`

- [ ] **Step 1: Rewrite `streamlit_app.py`**

Replace the entire file with a thin API client that communicates with the FastAPI backend via `httpx.Client`. Note: page image previews in search results are not included in this version — the API does not serve rendered page images. This can be added as a future enhancement via a `GET /jobs/{id}/pages/{page_index}` endpoint.

```python
import os

import httpx
import streamlit as st

from core.constants import DPI_OPTIONS, IMAGE_EXTENSIONS, MODEL_ID

API_URL = os.environ.get("API_URL", "http://localhost:8000")


def api_client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=120.0)


st.set_page_config(page_title="Granite Vision Embedding Pipeline", layout="centered")
st.title("Granite Vision Embedding Pipeline")
st.write(
    "Generate vector embeddings from PDFs and images with Granite Vision Embedding Pipeline."
)

uploaded_files = st.file_uploader(
    "Upload files",
    type=["pdf", *IMAGE_EXTENSIONS],
    accept_multiple_files=True,
)

dpi_label = st.radio("Render DPI", DPI_OPTIONS, index=1, horizontal=True)
dpi = DPI_OPTIONS[dpi_label]

if uploaded_files:
    total_size_mb = sum(len(f.getvalue()) for f in uploaded_files) / 1_048_576
    st.caption(f"{len(uploaded_files)} file(s) · {total_size_mb:.1f} MB")

    if st.button("Submit Jobs", type="primary"):
        with api_client() as client:
            for f in uploaded_files:
                try:
                    resp = client.post(
                        "/jobs",
                        files={"file": (f.name, f.getvalue())},
                        data={"dpi": str(dpi)},
                    )
                    if resp.status_code == 201:
                        st.success(f"Submitted: {f.name}")
                    else:
                        st.error(f"{f.name}: {resp.json().get('detail', 'Unknown error')}")
                except httpx.HTTPError as e:
                    st.error(f"{f.name}: {e}")

# Job Dashboard
st.subheader("Jobs")

col_refresh, col_filter = st.columns([1, 2])
if col_refresh.button("Refresh"):
    st.rerun()

status_filter = col_filter.selectbox(
    "Status filter",
    ["all", "pending", "processing", "completed", "failed"],
)

try:
    with api_client() as client:
        params = {} if status_filter == "all" else {"status": status_filter}
        resp = client.get("/jobs", params=params)
        jobs = resp.json() if resp.status_code == 200 else []
except httpx.HTTPError:
    jobs = []
    st.warning("Cannot connect to API server.")

if jobs:
    # Summary metrics
    completed = [j for j in jobs if j["status"] == "completed"]
    total_pages = sum(j.get("page_count") or 0 for j in completed)
    total_duration_ns = sum(j.get("duration_ns") or 0 for j in completed)

    if completed:
        st.subheader("Metrics")
        st.metric("Model", MODEL_ID)
        col1, col2, col3 = st.columns(3)
        col1.metric("Duration", f"{total_duration_ns / 1_000_000_000:.2f} s")
        col2.metric("Pages", total_pages)
        col3.metric("Documents", len(completed))

    for job in jobs:
        status_emoji = {
            "pending": "Pending",
            "processing": "Processing",
            "completed": "Completed",
            "failed": "Failed",
        }.get(job["status"], job["status"])

        with st.expander(f"{job['file_stem']} — {status_emoji}"):
            st.caption(
                f"Status: {job['status']} · Type: {job['file_type']} · DPI: {job['dpi']}"
            )
            if job.get("page_count"):
                duration_s = (job.get("duration_ns") or 0) / 1_000_000_000
                st.caption(f"Pages: {job['page_count']} · Duration: {duration_s:.2f} s")
            if job.get("error"):
                st.error(job["error"])

            col_dl, col_del = st.columns(2)

            if job["status"] == "completed":
                try:
                    with api_client() as client:
                        result_resp = client.get(f"/jobs/{job['id']}/result")
                        if result_resp.status_code == 200:
                            col_dl.download_button(
                                f"Download {job['file_stem']} JSON",
                                data=result_resp.content,
                                file_name=f"{job['file_stem']}_embedding.json",
                                mime="application/json",
                                key=f"dl_{job['id']}",
                            )
                except httpx.HTTPError:
                    pass

            if job["status"] != "processing":
                if col_del.button("Delete", key=f"del_{job['id']}"):
                    try:
                        with api_client() as client:
                            client.delete(f"/jobs/{job['id']}")
                        st.rerun()
                    except httpx.HTTPError as e:
                        st.error(str(e))

    # Download All
    if len(completed) > 1:
        try:
            with api_client() as client:
                all_results = []
                for j in completed:
                    result_resp = client.get(f"/jobs/{j['id']}/result")
                    if result_resp.status_code == 200:
                        all_results.append(result_resp.text)
                if all_results:
                    all_json = "[" + ",".join(all_results) + "]"
                    st.download_button(
                        "Download All JSON",
                        data=all_json,
                        file_name="all_embeddings.json",
                        mime="application/json",
                        key="download_all",
                    )
        except httpx.HTTPError:
            pass

    # Search
    if completed:
        st.subheader("Search")
        filter_options = ["All documents"] + [j["file_stem"] for j in completed]
        filter_ids: list[str | None] = [None] + [j["id"] for j in completed]
        filter_idx = st.selectbox(
            "Document filter",
            range(len(filter_options)),
            format_func=lambda i: filter_options[i],
        )

        col_topk, col_minscore = st.columns(2)
        top_k = col_topk.number_input("Top K", min_value=1, max_value=100, value=5)
        min_score = col_minscore.number_input(
            "Min score", min_value=0.0, value=0.0, step=0.1
        )

        query = st.text_input("Text query")
        if st.button("Search"):
            if not query:
                st.warning("Enter a search query.")
            else:
                try:
                    with api_client() as client:
                        search_resp = client.post(
                            "/search",
                            json={
                                "query": query,
                                "top_k": top_k,
                                "min_score": min_score,
                                "filter_file_id": filter_ids[filter_idx],
                            },
                        )
                        if search_resp.status_code == 200:
                            st.session_state.search_results = search_resp.json()["results"]
                        else:
                            st.error(search_resp.json().get("detail", "Search failed"))
                except httpx.HTTPError as e:
                    st.error(str(e))

        if "search_results" in st.session_state:
            search_results = st.session_state.search_results
            if search_results:
                job_lookup = {j["id"]: j for j in completed}
                for rank, sr in enumerate(search_results):
                    j = job_lookup.get(sr["file_id"])
                    if j:
                        st.caption(
                            f"{j['file_stem']} · Page {sr['page_index'] + 1} · {sr['score']:.4f}"
                        )
            else:
                st.info("No results above the score threshold.")

elif not uploaded_files:
    st.info("Upload files to get started.")

# Footer
try:
    with api_client() as client:
        health = client.get("/health").json()
        device = health.get("device", "unknown").upper()
        queue_depth = health.get("queue_depth", 0)
        st.caption(f"Device: {device} · Queue: {queue_depth}")
except httpx.HTTPError:
    st.caption("API server not connected")
```

- [ ] **Step 2: Run lint**

Run: `uv run ruff check .`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: rewrite streamlit_app.py as FastAPI client"
```

---

## Phase 4: Configuration and Documentation

### Task 14: Update `.gitignore`

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add new directories to `.gitignore`**

Append to `.gitignore`:

```
uploads/
results/
data/
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: add uploads/, results/, data/ to .gitignore"
```

---

### Task 15: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md**

Update the following sections to reflect the new architecture:

- **Setup**: Add API server startup command
- **Commands**: Add `uv run uvicorn api.app:create_app --factory --port 8000`
- **Dependencies**: Add fastapi, uvicorn, httpx, python-multipart
- **Architecture**: Update to describe `core/`, `api/`, and thin Streamlit client
- **Constants**: Add `MAX_UPLOAD_BYTES`
- **Tests**: Update test file list to `test_core.py`, `test_api.py`, `test_worker.py`, `test_database.py`

Also update `README.md` with two-process startup instructions.

- [ ] **Step 2: Run lint on all files**

Run: `uv run ruff check .`
Expected: No errors

- [ ] **Step 3: Run all tests**

Run: `uv run pytest -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for batch processing architecture"
```
