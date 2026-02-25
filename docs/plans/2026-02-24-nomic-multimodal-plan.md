# Nomic Embed Multimodal 3B Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace IBM Granite text-only embedding with nomic-embed-multimodal-3b, changing the pipeline from PDF-to-Markdown-to-embedding to PDF-to-page-images-to-embedding.

**Architecture:** Single-file Streamlit app. PDF pages are rendered as PIL Images via pymupdf, then embedded using BiQwen2_5 from colpali-engine. Each page produces a matrix of 128-dim patch vectors. Output JSON contains per-page embeddings.

**Tech Stack:** colpali-engine (BiQwen2_5, BiQwen2_5_Processor), pymupdf, Pillow, torch, streamlit

---

### Task 1: Update dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update pyproject.toml**

Replace docling and transformers with colpali-engine and pymupdf. Update project description.

```toml
[project]
name = "embedding-pipeline"
version = "0.1.0"
description = "Streamlit web app for generating vector embeddings from PDF documents using Nomic's multimodal embedding model."
requires-python = ">=3.12"
dependencies = [
    "colpali-engine>=0.3.3",
    "pymupdf>=1.25.0",
    "streamlit==1.52.2",
    "torch==2.9.1",
]

[dependency-groups]
dev = [
    "pytest==9.0.2",
    "ruff==0.15.0",
    "ty==0.0.17",
]

[tool.ruff.lint.isort]
combine-as-imports = true

[tool.ty.environment]
python-version = "3.12"
```

**Step 2: Sync dependencies**

Run: `uv sync`
Expected: Dependencies install successfully

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Replace docling/transformers with colpali-engine/pymupdf"
```

---

### Task 2: Write render_pages tests and implementation

**Files:**
- Modify: `tests/test_app.py`
- Modify: `streamlit_app.py`

**Step 1: Write the failing test**

Add to `tests/test_app.py`, replacing `TestConvert` and `TestBuildPipelineOptions`:

```python
from PIL import Image

from streamlit_app import render_pages

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class TestRenderPages:
    def test_renders_fixture_pdf(self) -> None:
        pages = render_pages(str(FIXTURE_DIR / "test.pdf"))
        assert len(pages) >= 1
        assert all(isinstance(p, Image.Image) for p in pages)
        assert all(p.mode == "RGB" for p in pages)

    def test_returns_empty_for_no_pages(self, tmp_path: Path) -> None:
        empty_pdf = tmp_path / "empty.pdf"
        empty_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
        pages = render_pages(str(empty_pdf))
        assert pages == []
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_app.py::TestRenderPages -v`
Expected: FAIL — `ImportError: cannot import name 'render_pages'`

**Step 3: Write render_pages in streamlit_app.py**

```python
import fitz
from PIL import Image


def render_pages(source: str) -> list[Image.Image]:
    """Render PDF pages as PIL Images."""
    doc = fitz.open(source)
    pages = []
    for page in doc:
        pix = page.get_pixmap()
        pages.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    doc.close()
    return pages
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_app.py::TestRenderPages -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_app.py streamlit_app.py
git commit -m "Add render_pages function for PDF-to-image conversion"
```

---

### Task 3: Write embed tests and implementation

**Files:**
- Modify: `tests/test_app.py`
- Modify: `streamlit_app.py`

**Step 1: Write the failing test**

Replace `TestEmbed` in `tests/test_app.py`:

```python
class TestEmbed:
    def test_returns_per_page_embeddings(self) -> None:
        num_pages = 2
        num_patches = 4
        embedding_dim = 128

        mock_processor = MagicMock()
        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_batch
        mock_processor.process_images.return_value = mock_batch

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.return_value = torch.randn(num_pages, num_patches, embedding_dim)

        images = [Image.new("RGB", (64, 64)) for _ in range(num_pages)]
        embeddings = embed(images, mock_model, mock_processor)

        assert len(embeddings) == num_pages
        assert all(len(page) == num_patches for page in embeddings)
        assert all(len(vec) == embedding_dim for page in embeddings for vec in page)

    def test_calls_process_images(self) -> None:
        mock_processor = MagicMock()
        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_batch
        mock_processor.process_images.return_value = mock_batch

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.return_value = torch.randn(1, 4, 128)

        images = [Image.new("RGB", (64, 64))]
        embed(images, mock_model, mock_processor)

        mock_processor.process_images.assert_called_once_with(images)
        mock_batch.to.assert_called_once_with("cpu")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_app.py::TestEmbed -v`
Expected: FAIL — signature mismatch or import error

**Step 3: Write new embed function in streamlit_app.py**

```python
def embed(
    images: list[Image.Image], model: BiQwen2_5, processor: BiQwen2_5_Processor
) -> list[list[list[float]]]:
    """Generate per-page multi-vector embeddings from images."""
    batch = processor.process_images(images).to(model.device)
    with torch.inference_mode():
        embeddings = model(**batch)
    return embeddings.tolist()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_app.py::TestEmbed -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_app.py streamlit_app.py
git commit -m "Rewrite embed() for multimodal image embeddings"
```

---

### Task 4: Rewrite streamlit_app.py (full refactor)

**Files:**
- Modify: `streamlit_app.py`

**Step 1: Rewrite the full app**

Remove all Docling/Granite code and rebuild the UI. The final `streamlit_app.py`:

```python
import json
import tempfile
import time
from pathlib import Path

import fitz
import streamlit as st
import torch
from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor
from PIL import Image

MODEL_ID = "nomic-ai/nomic-embed-multimodal-3b"
MAX_PDF_PAGES = 100
MAX_FILE_SIZE_BYTES = 20_971_520  # 20MB


def get_device() -> str:
    """Detect best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@st.cache_resource
def load_model(device: str) -> tuple[BiQwen2_5, BiQwen2_5_Processor]:
    """Load embedding model and processor."""
    model = BiQwen2_5.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=device
    ).eval()
    processor = BiQwen2_5_Processor.from_pretrained(MODEL_ID)
    return model, processor


def render_pages(source: str) -> list[Image.Image]:
    """Render PDF pages as PIL Images."""
    doc = fitz.open(source)
    pages = []
    for page in doc:
        pix = page.get_pixmap()
        pages.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    doc.close()
    return pages


def embed(
    images: list[Image.Image], model: BiQwen2_5, processor: BiQwen2_5_Processor
) -> list[list[list[float]]]:
    """Generate per-page multi-vector embeddings from images."""
    batch = processor.process_images(images).to(model.device)
    with torch.inference_mode():
        embeddings = model(**batch)
    return embeddings.tolist()


# UI
st.title("Embedding Pipeline")
st.write(
    "Generate vector embeddings from PDF documents with Nomic Embed Multimodal."
)

uploaded_file = st.file_uploader("Upload file", type=["pdf"])

device = get_device()

if st.button("Embed", type="primary"):
    if uploaded_file is None:
        st.warning("Upload a PDF file.")
    else:
        tmp_file_path = None
        try:
            total_start = time.perf_counter_ns()

            # Load model
            with st.spinner(f"Loading model on {device.upper()}..."):
                model, processor = load_model(device)

            # Render PDF pages as images
            with st.spinner("Rendering pages..."):
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp_file.write(uploaded_file.read())
                tmp_file.close()
                tmp_file_path = tmp_file.name
                pages = render_pages(tmp_file_path)

            if not pages:
                raise ValueError("PDF contains no pages to embed.")

            # Generate embeddings
            with st.spinner("Generating embeddings..."):
                page_embeddings = embed(pages, model, processor)

            total_duration = time.perf_counter_ns() - total_start

            # Display results
            st.success("Done.")

            st.subheader("Metrics")
            st.metric("Model", MODEL_ID)
            col1, col2 = st.columns(2)
            col1.metric("Total Duration (ns)", f"{total_duration:,}")
            col2.metric("Page Count", len(pages))

            embedding_data = {
                "model": MODEL_ID,
                "embeddings": page_embeddings,
                "total_duration": total_duration,
                "page_count": len(pages),
            }
            st.download_button(
                label="Download JSON",
                data=json.dumps(embedding_data),
                file_name="embedding.json",
                mime="application/json",
            )

        except OSError as e:
            st.error(f"File error: {e}")
        except RuntimeError as e:
            st.error(f"Model error: {e}")
        except ValueError as e:
            st.error(f"Processing error: {e}")
        except Exception as e:
            st.exception(e)

        finally:
            if tmp_file_path:
                Path(tmp_file_path).unlink(missing_ok=True)
```

**Step 2: Run all tests**

Run: `uv run pytest -v`
Expected: All tests pass

**Step 3: Lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: No errors

**Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "Refactor app for nomic-embed-multimodal-3b pipeline"
```

---

### Task 5: Update tests (final cleanup)

**Files:**
- Modify: `tests/test_app.py`

**Step 1: Write final test file**

The complete `tests/test_app.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from streamlit_app import embed, get_device, render_pages

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class TestGetDevice:
    @patch("streamlit_app.torch")
    def test_prefers_mps(self, mock_torch: MagicMock) -> None:
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.cuda.is_available.return_value = True
        device = get_device()
        assert device == "mps"

    @patch("streamlit_app.torch")
    def test_falls_back_to_cuda(self, mock_torch: MagicMock) -> None:
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = True
        device = get_device()
        assert device == "cuda"

    @patch("streamlit_app.torch")
    def test_falls_back_to_cpu(self, mock_torch: MagicMock) -> None:
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False
        device = get_device()
        assert device == "cpu"


class TestRenderPages:
    def test_renders_fixture_pdf(self) -> None:
        pages = render_pages(str(FIXTURE_DIR / "test.pdf"))
        assert len(pages) >= 1
        assert all(isinstance(p, Image.Image) for p in pages)
        assert all(p.mode == "RGB" for p in pages)

    def test_returns_empty_for_no_pages(self, tmp_path: Path) -> None:
        empty_pdf = tmp_path / "empty.pdf"
        empty_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
        pages = render_pages(str(empty_pdf))
        assert pages == []


class TestEmbed:
    def test_returns_per_page_embeddings(self) -> None:
        num_pages = 2
        num_patches = 4
        embedding_dim = 128

        mock_processor = MagicMock()
        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_batch
        mock_processor.process_images.return_value = mock_batch

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.return_value = torch.randn(num_pages, num_patches, embedding_dim)

        images = [Image.new("RGB", (64, 64)) for _ in range(num_pages)]
        embeddings = embed(images, mock_model, mock_processor)

        assert len(embeddings) == num_pages
        assert all(len(page) == num_patches for page in embeddings)
        assert all(len(vec) == embedding_dim for page in embeddings for vec in page)

    def test_calls_process_images(self) -> None:
        mock_processor = MagicMock()
        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_batch
        mock_processor.process_images.return_value = mock_batch

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.return_value = torch.randn(1, 4, 128)

        images = [Image.new("RGB", (64, 64))]
        embed(images, mock_model, mock_processor)

        mock_processor.process_images.assert_called_once_with(images)
        mock_batch.to.assert_called_once_with("cpu")
```

**Step 2: Run all tests**

Run: `uv run pytest -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add tests/test_app.py
git commit -m "Update tests for multimodal embedding pipeline"
```

---

### Task 6: Update CLAUDE.md and README.md

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Step 1: Update CLAUDE.md**

Key changes:
- Update project description to mention Nomic multimodal model
- Remove Docling/transformers from dependencies, add colpali-engine/pymupdf
- Remove PDF Pipeline Options section (TableFormer, enrichment toggles)
- Replace Embedding Models section with single model: `nomic-ai/nomic-embed-multimodal-3b`
- Update Architecture to reflect new pipeline: PDF upload -> render pages -> embed images
- Update `convert` -> `render_pages` in function list
- Update `embed` signature description
- Remove `AcceleratorDevice` from `get_device()` return
- Update JSON download fields: replace `prompt_eval_count` with `page_count`, note `embeddings` is `number[][][]`
- Update test descriptions
- Update Resources section

**Step 2: Update README.md**

Update to match the new project description and model.

**Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "Update docs for nomic multimodal pipeline"
```

---

### Task 7: Final verification

**Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass

**Step 2: Lint and typecheck**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check`
Expected: No errors

**Step 3: Verify app starts**

Run: `uv run streamlit run streamlit_app.py --server.headless true &` then stop it.
Expected: App starts without import errors.
