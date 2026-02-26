# Configurable Render DPI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let users control PDF render resolution (72–300 DPI) via an inline slider to improve embedding quality or save resources.

**Architecture:** Add a `dpi` parameter to `render_pages`, use `fitz.Matrix` to scale rendering, expose a Streamlit slider in the UI, store DPI in session state and the JSON export.

**Tech Stack:** pymupdf (`fitz.Matrix`), Streamlit (`st.slider`), pytest

---

### Task 1: Add DPI parameter to `render_pages`

**Files:**
- Modify: `streamlit_app.py:32-42`
- Test: `tests/test_app.py`

**Step 1: Write the failing tests**

Add to `TestRenderPages` in `tests/test_app.py`:

```python
def test_higher_dpi_produces_larger_images(self) -> None:
    data = (FIXTURE_DIR / "test.pdf").read_bytes()
    pages_72 = render_pages(data, dpi=72)
    pages_150 = render_pages(data, dpi=150)
    assert pages_150[0].width > pages_72[0].width
    assert pages_150[0].height > pages_72[0].height

def test_default_dpi_is_150(self) -> None:
    data = (FIXTURE_DIR / "test.pdf").read_bytes()
    pages_default = render_pages(data)
    pages_150 = render_pages(data, dpi=150)
    assert pages_default[0].size == pages_150[0].size
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_app.py::TestRenderPages::test_higher_dpi_produces_larger_images tests/test_app.py::TestRenderPages::test_default_dpi_is_150 -v`
Expected: FAIL — `render_pages() got an unexpected keyword argument 'dpi'`

**Step 3: Implement `dpi` parameter**

Change `render_pages` in `streamlit_app.py:32-42` to:

```python
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

**Step 4: Run all `TestRenderPages` tests**

Run: `uv run pytest tests/test_app.py::TestRenderPages -v`
Expected: PASS (all 5 tests including existing ones)

**Step 5: Commit**

```bash
git add streamlit_app.py tests/test_app.py
git commit -m "feat: add dpi parameter to render_pages"
```

---

### Task 2: Add DPI slider and wire it into the UI

**Files:**
- Modify: `streamlit_app.py:86-101` (slider + session state clear)
- Modify: `streamlit_app.py:113` (pass dpi to render_pages)
- Modify: `streamlit_app.py:128-133` (store dpi in session state)

**Step 1: Add the slider**

After line 88 (`st.caption(...)`) in `streamlit_app.py`, add:

```python
    dpi = st.slider(
        "Render DPI",
        min_value=72,
        max_value=300,
        value=150,
        step=1,
        help="72 = Low · 150 = Medium · 300 = High",
    )
```

**Step 2: Add `"dpi"` to session state keys cleared on new file upload**

In `streamlit_app.py:92-100`, add `"dpi"` to the tuple of keys:

```python
        for key in (
            "pages",
            "page_embeddings",
            "total_duration",
            "file_stem",
            "search_results",
            "file_id",
            "dpi",
        ):
```

**Step 3: Pass `dpi` to `render_pages`**

Change `streamlit_app.py:113` from:

```python
            pages = render_pages(uploaded_file.read())
```

to:

```python
            pages = render_pages(uploaded_file.read(), dpi=dpi)
```

**Step 4: Store `dpi` in session state after embedding**

After `st.session_state.file_stem = ...` (line 132), add:

```python
            st.session_state.dpi = dpi
```

**Step 5: Run lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: No errors

**Step 6: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add DPI slider to UI"
```

---

### Task 3: Add DPI to metrics and JSON output

**Files:**
- Modify: `streamlit_app.py:140-171` (metrics display + JSON)

**Step 1: Read `dpi` from session state**

After line 144 (`file_stem = st.session_state.file_stem`), add:

```python
        dpi = st.session_state.dpi
```

**Step 2: Add DPI metric**

Change the metrics section from:

```python
        st.subheader("Metrics")
        st.metric("Model", MODEL_ID)
        col1, col2 = st.columns(2)
        duration_s = total_duration / 1_000_000_000
        col1.metric("Duration", f"{duration_s:.2f} s")
        col2.metric("Page Count", len(pages))
```

to:

```python
        st.subheader("Metrics")
        st.metric("Model", MODEL_ID)
        col1, col2, col3 = st.columns(3)
        duration_s = total_duration / 1_000_000_000
        col1.metric("Duration", f"{duration_s:.2f} s")
        col2.metric("Page Count", len(pages))
        col3.metric("DPI", dpi)
```

**Step 3: Add `dpi` to JSON output**

Change the `embedding_data` dict from:

```python
        embedding_data = {
            "model": MODEL_ID,
            "embeddings": page_embeddings.tolist(),
            "total_duration": total_duration,
            "page_count": len(pages),
        }
```

to:

```python
        embedding_data = {
            "model": MODEL_ID,
            "dpi": dpi,
            "embeddings": page_embeddings.tolist(),
            "total_duration": total_duration,
            "page_count": len(pages),
        }
```

**Step 4: Run lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: No errors

**Step 5: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add DPI to metrics and JSON output"
```

---

### Task 4: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update the JSON Download section**

Add `dpi (integer)` to the JSON fields list in `CLAUDE.md`, after `model`:

```markdown
- `model` (string) — model that produced the embeddings
- `dpi` (integer) — render resolution in dots per inch (72–300)
- `embeddings` (number[][][]) — per-page multi-vector embeddings (page → patches → 128-dim vectors)
```

**Step 2: Update the Pipeline section**

Change the pipeline description from:

```markdown
PDF upload → render pages as images (`pymupdf`) → embed images (`BiQwen2_5`) → download JSON / search pages by text query
```

to:

```markdown
PDF upload → render pages as images at configurable DPI (`pymupdf`) → embed images (`BiQwen2_5`) → download JSON / search pages by text query
```

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for configurable DPI"
```

---

### Task 5: Final verification

**Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS

**Step 2: Run lint, format, and typecheck**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check`
Expected: No errors

**Step 3: Run the app manually (optional)**

Run: `uv run streamlit run streamlit_app.py`
Verify: Slider appears between file caption and Embed button, DPI shows in metrics, JSON includes `dpi` field.
