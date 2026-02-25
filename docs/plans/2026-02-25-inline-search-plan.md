# Inline Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add text query search over embedded PDF pages, showing results as ranked thumbnails with relevance scores.

**Architecture:** After embedding a PDF, a search section appears with a text input and button. The query is embedded with the same model, scored against page embeddings using `processor.score()`, and pages are displayed ranked by similarity. All changes in `streamlit_app.py` and `tests/test_app.py`.

**Tech Stack:** Streamlit, colpali-engine (BiQwen2_5, BiQwen2_5_Processor), PyTorch

---

### Task 1: Change `embed()` to return a tensor

The `embed()` function currently calls `.tolist()` to convert the model output to nested Python lists. We need the raw tensor for scoring, so move the `.tolist()` call to the JSON serialization site.

**Files:**
- Modify: `streamlit_app.py:45-52` (`embed` function)
- Modify: `streamlit_app.py:110-114` (JSON serialization)
- Modify: `tests/test_app.py:45-81` (embed tests)

**Step 1: Update `embed()` return type and remove `.tolist()`**

In `streamlit_app.py`, change the `embed` function:

```python
def embed(
    images: list[Image.Image], model: BiQwen2_5, processor: BiQwen2_5_Processor
) -> torch.Tensor:
    """Generate per-page multi-vector embeddings from images."""
    batch = processor.process_images(images).to(model.device)
    with torch.inference_mode():
        embeddings = model(**batch)
    return embeddings
```

**Step 2: Update JSON serialization to call `.tolist()`**

In the UI block where `embedding_data` is built (~line 110), change:

```python
"embeddings": page_embeddings,
```

to:

```python
"embeddings": page_embeddings.tolist(),
```

**Step 3: Update tests for tensor return type**

In `tests/test_app.py`, update `TestEmbed.test_returns_per_page_embeddings`:

```python
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

    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (num_pages, num_patches, embedding_dim)
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_app.py -v`
Expected: all 7 tests PASS

**Step 5: Run lint**

Run: `uv run ruff check . && uv run ruff format .`
Expected: no errors

---

### Task 2: Add `search()` function with test

**Files:**
- Modify: `streamlit_app.py` (add `search` function after `embed`)
- Modify: `tests/test_app.py` (add `TestSearch` class)

**Step 1: Write the test**

Add to `tests/test_app.py`:

- Import `search` in the import line
- Add a new test class:

```python
class TestSearch:
    def test_returns_ranked_results(self) -> None:
        mock_processor = MagicMock()
        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_batch
        mock_processor.process_texts.return_value = mock_batch

        query_embedding = torch.tensor([[0.1, 0.2, 0.3]])
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.return_value = query_embedding

        scores = torch.tensor([[0.5, 0.9, 0.2]])
        mock_processor.score.return_value = scores

        image_embeddings = torch.randn(3, 128)

        results = search("test query", mock_model, mock_processor, image_embeddings)

        assert len(results) == 3
        assert results[0] == (1, 0.9)
        assert results[1] == (0, 0.5)
        assert results[2] == (2, 0.2)
        mock_processor.process_texts.assert_called_once_with(["test query"])
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_app.py::TestSearch -v`
Expected: ImportError — `search` not defined

**Step 3: Implement `search()`**

Add to `streamlit_app.py` after the `embed` function:

```python
def search(
    query: str,
    model: BiQwen2_5,
    processor: BiQwen2_5_Processor,
    image_embeddings: torch.Tensor,
) -> list[tuple[int, float]]:
    """Score a text query against image embeddings and return ranked results."""
    batch = processor.process_texts([query]).to(model.device)
    with torch.inference_mode():
        query_embedding = model(**batch)
    scores = processor.score(
        qs=[query_embedding[0]],
        ps=[emb for emb in image_embeddings],
    )
    ranked = sorted(
        [(i, score.item()) for i, score in enumerate(scores[0])],
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_app.py::TestSearch -v`
Expected: PASS

**Step 5: Run all tests and lint**

Run: `uv run pytest -v && uv run ruff check . && uv run ruff format .`
Expected: all tests PASS, no lint errors

---

### Task 3: Add search UI

**Files:**
- Modify: `streamlit_app.py` (UI block, after download button)

**Step 1: Add search UI after the download button**

Inside the `if st.button("Embed")` block, after the `st.download_button(...)` call and before the `except` blocks, add:

```python
            st.subheader("Search")
            query = st.text_input("Text query")
            if st.button("Search") and query:
                search_results = search(query, model, processor, page_embeddings)
                cols = st.columns(min(len(search_results), 4))
                for rank, (page_idx, score) in enumerate(search_results):
                    cols[rank % 4].image(
                        pages[page_idx],
                        caption=f"Page {page_idx + 1} · {score:.4f}",
                        width="stretch",
                    )
```

**Step 2: Run lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: no errors

**Step 3: Run all tests**

Run: `uv run pytest -v`
Expected: all tests PASS

---

### Task 4: Final verification

**Step 1: Run full verification suite**

Run: `uv run ruff check . && uv run ruff format --check . && uv run pytest -v`
Expected: all checks pass, all tests pass

**Step 2: Commit**

Commit all changes with a descriptive message.
