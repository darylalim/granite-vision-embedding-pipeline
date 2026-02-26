# Multi-PDF Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable multi-PDF upload, batch embedding, and cross-document search with per-document filtering.

**Architecture:** Replace single-file uploader with `accept_multiple_files=True`. Session state stores a `dict[str, EmbedResults]` keyed by `file_id`. Two new pure functions (`cleanup_stale_results`, `search_multi`) handle state cleanup and cross-document search. UI adds Embed/Re-embed All buttons, per-document expanders, Download All, and a search filter dropdown.

**Tech Stack:** Streamlit (`st.file_uploader`, `st.selectbox`, `st.expander`), pytest, torch

---

### Task 1: Add `cleanup_stale_results` function

**Files:**
- Modify: `streamlit_app.py` (add function after `search`)
- Test: `tests/test_app.py`

**Step 1: Write the failing tests**

Add a helper and test class to `tests/test_app.py`. First, update the import line:

```python
from streamlit_app import (
    DPI_OPTIONS,
    EmbedResults,
    cleanup_stale_results,
    embed,
    get_device,
    render_pages,
    search,
)
```

Add a helper function above `TestDpiOptions`:

```python
def _make_result(
    file_id: str,
    file_stem: str,
    *,
    page_embeddings: torch.Tensor | None = None,
) -> EmbedResults:
    return {
        "file_id": file_id,
        "pages": [],
        "page_embeddings": page_embeddings if page_embeddings is not None else torch.empty(0),
        "total_duration": 0,
        "file_stem": file_stem,
        "dpi": 150,
        "json": "{}",
    }
```

Add the test class after `TestSearch`:

```python
class TestCleanupStaleResults:
    def test_removes_stale_entries(self) -> None:
        results: dict[str, EmbedResults] = {
            "id1": _make_result("id1", "doc1"),
            "id2": _make_result("id2", "doc2"),
        }
        cleanup_stale_results({"id1"}, results)
        assert "id1" in results
        assert "id2" not in results

    def test_keeps_all_when_none_stale(self) -> None:
        results: dict[str, EmbedResults] = {
            "id1": _make_result("id1", "doc1"),
        }
        cleanup_stale_results({"id1"}, results)
        assert len(results) == 1

    def test_removes_all_when_all_stale(self) -> None:
        results: dict[str, EmbedResults] = {
            "id1": _make_result("id1", "doc1"),
        }
        cleanup_stale_results(set(), results)
        assert len(results) == 0

    def test_handles_empty_results(self) -> None:
        results: dict[str, EmbedResults] = {}
        cleanup_stale_results({"id1"}, results)
        assert len(results) == 0
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_app.py::TestCleanupStaleResults -v`
Expected: FAIL — `ImportError: cannot import name 'cleanup_stale_results'`

**Step 3: Implement `cleanup_stale_results`**

Add after the `search` function in `streamlit_app.py`:

```python
def cleanup_stale_results(
    current_file_ids: set[str], results: dict[str, EmbedResults]
) -> None:
    """Remove results for files no longer in the uploader."""
    for file_id in set(results.keys()) - current_file_ids:
        del results[file_id]
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_app.py::TestCleanupStaleResults -v`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add streamlit_app.py tests/test_app.py
git commit -m "feat: add cleanup_stale_results function"
```

---

### Task 2: Add `search_multi` function

**Files:**
- Modify: `streamlit_app.py` (add function after `cleanup_stale_results`)
- Modify: `tests/test_app.py`

**Step 1: Write the failing tests**

Update the import in `tests/test_app.py` to add `search_multi`:

```python
from streamlit_app import (
    DPI_OPTIONS,
    EmbedResults,
    cleanup_stale_results,
    embed,
    get_device,
    render_pages,
    search,
    search_multi,
)
```

Add after `TestCleanupStaleResults`:

```python
class TestSearchMulti:
    def test_returns_cross_document_ranked_results(self) -> None:
        mock_processor = MagicMock()
        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_batch
        mock_processor.process_texts.return_value = mock_batch

        query_embedding = torch.tensor([[0.1, 0.2, 0.3]])
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.return_value = query_embedding

        # Doc A: 2 pages scoring [0.3, 0.8], Doc B: 1 page scoring [0.6]
        mock_processor.score.side_effect = [
            torch.tensor([[0.3, 0.8]]),
            torch.tensor([[0.6]]),
        ]

        results: dict[str, EmbedResults] = {
            "id_a": _make_result("id_a", "doc_a", page_embeddings=torch.randn(2, 4, 128)),
            "id_b": _make_result("id_b", "doc_b", page_embeddings=torch.randn(1, 4, 128)),
        }

        ranked = search_multi("test query", mock_model, mock_processor, results)

        assert len(ranked) == 3
        assert ranked[0] == ("id_a", 1, approx(0.8))
        assert ranked[1] == ("id_b", 0, approx(0.6))
        assert ranked[2] == ("id_a", 0, approx(0.3))

    def test_filters_to_single_document(self) -> None:
        mock_processor = MagicMock()
        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_batch
        mock_processor.process_texts.return_value = mock_batch

        query_embedding = torch.tensor([[0.1, 0.2, 0.3]])
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.return_value = query_embedding

        mock_processor.score.return_value = torch.tensor([[0.5]])

        results: dict[str, EmbedResults] = {
            "id_a": _make_result("id_a", "doc_a", page_embeddings=torch.randn(1, 4, 128)),
            "id_b": _make_result("id_b", "doc_b", page_embeddings=torch.randn(1, 4, 128)),
        }

        ranked = search_multi(
            "test query", mock_model, mock_processor, results, filter_file_id="id_b"
        )

        assert len(ranked) == 1
        assert ranked[0][0] == "id_b"
        assert mock_processor.score.call_count == 1

    def test_encodes_query_once_for_multiple_docs(self) -> None:
        mock_processor = MagicMock()
        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_batch
        mock_processor.process_texts.return_value = mock_batch

        query_embedding = torch.tensor([[0.1, 0.2, 0.3]])
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.return_value = query_embedding

        mock_processor.score.side_effect = [
            torch.tensor([[0.3]]),
            torch.tensor([[0.6]]),
        ]

        results: dict[str, EmbedResults] = {
            "id_a": _make_result("id_a", "doc_a", page_embeddings=torch.randn(1, 4, 128)),
            "id_b": _make_result("id_b", "doc_b", page_embeddings=torch.randn(1, 4, 128)),
        }

        search_multi("test", mock_model, mock_processor, results)

        mock_processor.process_texts.assert_called_once_with(["test"])
        assert mock_model.call_count == 1
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_app.py::TestSearchMulti -v`
Expected: FAIL — `ImportError: cannot import name 'search_multi'`

**Step 3: Implement `search_multi`**

Add after `cleanup_stale_results` in `streamlit_app.py`:

```python
def search_multi(
    query: str,
    model: BiQwen2_5,
    processor: BiQwen2_5_Processor,
    results: dict[str, EmbedResults],
    filter_file_id: str | None = None,
) -> list[tuple[str, int, float]]:
    """Score a text query across multiple documents and return ranked results."""
    docs = {filter_file_id: results[filter_file_id]} if filter_file_id else results
    batch = processor.process_texts([query]).to(model.device)
    with torch.inference_mode():
        query_embedding = model(**batch)
    ranked: list[tuple[str, int, float]] = []
    for file_id, r in docs.items():
        scores = processor.score(
            qs=[query_embedding[0]],
            ps=list(r["page_embeddings"]),
        )
        for page_idx, score in enumerate(scores[0]):
            ranked.append((file_id, page_idx, score.item()))
    ranked.sort(key=lambda x: x[2], reverse=True)
    return ranked
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_app.py::TestSearchMulti -v`
Expected: PASS (all 3 tests)

**Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS (existing tests unchanged)

**Step 6: Commit**

```bash
git add streamlit_app.py tests/test_app.py
git commit -m "feat: add search_multi for cross-document search"
```

---

### Task 3: Rewrite UI for multi-PDF support

**Files:**
- Modify: `streamlit_app.py:91-214` (replace entire UI section)

**Step 1: Replace the UI section**

Replace everything from `# UI` (line 91) through the end of the file with:

```python
# UI
st.set_page_config(page_title="Embedding Pipeline", layout="centered")
st.title("Embedding Pipeline")
st.write("Generate vector embeddings from PDF documents with Nomic Embed Multimodal.")

uploaded_files = st.file_uploader(
    "Upload files", type=["pdf"], accept_multiple_files=True
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
        cleanup_stale_results(current_ids, st.session_state.results)
        if not st.session_state.results:
            del st.session_state["results"]
            st.session_state.pop("search_results", None)

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
                model, processor = load_model(device)
                results: dict[str, EmbedResults] = st.session_state.get(
                    "results", {}
                )

                for i, f in enumerate(files_to_embed):
                    file_stem = f.name.rsplit(".", 1)[0]
                    progress.progress(
                        i / len(files_to_embed),
                        text=f"Processing {f.name}...",
                    )
                    try:
                        total_start = time.perf_counter_ns()
                        pages = render_pages(f.read(), dpi=dpi)
                        if not pages:
                            st.error(
                                f"{f.name}: PDF contains no pages to embed."
                            )
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
        all_results: dict[str, EmbedResults] = st.session_state.results

        st.success(f"Embedded {len(all_results)} document(s).")

        # Summary metrics
        total_pages = sum(len(r["pages"]) for r in all_results.values())
        total_duration_ns = sum(
            r["total_duration"] for r in all_results.values()
        )

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
                    cols[i % 4].image(
                        page, caption=f"Page {i + 1}", width="stretch"
                    )
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
            all_json = (
                "[" + ",".join(r["json"] for r in all_results.values()) + "]"
            )
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
        filter_file_ids: list[str | None] = [None] + list(
            all_results.keys()
        )
        filter_idx = st.selectbox(
            "Document filter",
            range(len(filter_options)),
            format_func=lambda i: filter_options[i],
        )
        selected_file_id = filter_file_ids[filter_idx]

        query = st.text_input("Text query")
        if st.button("Search"):
            if not query:
                st.warning("Enter a search query.")
            else:
                try:
                    model, processor = load_model(device)
                    st.session_state.search_results = search_multi(
                        query,
                        model,
                        processor,
                        all_results,
                        filter_file_id=selected_file_id,
                    )
                except (OSError, RuntimeError, ValueError) as e:
                    st.error(str(e))
                except Exception as e:
                    st.exception(e)

        if "search_results" in st.session_state:
            search_results = st.session_state.search_results
            if search_results:
                cols = st.columns(min(len(search_results), 4))
                for rank, (fid, page_idx, score) in enumerate(
                    search_results
                ):
                    r = all_results[fid]
                    cols[rank % 4].image(
                        r["pages"][page_idx],
                        caption=(
                            f"{r['file_stem']} · Page {page_idx + 1}"
                            f" · {score:.4f}"
                        ),
                        width="stretch",
                    )

st.caption(f"Device: {device.upper()}")
```

**Step 2: Run lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: No errors

**Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS

**Step 4: Run typecheck**

Run: `uv run ty check`
Expected: No errors (or only pre-existing warnings)

**Step 5: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: rewrite UI for multi-PDF support"
```

---

### Task 4: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update Session State section**

Change:

```markdown
Embed results stored in a single `st.session_state.results` dict with keys: `file_id`, `pages`, `page_embeddings`, `total_duration`, `file_stem`, `dpi`, `json`. Cleared on new file upload.
```

to:

```markdown
Embed results stored in `st.session_state.results: dict[str, EmbedResults]` keyed by `file_id`. Each entry has keys: `file_id`, `pages`, `page_embeddings`, `total_duration`, `file_stem`, `dpi`, `json`. Stale entries cleaned up when files are removed from the uploader via `cleanup_stale_results`.
```

**Step 2: Update Pipeline section**

Change:

```markdown
PDF upload → render pages as images at configurable DPI (`pymupdf`) → embed images (`BiQwen2_5`) → download JSON / search pages by text query
```

to:

```markdown
Multi-PDF upload → render pages as images at configurable DPI (`pymupdf`) → embed images (`BiQwen2_5`) → download per-document or combined JSON / search pages by text query across all documents or filtered to one
```

**Step 3: Update JSON Download section**

Change:

```markdown
Pre-computed JSON string cached in `results["json"]` at embed time. Fields in the downloadable JSON via `st.download_button`:

- `model` (string) — model that produced the embeddings
- `dpi` (integer) — render resolution in dots per inch (72–300)
- `embeddings` (number[][][]) — per-page multi-vector embeddings (page → patches → 128-dim vectors)
- `total_duration` (integer) — total duration in nanoseconds
- `page_count` (integer) — number of PDF pages processed
```

to:

```markdown
Pre-computed JSON string cached in each `results[file_id]["json"]` at embed time. Per-document download via `st.download_button`, plus "Download All" that concatenates all entries into a JSON array. Fields per document:

- `file_name` (string) — file stem without extension
- `model` (string) — model that produced the embeddings
- `dpi` (integer) — render resolution in dots per inch (72–300)
- `embeddings` (number[][][]) — per-page multi-vector embeddings (page → patches → 128-dim vectors)
- `total_duration` (integer) — total duration in nanoseconds
- `page_count` (integer) — number of PDF pages processed
```

**Step 4: Update Search section**

Change:

```markdown
Text query input scores against page embeddings via `processor.score()` and displays pages ranked by relevance. Search results persist as a separate `st.session_state.search_results` key, cleared on new file upload or new embed.
```

to:

```markdown
Text query scores against page embeddings across all documents via `search_multi`, with an optional document filter (`st.selectbox`). Results display page image, document name, page number, and score. Search results persist as `st.session_state.search_results`, cleared on new embed.
```

**Step 5: Update Tests section**

Change:

```markdown
- `tests/test_app.py` — unit tests for `DPI_OPTIONS`, `get_device`, `render_pages`, `embed`, and `search`
```

to:

```markdown
- `tests/test_app.py` — unit tests for `DPI_OPTIONS`, `get_device`, `render_pages`, `embed`, `search`, `cleanup_stale_results`, and `search_multi`
```

**Step 6: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for multi-PDF support"
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
Verify:
- Multi-file uploader accepts multiple PDFs
- File count and total size shown in caption
- "Embed" processes only new files, "Re-embed All" processes all
- Per-document expanders show page previews, metrics, and individual download
- "Download All" button appears when 2+ documents are embedded
- Search filter dropdown defaults to "All documents"
- Cross-document search returns results attributed to correct documents
- Removing a file from the uploader cleans up its results
