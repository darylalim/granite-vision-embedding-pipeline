# CLAUDE.md

## Project Overview

Streamlit web app for generating vector embeddings from PDF documents and searching over them using Nomic's [Embed Multimodal](https://huggingface.co/nomic-ai/nomic-embed-multimodal-3b) model.

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## Commands

- **Lint**: `uv run ruff check .`
- **Format**: `uv run ruff format .`
- **Typecheck**: `uv run ty check`
- **Test**: `uv run pytest`

## Code Style

- snake_case for functions/variables, PascalCase for classes
- Type annotations on all parameters and returns
- isort with combine-as-imports (configured in `pyproject.toml`)

## Dependencies

- `colpali-engine` — multimodal embedding model (`ColQwen2_5`, `ColQwen2_5_Processor`)
- `pymupdf` — PDF page rendering
- `torch` — tensor operations
- `streamlit` — web user interface
- `ruff` — linting/formatting (dev)
- `ty` — type checking (dev)
- `pytest` — testing (dev)

## Configuration

`pyproject.toml` — project metadata, dependencies, dev dependency group, ruff lint isort (`combine-as-imports`), ty (`python-version = "3.12"`)

## Architecture

### Entry Point

`streamlit_app.py` — single-file app

### Embedding Model

[ColNomic Embed Multimodal 3B](https://huggingface.co/nomic-ai/colnomic-embed-multimodal-3b) — multi-vector vision-language embedding model

### Pipeline

Multi-PDF upload → render pages as images at configurable DPI (`pymupdf`) → embed images (`ColQwen2_5`) → download per-document or combined JSON / search pages by text query across all documents or filtered to one

### Performance

- Best available device: MPS > CUDA > CPU
- `@st.cache_resource` to cache model and processor
- `torch.inference_mode()` for inference
- `torch.bfloat16` for model precision
- `time.perf_counter_ns()` for timing (nanoseconds)
- JSON string pre-computed at embed time to avoid repeated `tolist()` on reruns

### Constants

- `MODEL_ID = "nomic-ai/colnomic-embed-multimodal-3b"`
- `DPI_OPTIONS = {"Low (72)": 72, "Medium (150)": 150, "High (300)": 300}`

### Error Handling

- `OSError`, `RuntimeError`, `ValueError` caught with `st.error()`
- Empty or malformed PDFs raise `ValueError`
- Unexpected exceptions shown with `st.exception()`

### Session State

Embed results stored in `st.session_state.results: dict[str, EmbedResults]` keyed by `file_id`. Each entry has keys: `file_id`, `pages`, `page_embeddings`, `total_duration`, `file_stem`, `dpi`, `json`. Stale entries cleaned up when files are removed from the uploader via `cleanup_stale_results`.

### JSON Download

Pre-computed JSON string cached in each `results[file_id]["json"]` at embed time. Per-document download via `st.download_button`, plus "Download All" that concatenates all entries into a JSON array. Fields per document:

- `file_name` (string) — file stem without extension
- `model` (string) — model that produced the embeddings
- `dpi` (integer) — render resolution in dots per inch (72–300)
- `embeddings` (number[][][]) — per-page multi-vector embeddings (page → patches → 128-dim vectors)
- `total_duration` (integer) — total duration in nanoseconds
- `page_count` (integer) — number of PDF pages processed

### Search

Text query scores against page embeddings across all documents via `search_multi`, with an optional document filter (`st.selectbox`). Results display page image, document name, page number, and score. Search results persist as `st.session_state.search_results`, cleared on new embed.

### Metrics

`st.metric` displays model, duration (seconds), page count, and document count. Per-document expanders show duration, page count, and DPI.

## Tests

- `tests/test_app.py` — unit tests for `DPI_OPTIONS`, `get_device`, `render_pages`, `embed`, `search`, `cleanup_stale_results`, and `search_multi`
- `tests/data/pdf/single_page.pdf` — single-page PDF fixture for `render_pages` tests
- `tests/data/pdf/multi_page.pdf` — multi-page PDF fixture for `render_pages` tests

## Resources

- [Model Card](https://huggingface.co/nomic-ai/colnomic-embed-multimodal-3b)
- [ColPali Engine](https://github.com/illuin-tech/colpali)
