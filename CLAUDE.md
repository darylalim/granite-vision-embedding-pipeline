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

- `colpali-engine` — multimodal embedding model (`BiQwen2_5`, `BiQwen2_5_Processor`)
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

[Nomic Embed Multimodal 3B](https://huggingface.co/nomic-ai/nomic-embed-multimodal-3b) — multi-vector vision-language embedding model

### Pipeline

PDF upload → render pages as images at configurable DPI (`pymupdf`) → embed images (`BiQwen2_5`) → download JSON / search pages by text query

### Performance

- Best available device: MPS > CUDA > CPU
- `@st.cache_resource` to cache model and processor
- `torch.inference_mode()` for inference
- `torch.bfloat16` for model precision
- `time.perf_counter_ns()` for timing (nanoseconds)

### Constants

- `MODEL_ID = "nomic-ai/nomic-embed-multimodal-3b"`

### Error Handling

- `OSError`, `RuntimeError`, `ValueError` caught with `st.error()`
- Empty or malformed PDFs raise `ValueError`
- Unexpected exceptions shown with `st.exception()`

### JSON Download

Fields in the downloadable JSON via `st.download_button`:

- `model` (string) — model that produced the embeddings
- `dpi` (integer) — render resolution in dots per inch (72–300)
- `embeddings` (number[][][]) — per-page multi-vector embeddings (page → patches → 128-dim vectors)
- `total_duration` (integer) — total duration in nanoseconds
- `page_count` (integer) — number of PDF pages processed

### Search

Text query input scores against page embeddings via `processor.score()` and displays pages ranked by relevance. Results persist across Streamlit reruns via `st.session_state`.

### Metrics

`st.metric` displays model, duration (seconds), and page count.

## Tests

- `tests/test_app.py` — unit tests for `get_device`, `render_pages`, `embed`, and `search`
- `tests/fixtures/test.pdf` — minimal PDF fixture for `render_pages` tests

## Resources

- [Model Card](https://huggingface.co/nomic-ai/nomic-embed-multimodal-3b)
- [ColPali Engine](https://github.com/illuin-tech/colpali)
