# CLAUDE.md

## Project Overview

Streamlit web app for converting PDF documents to Markdown and generating vector embeddings using IBM's [Granite Embedding](https://huggingface.co/collections/ibm-granite/granite-embedding-models) models.

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

- `docling` — PDF document processing and conversion
- `transformers` — Hugging Face model loading
- `torch` — tensor operations and normalization
- `streamlit` — web user interface
- `ruff` — linting/formatting (dev)
- `ty` — type checking (dev)
- `pytest` — testing (dev)

## Configuration

`pyproject.toml` — project metadata, dependencies, dev dependency group, ruff lint isort (`combine-as-imports`), ty (`python-version = "3.12"`)

## Architecture

### Entry Point

`streamlit_app.py` — single-file app

### PDF Pipeline Options

Selected via `st.radio`:

- TableFormer mode (Accurate/Fast)

Selected via `st.toggle`:

- Structure prediction for table cells
- Code understanding
- Formula understanding
- Picture classification

### Embedding Models

Selected via `st.radio`:

- [Granite Embedding English R2](https://huggingface.co/ibm-granite/granite-embedding-english-r2)
- [Granite Embedding Small English R2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2)

### Performance

- Best available device: MPS > CUDA > CPU
- `@st.cache_resource` to cache models
- `torch.inference_mode()` for inference
- `BatchEncoding.to(device)` for device transfer
- `time.perf_counter_ns()` for timing (nanoseconds)

### Constants

- `MAX_PDF_PAGES = 100`
- `MAX_FILE_SIZE_BYTES = 20_971_520` (20 MB)
- `NUM_DOCLING_THREADS = 8`

### Error Handling

- `OSError`, `RuntimeError`, `ValueError` caught with `st.error()`
- Empty markdown from PDF conversion raises `ValueError`
- Unexpected exceptions shown with `st.exception()`

### JSON Download

Fields in the downloadable JSON via `st.download_button`:

- `model` (string) — model that produced the embeddings
- `embeddings` (number[][]) — array of vector embeddings
- `total_duration` (integer) — total duration in nanoseconds
- `prompt_eval_count` (integer) — number of input tokens processed

### Metrics

`st.metric` displays all JSON fields except `embeddings`.

## Tests

- `tests/test_app.py` — unit tests for `get_device`, `build_pipeline_options`, `convert`, and `embed`
- `tests/fixtures/test.pdf` — minimal PDF fixture for `convert` tests

## Resources

- [Repository](https://github.com/ibm-granite/granite-embedding-models)
- [Paper](https://arxiv.org/abs/2508.21085)
