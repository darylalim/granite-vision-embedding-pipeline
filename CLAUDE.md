# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Streamlit web app for converting PDF documents to Markdown and generating vector embeddings using IBM's [Granite Embedding](https://huggingface.co/collections/ibm-granite/granite-embedding-models) models.

## Setup

```bash
python3.12 -m venv streamlit_env
source streamlit_env/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Commands

- **Lint**: `ruff check .`
- **Format**: `ruff format .`
- **Typecheck**: `pyright`
- **Test**: `pytest`

## Code Style

- snake_case for functions/variables, PascalCase for classes
- Type annotations on all parameters and returns
- isort with combine-as-imports (configured in `pyproject.toml`)

## Dependencies

- `docling` - PDF document processing and conversion
- `transformers` - Hugging Face model loading
- `torch` - Tensor operations and normalization
- `streamlit` - Web user interface
- `ruff` — linting/formatting (dev)
- `pyright` — type checking (dev)
- `pytest` — testing (dev)

## Configuration

`pyproject.toml` — ruff lint isort (`combine-as-imports`) and pyright (`pythonVersion = "3.12"`).

## Architecture

### Entry Point

`streamlit_app.py` - single-file app.

### PDF Pipeline Options

Option selected via `st.radio`:

- TableFormer mode (Accurate/Fast)

Options selected via `st.toggle`:

- Structure prediction for table cells
- Code understanding
- Formula understanding
- Picture classification

### Embedding Models

Granite Embedding models selected via `st.radio`:

- [Granite Embedding English R2](https://huggingface.co/ibm-granite/granite-embedding-english-r2)
- [Granite Embedding Small English R2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2)

### Performance

- Use best available device: MPS > CUDA > CPU
- `@st.cache_resource` to cache models
- `docling` for PDF conversion
- `time.perf_counter_ns()` for timing (nanoseconds)

### Constants

- `MAX_PDF_PAGES = 100`
- `MAX_FILE_SIZE_BYTES = 20_971_520` (20MB)
- `NUM_DOCLING_THREADS = 8`

### Error Handling

- `OSError`, `RuntimeError`, `ValueError` caught with `st.error()`
- Empty markdown from PDF conversion raises `ValueError`
- Unexpected exceptions shown with `st.exception()` for debugging

### JSON Download

Fields in the downloadable JSON via `st.download_button`:

- `model` (string) - Model that produced the embeddings
- `embeddings` (number[][]) - Array of vector embeddings
- `total_duration` (integer) - Total time spent generating in nanoseconds
- `prompt_eval_count` (integer) - Number of input tokens processed to generated embeddings

### Metrics

`st.metric` displays all JSON fields except embeddings.

## Tests

`tests/test_app.py` — unit tests for `get_device`, `build_pipeline_options`, and `embed`.

## Resources

- [Repository](https://github.com/ibm-granite/granite-embedding-models)
- [Paper](https://arxiv.org/abs/2508.21085)
