# Embedding Pipeline

Streamlit web app for converting PDF documents to Markdown and generating vector embeddings using IBM's [Granite Embedding](https://huggingface.co/collections/ibm-granite/granite-embedding-models) models.

## Features

- PDF to Markdown conversion via [Docling](https://github.com/docling-ai/docling)
  - Configurable table extraction, code/formula understanding, and picture classification
- Vector embedding generation with [Granite Embedding English R2](https://huggingface.co/ibm-granite/granite-embedding-english-r2) or [Granite Embedding Small English R2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2)
- Automatic device selection (MPS > CUDA > CPU)
- Downloadable JSON output with embeddings, timing, and token count

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## Development

```bash
uv run ruff check .   # lint
uv run ruff format .  # format
uv run ty check       # typecheck
uv run pytest         # test
```
