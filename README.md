# Granite Vision Embedding Pipeline

Streamlit web app for generating vector embeddings from PDF documents and images and searching over them using IBM Granite's [Vision Embedding](https://huggingface.co/ibm-granite/granite-vision-3.3-2b-embedding) model.

## Features

- Multi-PDF and image upload (PNG, JPG, JPEG, WebP) with batch or incremental embedding
- PDF page rendering at configurable DPI (72, 150, 300) via [PyMuPDF](https://pymupdf.readthedocs.io/)
- Multi-vector embeddings with [Granite Vision 3.3 2B Embedding](https://huggingface.co/ibm-granite/granite-vision-3.3-2b-embedding)
- Cross-document text search with top-K and score threshold filtering
- Optional per-document search filtering
- Automatic device selection (MPS > CUDA > CPU)
- Per-document and combined JSON downloads with embeddings, DPI, and timing

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
