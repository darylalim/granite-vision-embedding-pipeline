# Embedding Pipeline

Streamlit web app for generating vector embeddings from multiple PDF documents and searching over them using Nomic's [Embed Multimodal](https://huggingface.co/nomic-ai/nomic-embed-multimodal-3b) model.

## Features

- Multi-PDF upload with batch or incremental embedding
- PDF page rendering at configurable DPI (72, 150, 300) via [PyMuPDF](https://pymupdf.readthedocs.io/)
- Multi-vector embeddings with [Nomic Embed Multimodal 3B](https://huggingface.co/nomic-ai/nomic-embed-multimodal-3b)
- Cross-document text search with optional per-document filtering
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
