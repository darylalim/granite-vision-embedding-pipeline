# Embedding Pipeline

Streamlit web app for generating vector embeddings from PDF documents and searching over them using Nomic's [Embed Multimodal](https://huggingface.co/nomic-ai/nomic-embed-multimodal-3b) model.

## Features

- PDF page rendering at configurable DPI (72, 150, 300) via [PyMuPDF](https://pymupdf.readthedocs.io/)
- Multi-vector embeddings with [Nomic Embed Multimodal 3B](https://huggingface.co/nomic-ai/nomic-embed-multimodal-3b)
- Text query search over embedded pages ranked by relevance
- Automatic device selection (MPS > CUDA > CPU)
- Downloadable JSON output with per-page embeddings, DPI, and timing

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
