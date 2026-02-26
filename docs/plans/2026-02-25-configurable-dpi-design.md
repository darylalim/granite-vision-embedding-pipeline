# Configurable Render Resolution

## Goal

Let users control PDF render DPI (72–300) to improve embedding quality for visually rich content or save memory on text-heavy documents.

## Changes

### `render_pages` function

Add a `dpi` parameter (default 150). Compute scale as `dpi / 72` and pass `fitz.Matrix(scale, scale)` to `get_pixmap()`.

```python
def render_pages(data: bytes, dpi: int = 150) -> list[Image.Image]:
```

### UI

Inline `st.slider` between the file info caption and the Embed button:

```python
dpi = st.slider("Render DPI", min_value=72, max_value=300, value=150, step=1,
                help="72 = Low · 150 = Medium · 300 = High")
```

Pass `dpi` to `render_pages` and store in session state.

### JSON output

Add a `"dpi"` field to the downloadable JSON:

```python
embedding_data = {
    "model": MODEL_ID,
    "dpi": dpi,
    "embeddings": ...,
    "total_duration": ...,
    "page_count": ...,
}
```

### Metrics

Add a DPI metric alongside duration and page count.

### Session state

Add `"dpi"` to the list of keys cleared on new file upload. Store after embedding.

### Tests

- Test `render_pages` with explicit DPI values and verify image dimensions scale with DPI.
- Update existing `render_pages` tests if needed.
