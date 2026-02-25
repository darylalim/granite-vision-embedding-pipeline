# Inline Search Design

## Goal

Add text query search over embedded PDF pages. After embedding a PDF, users type a query and see pages ranked by relevance with thumbnails and scores.

## Approach

Inline search — a text input and search button appear below embed results. No tabs, no session state, no persistence. Everything stays in `streamlit_app.py`.

## Changes

### `embed()` — return tensor

Change return type from `list[list[list[float]]]` to `torch.Tensor`. Remove `.tolist()` call. Convert to list only at JSON serialization time.

### New function: `search()`

```python
def search(
    query: str,
    model: BiQwen2_5,
    processor: BiQwen2_5_Processor,
    image_embeddings: torch.Tensor,
) -> list[tuple[int, float]]:
```

- Process query: `processor.process_texts([query])`
- Embed query: `model(**batch)` under `torch.inference_mode()`
- Score: `processor.score(qs=..., ps=...)`
- Return `(page_index, score)` pairs sorted by descending score

### UI additions

After download button, inside the embed block:

- `st.subheader("Search")`
- `st.text_input` for query
- `st.button("Search")` triggers `search()` call
- Results: 4-column grid of page thumbnails ranked by score, with score captions

### JSON download

Call `.tolist()` on tensor at serialization time.

### Tests

Update `test_embed` to assert `torch.Tensor` return type.

## UI Flow

```
Upload PDF → Embed → metrics + download
                   → Search input + button
                   → Ranked page thumbnails with scores
```
