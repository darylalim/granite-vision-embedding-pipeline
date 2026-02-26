# Multi-PDF Support Design

## Goal

Extend the app from single-PDF to multi-PDF support, enabling users to upload multiple documents, embed them individually or as a batch, and search across all documents or within a specific one.

## Approach

Multi-file uploader with shared session state (Approach A). Extends the existing pattern with minimal architectural change using Streamlit's native `accept_multiple_files` support.

## File Upload

Replace the single-file uploader with `st.file_uploader("Upload files", type=["pdf"], accept_multiple_files=True)`. Display a summary line showing count and total size of uploaded files. Clean up stale results when files are removed.

## Session State

Replace `st.session_state.results: EmbedResults` with `st.session_state.results: dict[str, EmbedResults]` keyed by `file_id`. Each entry uses the existing `EmbedResults` TypedDict. `search_results` becomes a list of `(file_stem, page_idx, score)` tuples to attribute results across documents.

On each rerun, compare current `file_id`s from the uploader against keys in `results` and remove stale entries.

## Embedding Flow

Two buttons below the file list:

- **Embed** — embeds only files without existing results (skips already-embedded). Disabled if all files are embedded.
- **Re-embed All** — clears all results and re-embeds everything (useful after DPI change).

Single progress bar tracks overall progress. Model loaded once at the start. DPI selection applies globally.

## Results Display

- Per-document expanders with page previews and metrics (duration, page count, DPI).
- Summary metrics row at top: total pages, total duration, document count.
- "Download All" button exporting a single JSON array of document entries (each with model, dpi, embeddings, total_duration, page_count, file_name). Individual per-document download buttons inside each expander.

## Search

- Default: cross-document search scoring against all page embeddings.
- Filter dropdown (`st.selectbox`): "All documents" plus each embedded document's file stem. Scopes search when a specific document is selected.
- Results show page image, document name, page number, and score in captions: `"report.pdf · Page 3 · 0.8921"`.
- Search results cleared on new embed.

## Error Handling

Same pattern as today. Applied per-file during batch embedding so one failure doesn't stop others. Failed files show an error badge.

## Testing

- Update existing tests for multi-result state shape.
- Add tests for cross-document search scoring.
- Add tests for stale file cleanup logic.
- Existing `render_pages`, `embed`, `get_device` tests unchanged.
