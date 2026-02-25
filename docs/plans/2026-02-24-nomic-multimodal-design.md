# Refactor to Nomic Embed Multimodal 3B

## Overview

Replace IBM Granite Embedding text-only models with nomic-embed-multimodal-3b. The pipeline changes from PDF-to-Markdown-to-text-embedding to PDF-to-page-images-to-multimodal-embedding.

## What's removed

- Docling dependency and all related code (`convert()`, `build_pipeline_options()`, pipeline options UI)
- `transformers` AutoModel/AutoTokenizer usage
- Granite model selection UI
- Table extraction and enrichment settings

## What's added

- `colpali-engine` dependency (provides `BiQwen2_5`, `BiQwen2_5_Processor`)
- `Pillow` for image handling
- `pymupdf` for PDF page rendering
- `embed()` rewritten to process images through the multimodal model

## Core functions

| Function | Purpose |
|----------|---------|
| `get_device()` | Unchanged — MPS > CUDA > CPU |
| `load_model()` | Load `BiQwen2_5` + `BiQwen2_5_Processor`, cached |
| `render_pages()` | New — renders PDF pages to PIL Images via pymupdf |
| `embed()` | Rewritten — takes list of PIL Images, returns list of embedding matrices |

## UI

- File uploader (PDF only, same limits)
- No model selector (single model)
- No pipeline/enrichment options (no Docling)
- "Embed" button -> spinner -> metrics + JSON download

## JSON output

```json
{
  "model": "nomic-ai/nomic-embed-multimodal-3b",
  "embeddings": [[[...], [...]], [[...], [...]]],
  "total_duration": 123456789,
  "page_count": 2
}
```

`embeddings` is `number[][][]` — page -> patch vectors. `prompt_eval_count` replaced with `page_count`.

## Tests

- `TestGetDevice` — unchanged
- `TestRenderPages` — test with `test.pdf` fixture
- `TestEmbed` — rewritten with mocked `BiQwen2_5` / `BiQwen2_5_Processor`
- `TestBuildPipelineOptions` and `TestConvert` — removed
