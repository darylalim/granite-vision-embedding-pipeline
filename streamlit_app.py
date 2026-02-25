from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import fitz
import streamlit as st
import torch
try:
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption
except ModuleNotFoundError:
    pass
from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor
from PIL import Image
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

ARTIFACTS_PATH = str(Path.home() / ".cache" / "docling" / "models")

EMBEDDING_MODELS: dict[str, str] = {
    "Granite Embedding English R2": "ibm-granite/granite-embedding-english-r2",
    "Granite Embedding Small English R2": "ibm-granite/granite-embedding-small-english-r2",
}
MAX_PDF_PAGES = 100
MAX_FILE_SIZE_BYTES = 20_971_520  # 20MB
NUM_DOCLING_THREADS = 8


def get_device() -> tuple[str, AcceleratorDevice]:
    """Detect best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return "mps", AcceleratorDevice.MPS
    if torch.cuda.is_available():
        return "cuda", AcceleratorDevice.CUDA
    return "cpu", AcceleratorDevice.CPU


@st.cache_resource
def load_model(
    model_id: str, device: str
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load embedding model and tokenizer."""
    return (
        AutoModel.from_pretrained(model_id, device_map=device),
        AutoTokenizer.from_pretrained(model_id),
    )


def convert(source: str, doc_converter: DocumentConverter) -> str:
    """Convert PDF to markdown."""
    result = doc_converter.convert(
        source=source, max_num_pages=MAX_PDF_PAGES, max_file_size=MAX_FILE_SIZE_BYTES
    )
    return result.document.export_to_markdown()


def render_pages(source: str) -> list[Image.Image]:
    """Render PDF pages as PIL Images."""
    try:
        doc = fitz.open(source)
    except (fitz.FileDataError, fitz.EmptyFileError):
        return []
    pages = []
    for page in doc:
        pix = page.get_pixmap()
        pages.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    doc.close()
    return pages


def embed(
    images: list[Image.Image], model: BiQwen2_5, processor: BiQwen2_5_Processor
) -> list[list[list[float]]]:
    """Generate per-page multi-vector embeddings from images."""
    batch = processor.process_images(images).to(model.device)
    with torch.inference_mode():
        embeddings = model(**batch)
    return embeddings.tolist()


def build_pipeline_options(
    tableformer_mode: str,
    use_structure_prediction: bool,
    code_understanding: bool,
    formula_understanding: bool,
    picture_classification: bool,
    accelerator_device: AcceleratorDevice,
) -> PdfPipelineOptions:
    """Build PDF pipeline options from UI settings."""
    options = PdfPipelineOptions(artifacts_path=ARTIFACTS_PATH, do_table_structure=True)

    options.table_structure_options.mode = (
        TableFormerMode.ACCURATE
        if tableformer_mode == "Accurate"
        else TableFormerMode.FAST
    )
    options.table_structure_options.do_cell_matching = not use_structure_prediction
    options.do_code_enrichment = code_understanding
    options.do_formula_enrichment = formula_understanding

    if picture_classification:
        options.generate_picture_images = True
        options.images_scale = 2
        options.do_picture_classification = True

    options.accelerator_options = AcceleratorOptions(
        num_threads=NUM_DOCLING_THREADS, device=accelerator_device
    )
    return options


# UI
if st.runtime.exists():
    st.title("Embedding Pipeline")
    st.write(
        "Generate vector embeddings from PDF documents with IBM Granite Embedding models."
    )

    uploaded_file = st.file_uploader("Upload file", type=["pdf"])

    device, accelerator_device = get_device()

    st.subheader("Embedding Model")
    model_name = st.radio(
        "Model",
        options=list(EMBEDDING_MODELS.keys()),
        index=0,
    )
    assert model_name is not None
    model_id = EMBEDDING_MODELS[model_name]

    st.subheader("PDF Table Extraction")
    use_structure_prediction = st.toggle(
        "Use text cells from structure prediction",
        value=False,
        help="Uses text cells predicted from the table structure model instead of mapping back to PDF cells.",
    )

    tableformer_mode = st.radio(
        "TableFormer Mode",
        options=["Accurate", "Fast"],
        index=0,
        help="Accurate mode provides better quality. Fast mode is faster but less accurate.",
    )
    assert tableformer_mode is not None

    st.subheader("Enrichment")
    code_understanding = st.toggle(
        "Code understanding", value=False, help="Advanced parsing for code blocks."
    )
    formula_understanding = st.toggle(
        "Formula understanding", value=False, help="Extracts LaTeX from equations."
    )
    picture_classification = st.toggle(
        "Picture classification", value=False, help="Classifies pictures in the document."
    )

    if st.button("Embed", type="primary"):
        if uploaded_file is None:
            st.warning("Upload a PDF file.")
        else:
            tmp_file_path = None
            try:
                total_start = time.perf_counter_ns()

                # Load model
                with st.spinner(f"Loading model on {device.upper()}..."):
                    model, tokenizer = load_model(model_id, device)

                # Build converter with pipeline options
                pipeline_options = build_pipeline_options(
                    tableformer_mode,
                    use_structure_prediction,
                    code_understanding,
                    formula_understanding,
                    picture_classification,
                    accelerator_device,
                )
                doc_converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_options=pipeline_options
                        )
                    }
                )

                # Convert PDF to markdown
                with st.spinner("Converting document..."):
                    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    tmp_file.write(uploaded_file.read())
                    tmp_file.close()
                    tmp_file_path = tmp_file.name
                    doc_markdown = convert(tmp_file_path, doc_converter)

                if not doc_markdown.strip():
                    raise ValueError("Document produced no text content to embed.")

                # Generate embedding
                with st.spinner("Generating embedding..."):
                    embedding_vector, prompt_eval_count = embed(
                        doc_markdown, model, tokenizer, device
                    )

                total_duration = time.perf_counter_ns() - total_start

                # Display results
                st.success("Done.")

                st.subheader("Metrics")
                st.metric("Model", model_id)
                col1, col2 = st.columns(2)
                col1.metric("Total Duration (ns)", f"{total_duration:,}")
                col2.metric("Prompt Eval Count", prompt_eval_count)

                embedding_data = {
                    "model": model_id,
                    "embeddings": [embedding_vector],
                    "total_duration": total_duration,
                    "prompt_eval_count": prompt_eval_count,
                }
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(embedding_data, indent=2),
                    file_name="embedding.json",
                    mime="application/json",
                )

            except OSError as e:
                st.error(f"File error: {e}")
            except RuntimeError as e:
                st.error(f"Model error: {e}")
            except ValueError as e:
                st.error(f"Processing error: {e}")
            except Exception as e:
                st.exception(e)

            finally:
                if tmp_file_path:
                    Path(tmp_file_path).unlink(missing_ok=True)
