import json
import tempfile
import time
from pathlib import Path

import fitz
import streamlit as st
import torch
from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor
from PIL import Image

MODEL_ID = "nomic-ai/nomic-embed-multimodal-3b"
MAX_PDF_PAGES = 100
MAX_FILE_SIZE_BYTES = 20_971_520  # 20MB


def get_device() -> str:
    """Detect best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@st.cache_resource
def load_model(device: str) -> tuple[BiQwen2_5, BiQwen2_5_Processor]:
    """Load embedding model and processor."""
    model = BiQwen2_5.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=device
    ).eval()
    processor = BiQwen2_5_Processor.from_pretrained(MODEL_ID)
    return model, processor


def render_pages(source: str) -> list[Image.Image]:
    """Render PDF pages as PIL Images."""
    doc = fitz.open(source)
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


# UI
st.title("Embedding Pipeline")
st.write("Generate vector embeddings from PDF documents with Nomic Embed Multimodal.")

uploaded_file = st.file_uploader("Upload file", type=["pdf"])

device = get_device()

if st.button("Embed", type="primary"):
    if uploaded_file is None:
        st.warning("Upload a PDF file.")
    else:
        tmp_file_path = None
        try:
            total_start = time.perf_counter_ns()

            # Load model
            with st.spinner(f"Loading model on {device.upper()}..."):
                model, processor = load_model(device)

            # Render PDF pages as images
            with st.spinner("Rendering pages..."):
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp_file.write(uploaded_file.read())
                tmp_file.close()
                tmp_file_path = tmp_file.name
                pages = render_pages(tmp_file_path)

            if not pages:
                raise ValueError("PDF contains no pages to embed.")

            # Generate embeddings
            with st.spinner("Generating embeddings..."):
                page_embeddings = embed(pages, model, processor)

            total_duration = time.perf_counter_ns() - total_start

            # Display results
            st.success("Done.")

            st.subheader("Metrics")
            st.metric("Model", MODEL_ID)
            col1, col2 = st.columns(2)
            col1.metric("Total Duration (ns)", f"{total_duration:,}")
            col2.metric("Page Count", len(pages))

            embedding_data = {
                "model": MODEL_ID,
                "embeddings": page_embeddings,
                "total_duration": total_duration,
                "page_count": len(pages),
            }
            st.download_button(
                label="Download JSON",
                data=json.dumps(embedding_data),
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
