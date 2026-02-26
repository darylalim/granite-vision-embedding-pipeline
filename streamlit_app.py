import json
import time

import fitz
import streamlit as st
import torch
from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor
from PIL import Image

MODEL_ID = "nomic-ai/nomic-embed-multimodal-3b"


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


def render_pages(data: bytes, dpi: int = 150) -> list[Image.Image]:
    """Render PDF pages as PIL Images."""
    try:
        scale = dpi / 72
        matrix = fitz.Matrix(scale, scale)
        with fitz.open(stream=data, filetype="pdf") as doc:
            return [
                Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                for page in doc
                for pix in [page.get_pixmap(matrix=matrix)]
            ]
    except (fitz.FileDataError, fitz.EmptyFileError):
        return []


def embed(
    images: list[Image.Image], model: BiQwen2_5, processor: BiQwen2_5_Processor
) -> torch.Tensor:
    """Generate per-page multi-vector embeddings from images."""
    batch = processor.process_images(images).to(model.device)
    with torch.inference_mode():
        embeddings = model(**batch)
    return embeddings


def search(
    query: str,
    model: BiQwen2_5,
    processor: BiQwen2_5_Processor,
    image_embeddings: torch.Tensor,
) -> list[tuple[int, float]]:
    """Score a text query against image embeddings and return ranked results."""
    batch = processor.process_texts([query]).to(model.device)
    with torch.inference_mode():
        query_embedding = model(**batch)
    scores = processor.score(
        qs=[query_embedding[0]],
        ps=list(image_embeddings),
    )
    ranked = sorted(
        [(i, score.item()) for i, score in enumerate(scores[0])],
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked


# UI
st.set_page_config(page_title="Embedding Pipeline", layout="centered")
st.title("Embedding Pipeline")
st.write("Generate vector embeddings from PDF documents with Nomic Embed Multimodal.")

uploaded_file = st.file_uploader("Upload file", type=["pdf"])

device = get_device()

if uploaded_file:
    size_mb = len(uploaded_file.getvalue()) / 1_048_576
    st.caption(f"{uploaded_file.name} · {size_mb:.1f} MB")

    dpi = st.slider(
        "Render DPI",
        min_value=72,
        max_value=300,
        value=150,
        step=1,
        help="72 = Low · 150 = Medium · 300 = High",
    )

    # Clear stale results if a different file was uploaded
    if st.session_state.get("file_id") != uploaded_file.file_id:
        for key in (
            "pages",
            "page_embeddings",
            "total_duration",
            "file_stem",
            "search_results",
            "file_id",
            "dpi",
        ):
            st.session_state.pop(key, None)

    if st.button("Embed", type="primary"):
        try:
            total_start = time.perf_counter_ns()

            progress = st.progress(0.0, text="Loading model...")

            # Load model
            model, processor = load_model(device)

            # Render PDF pages as images
            progress.progress(0.33, text="Rendering pages...")
            pages = render_pages(uploaded_file.read(), dpi=dpi)

            if not pages:
                raise ValueError("PDF contains no pages to embed.")

            # Generate embeddings
            progress.progress(0.66, text="Generating embeddings...")
            page_embeddings = embed(pages, model, processor)

            total_duration = time.perf_counter_ns() - total_start

            progress.progress(1.0, text="Complete.")
            progress.empty()

            # Store results in session state
            st.session_state.file_id = uploaded_file.file_id
            st.session_state.pages = pages
            st.session_state.page_embeddings = page_embeddings
            st.session_state.total_duration = total_duration
            st.session_state.file_stem = uploaded_file.name.rsplit(".", 1)[0]
            st.session_state.dpi = dpi
            st.session_state.pop("search_results", None)

        except (OSError, RuntimeError, ValueError) as e:
            st.error(str(e))
        except Exception as e:
            st.exception(e)

    if "pages" in st.session_state:
        pages = st.session_state.pages
        page_embeddings = st.session_state.page_embeddings
        total_duration = st.session_state.total_duration
        file_stem = st.session_state.file_stem
        dpi = st.session_state.dpi

        st.success("Done.")

        with st.expander(f"Page previews ({len(pages)})"):
            cols = st.columns(min(len(pages), 4))
            for i, page in enumerate(pages):
                cols[i % 4].image(page, caption=f"Page {i + 1}", width="stretch")

        st.subheader("Metrics")
        st.metric("Model", MODEL_ID)
        col1, col2, col3 = st.columns(3)
        duration_s = total_duration / 1_000_000_000
        col1.metric("Duration", f"{duration_s:.2f} s")
        col2.metric("Page Count", len(pages))
        col3.metric("DPI", dpi)

        embedding_data = {
            "model": MODEL_ID,
            "dpi": dpi,
            "embeddings": page_embeddings.tolist(),
            "total_duration": total_duration,
            "page_count": len(pages),
        }
        st.download_button(
            label="Download JSON",
            data=json.dumps(embedding_data),
            file_name=f"{file_stem}_embedding.json",
            mime="application/json",
        )

        st.subheader("Search")
        query = st.text_input("Text query")
        if st.button("Search"):
            if not query:
                st.warning("Enter a search query.")
            else:
                try:
                    model, processor = load_model(device)
                    st.session_state.search_results = search(
                        query, model, processor, page_embeddings
                    )
                except (OSError, RuntimeError, ValueError) as e:
                    st.error(str(e))
                except Exception as e:
                    st.exception(e)

        if "search_results" in st.session_state:
            results = st.session_state.search_results
            cols = st.columns(min(len(results), 4))
            for rank, (page_idx, score) in enumerate(results):
                cols[rank % 4].image(
                    pages[page_idx],
                    caption=f"Page {page_idx + 1} · {score:.4f}",
                    width="stretch",
                )

st.caption(f"Device: {device.upper()}")
