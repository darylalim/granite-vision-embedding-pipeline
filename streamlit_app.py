import json
import time
from typing import TypedDict

import fitz
import streamlit as st
import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from PIL import Image

MODEL_ID = "nomic-ai/colnomic-embed-multimodal-3b"
DPI_OPTIONS = {"Low (72)": 72, "Medium (150)": 150, "High (300)": 300}


class EmbedResults(TypedDict):
    file_id: str
    pages: list[Image.Image]
    page_embeddings: torch.Tensor
    total_duration: int
    file_stem: str
    dpi: int
    json: str


def get_device() -> str:
    """Detect best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@st.cache_resource
def load_model(device: str) -> tuple[ColQwen2_5, ColQwen2_5_Processor]:
    """Load embedding model and processor."""
    model = ColQwen2_5.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=device
    ).eval()
    processor = ColQwen2_5_Processor.from_pretrained(MODEL_ID)
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
    images: list[Image.Image], model: ColQwen2_5, processor: ColQwen2_5_Processor
) -> torch.Tensor:
    """Generate per-page multi-vector embeddings from images."""
    batch = processor.process_images(images).to(model.device)
    with torch.inference_mode():
        embeddings = model(**batch)
    return embeddings


def search(
    query: str,
    model: ColQwen2_5,
    processor: ColQwen2_5_Processor,
    image_embeddings: torch.Tensor,
) -> list[tuple[int, float]]:
    """Score a text query against image embeddings and return ranked results."""
    batch = processor.process_queries([query]).to(model.device)
    with torch.inference_mode():
        query_embedding = model(**batch)
    scores = processor.score_multi_vector(
        qs=[query_embedding[0]],
        ps=list(image_embeddings),
    )
    ranked = sorted(
        [(i, score.item()) for i, score in enumerate(scores[0])],
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked


def cleanup_stale_results(
    current_file_ids: set[str], results: dict[str, EmbedResults]
) -> None:
    """Remove results for files no longer in the uploader."""
    for file_id in set(results.keys()) - current_file_ids:
        del results[file_id]


def filter_results(
    results: list[tuple[str, int, float]],
    top_k: int = 5,
    min_score: float = 0.0,
) -> list[tuple[str, int, float]]:
    """Apply score threshold then top-K to ranked search results."""
    filtered = [r for r in results if r[2] >= min_score]
    return filtered[:top_k]


def search_multi(
    query: str,
    model: ColQwen2_5,
    processor: ColQwen2_5_Processor,
    results: dict[str, EmbedResults],
    filter_file_id: str | None = None,
) -> list[tuple[str, int, float]]:
    """Score a text query across multiple documents and return ranked results."""
    docs = (
        {filter_file_id: results[filter_file_id]}
        if filter_file_id is not None
        else results
    )
    batch = processor.process_queries([query]).to(model.device)
    with torch.inference_mode():
        query_embedding = model(**batch)
    ranked: list[tuple[str, int, float]] = []
    for file_id, r in docs.items():
        scores = processor.score_multi_vector(
            qs=[query_embedding[0]],
            ps=list(r["page_embeddings"]),
        )
        for page_idx, score in enumerate(scores[0]):
            ranked.append((file_id, page_idx, score.item()))
    ranked.sort(key=lambda x: x[2], reverse=True)
    return ranked


# UI
st.set_page_config(page_title="Embedding Pipeline", layout="centered")
st.title("Embedding Pipeline")
st.write(
    "Generate vector embeddings from PDF documents with ColNomic Embed Multimodal."
)

uploaded_files = st.file_uploader(
    "Upload files", type=["pdf"], accept_multiple_files=True
)

device = get_device()

if uploaded_files:
    total_size_mb = sum(len(f.getvalue()) for f in uploaded_files) / 1_048_576
    st.caption(f"{len(uploaded_files)} file(s) · {total_size_mb:.1f} MB")

    dpi_label = st.radio("Render DPI", DPI_OPTIONS, index=1, horizontal=True)
    dpi = DPI_OPTIONS[dpi_label]

    # Clean up stale results for removed files
    if "results" in st.session_state:
        current_ids = {f.file_id for f in uploaded_files}
        prev_count = len(st.session_state.results)
        cleanup_stale_results(current_ids, st.session_state.results)
        if len(st.session_state.results) < prev_count:
            st.session_state.pop("search_results", None)
        if not st.session_state.results:
            del st.session_state["results"]

    existing_results = st.session_state.get("results", {})
    files_to_embed = [f for f in uploaded_files if f.file_id not in existing_results]

    col_embed, col_reembed = st.columns(2)
    embed_clicked = col_embed.button(
        "Embed", type="primary", disabled=not files_to_embed
    )
    reembed_clicked = col_reembed.button("Re-embed All")

    if embed_clicked or reembed_clicked:
        if reembed_clicked:
            st.session_state.pop("results", None)
            files_to_embed = list(uploaded_files)

        if files_to_embed:
            try:
                progress = st.progress(0.0, text="Loading model...")
                model, processor = load_model(device)
                results: dict[str, EmbedResults] = st.session_state.get("results", {})

                for i, f in enumerate(files_to_embed):
                    file_stem = f.name.rsplit(".", 1)[0]
                    progress.progress(
                        i / len(files_to_embed),
                        text=f"Processing {f.name}...",
                    )
                    try:
                        total_start = time.perf_counter_ns()
                        pages = render_pages(f.read(), dpi=dpi)
                        if not pages:
                            st.error(f"{f.name}: PDF contains no pages to embed.")
                            continue
                        page_embeddings = embed(pages, model, processor)
                        total_duration = time.perf_counter_ns() - total_start

                        results[f.file_id] = {
                            "file_id": f.file_id,
                            "pages": pages,
                            "page_embeddings": page_embeddings,
                            "total_duration": total_duration,
                            "file_stem": file_stem,
                            "dpi": dpi,
                            "json": json.dumps(
                                {
                                    "file_name": file_stem,
                                    "model": MODEL_ID,
                                    "dpi": dpi,
                                    "embeddings": page_embeddings.tolist(),
                                    "total_duration": total_duration,
                                    "page_count": len(pages),
                                }
                            ),
                        }
                    except (OSError, RuntimeError, ValueError) as e:
                        st.error(f"{f.name}: {e}")
                    except Exception as e:
                        st.exception(e)

                progress.progress(1.0, text="Complete.")
                progress.empty()
                st.session_state.results = results
                st.session_state.pop("search_results", None)

            except (OSError, RuntimeError) as e:
                st.error(str(e))
            except Exception as e:
                st.exception(e)

    if "results" in st.session_state and st.session_state.results:
        all_results: dict[str, EmbedResults] = st.session_state.results

        st.success(f"Embedded {len(all_results)} document(s).")

        # Summary metrics
        total_pages = sum(len(r["pages"]) for r in all_results.values())
        total_duration_ns = sum(r["total_duration"] for r in all_results.values())

        st.subheader("Metrics")
        st.metric("Model", MODEL_ID)
        col1, col2, col3 = st.columns(3)
        col1.metric("Duration", f"{total_duration_ns / 1_000_000_000:.2f} s")
        col2.metric("Pages", total_pages)
        col3.metric("Documents", len(all_results))

        # Per-document expanders
        for file_id, r in all_results.items():
            with st.expander(f"{r['file_stem']} ({len(r['pages'])} pages)"):
                cols = st.columns(min(len(r["pages"]), 4))
                for i, page in enumerate(r["pages"]):
                    cols[i % 4].image(page, caption=f"Page {i + 1}", width="stretch")
                doc_duration = r["total_duration"] / 1_000_000_000
                st.caption(
                    f"Duration: {doc_duration:.2f} s · "
                    f"Pages: {len(r['pages'])} · DPI: {r['dpi']}"
                )
                st.download_button(
                    label=f"Download {r['file_stem']} JSON",
                    data=r["json"],
                    file_name=f"{r['file_stem']}_embedding.json",
                    mime="application/json",
                    key=f"download_{file_id}",
                )

        # Download All
        if len(all_results) > 1:
            all_json = "[" + ",".join(r["json"] for r in all_results.values()) + "]"
            st.download_button(
                label="Download All JSON",
                data=all_json,
                file_name="all_embeddings.json",
                mime="application/json",
                key="download_all",
            )

        # Search
        st.subheader("Search")
        filter_options = ["All documents"] + [
            r["file_stem"] for r in all_results.values()
        ]
        filter_file_ids: list[str | None] = [None] + list(all_results.keys())
        filter_idx = st.selectbox(
            "Document filter",
            range(len(filter_options)),
            format_func=lambda i: filter_options[i],
        )
        selected_file_id = filter_file_ids[filter_idx]

        col_topk, col_minscore = st.columns(2)
        top_k = col_topk.number_input("Top K", min_value=1, max_value=100, value=5)
        min_score = col_minscore.number_input(
            "Min score", min_value=0.0, value=0.0, step=0.1
        )

        query = st.text_input("Text query")
        if st.button("Search"):
            if not query:
                st.warning("Enter a search query.")
            else:
                try:
                    model, processor = load_model(device)
                    raw_results = search_multi(
                        query,
                        model,
                        processor,
                        all_results,
                        filter_file_id=selected_file_id,
                    )
                    st.session_state.search_results = filter_results(
                        raw_results, top_k=top_k, min_score=min_score
                    )
                except (OSError, RuntimeError, ValueError) as e:
                    st.error(str(e))
                except Exception as e:
                    st.exception(e)

        if "search_results" in st.session_state:
            search_results = st.session_state.search_results
            if search_results:
                cols = st.columns(min(len(search_results), 4))
                for rank, (fid, page_idx, score) in enumerate(search_results):
                    r = all_results[fid]
                    cols[rank % 4].image(
                        r["pages"][page_idx],
                        caption=(
                            f"{r['file_stem']} · Page {page_idx + 1} · {score:.4f}"
                        ),
                        width="stretch",
                    )
            else:
                st.info("No results above the score threshold.")

st.caption(f"Device: {device.upper()}")
