import os

import httpx
import streamlit as st

from core.constants import DPI_OPTIONS, IMAGE_EXTENSIONS, MODEL_ID

API_URL = os.environ.get("API_URL", "http://localhost:8000")


def api_client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=120.0)


st.set_page_config(page_title="Granite Vision Embedding Pipeline", layout="centered")
st.title("Granite Vision Embedding Pipeline")
st.write(
    "Generate vector embeddings from PDFs and images with Granite Vision Embedding Pipeline."
)

uploaded_files = st.file_uploader(
    "Upload files",
    type=["pdf", *IMAGE_EXTENSIONS],
    accept_multiple_files=True,
)

dpi_label = st.radio("Render DPI", DPI_OPTIONS, index=1, horizontal=True)
dpi = DPI_OPTIONS[dpi_label]

if uploaded_files:
    total_size_mb = sum(len(f.getvalue()) for f in uploaded_files) / 1_048_576
    st.caption(f"{len(uploaded_files)} file(s) · {total_size_mb:.1f} MB")

    if st.button("Submit Jobs", type="primary"):
        with api_client() as client:
            for f in uploaded_files:
                try:
                    resp = client.post(
                        "/jobs",
                        files={"file": (f.name, f.getvalue())},
                        data={"dpi": str(dpi)},
                    )
                    if resp.status_code == 201:
                        st.success(f"Submitted: {f.name}")
                    else:
                        st.error(
                            f"{f.name}: {resp.json().get('detail', 'Unknown error')}"
                        )
                except httpx.HTTPError as e:
                    st.error(f"{f.name}: {e}")

# Job Dashboard
st.subheader("Jobs")

col_refresh, col_filter = st.columns([1, 2])
if col_refresh.button("Refresh"):
    st.rerun()

status_filter = col_filter.selectbox(
    "Status filter",
    ["all", "pending", "processing", "completed", "failed"],
)

try:
    with api_client() as client:
        params = {} if status_filter == "all" else {"status": status_filter}
        resp = client.get("/jobs", params=params)
        jobs = resp.json() if resp.status_code == 200 else []
except httpx.HTTPError:
    jobs = []
    st.warning("Cannot connect to API server.")

if jobs:
    # Summary metrics
    completed = [j for j in jobs if j["status"] == "completed"]
    total_pages = sum(j.get("page_count") or 0 for j in completed)
    total_duration_ns = sum(j.get("duration_ns") or 0 for j in completed)

    if completed:
        st.subheader("Metrics")
        st.metric("Model", MODEL_ID)
        col1, col2, col3 = st.columns(3)
        col1.metric("Duration", f"{total_duration_ns / 1_000_000_000:.2f} s")
        col2.metric("Pages", total_pages)
        col3.metric("Documents", len(completed))

    for job in jobs:
        status_emoji = {
            "pending": "Pending",
            "processing": "Processing",
            "completed": "Completed",
            "failed": "Failed",
        }.get(job["status"], job["status"])

        with st.expander(f"{job['file_stem']} — {status_emoji}"):
            st.caption(
                f"Status: {job['status']} · Type: {job['file_type']} · DPI: {job['dpi']}"
            )
            if job.get("page_count"):
                duration_s = (job.get("duration_ns") or 0) / 1_000_000_000
                st.caption(f"Pages: {job['page_count']} · Duration: {duration_s:.2f} s")
            if job.get("error"):
                st.error(job["error"])

            col_dl, col_del = st.columns(2)

            if job["status"] == "completed":
                try:
                    with api_client() as client:
                        result_resp = client.get(f"/jobs/{job['id']}/result")
                        if result_resp.status_code == 200:
                            col_dl.download_button(
                                f"Download {job['file_stem']} JSON",
                                data=result_resp.content,
                                file_name=f"{job['file_stem']}_embedding.json",
                                mime="application/json",
                                key=f"dl_{job['id']}",
                            )
                except httpx.HTTPError:
                    pass

            if job["status"] != "processing":
                if col_del.button("Delete", key=f"del_{job['id']}"):
                    try:
                        with api_client() as client:
                            client.delete(f"/jobs/{job['id']}")
                        st.session_state.pop("search_results", None)
                        st.rerun()
                    except httpx.HTTPError as e:
                        st.error(str(e))

    # Download All
    if len(completed) > 1:
        try:
            with api_client() as client:
                all_results = []
                for j in completed:
                    result_resp = client.get(f"/jobs/{j['id']}/result")
                    if result_resp.status_code == 200:
                        all_results.append(result_resp.text)
                if all_results:
                    all_json = "[" + ",".join(all_results) + "]"
                    st.download_button(
                        "Download All JSON",
                        data=all_json,
                        file_name="all_embeddings.json",
                        mime="application/json",
                        key="download_all",
                    )
        except httpx.HTTPError:
            pass

    # Search
    if completed:
        st.subheader("Search")
        filter_options = ["All documents"] + [j["file_stem"] for j in completed]
        filter_ids: list[str | None] = [None] + [j["id"] for j in completed]
        filter_idx = st.selectbox(
            "Document filter",
            range(len(filter_options)),
            format_func=lambda i: filter_options[i],
        )

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
                    with api_client() as client:
                        search_resp = client.post(
                            "/search",
                            json={
                                "query": query,
                                "top_k": top_k,
                                "min_score": min_score,
                                "filter_file_id": filter_ids[filter_idx],
                            },
                        )
                        if search_resp.status_code == 200:
                            st.session_state.search_results = search_resp.json()[
                                "results"
                            ]
                        else:
                            st.error(search_resp.json().get("detail", "Search failed"))
                except httpx.HTTPError as e:
                    st.error(str(e))

        if "search_results" in st.session_state:
            search_results = st.session_state.search_results
            if search_results:
                job_lookup = {j["id"]: j for j in completed}
                for rank, sr in enumerate(search_results):
                    j = job_lookup.get(sr["file_id"])
                    if j:
                        st.caption(
                            f"{j['file_stem']} · Page {sr['page_index'] + 1} · {sr['score']:.4f}"
                        )
            else:
                st.info("No results above the score threshold.")

elif not uploaded_files:
    st.info("Upload files to get started.")

# Footer
try:
    with api_client() as client:
        health = client.get("/health").json()
        device = health.get("device", "unknown").upper()
        queue_depth = health.get("queue_depth", 0)
        st.caption(f"Device: {device} · Queue: {queue_depth}")
except httpx.HTTPError:
    st.caption("API server not connected")
