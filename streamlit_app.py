import os
import time

import httpx
import streamlit as st

from core.constants import DPI_OPTIONS, IMAGE_EXTENSIONS, MODEL_ID

API_URL = os.environ.get("API_URL", "http://localhost:8000")


@st.cache_resource
def api_client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=120.0)


def document_filter(completed: list[dict], key_prefix: str = "") -> str | None:
    """Render a document filter selectbox and return the selected file ID."""
    options = ["All documents"] + [j["file_stem"] for j in completed]
    ids: list[str | None] = [None] + [j["id"] for j in completed]
    idx = st.selectbox(
        "Document filter",
        range(len(options)),
        format_func=lambda i: options[i],
        key=f"{key_prefix}filter",
    )
    return ids[idx]


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
        job_ids: list[str] = []
        submit_errors: list[str] = []
        client = api_client()
        for f in uploaded_files:
            try:
                resp = client.post(
                    "/jobs",
                    files={"file": (f.name, f.getvalue())},
                    data={"dpi": str(dpi)},
                )
                if resp.status_code == 201:
                    job_ids.append(resp.json()["job_id"])
                else:
                    submit_errors.append(
                        f"{f.name}: {resp.json().get('detail', 'Unknown error')}"
                    )
            except httpx.HTTPError as e:
                submit_errors.append(f"{f.name}: {e}")

        for err in submit_errors:
            st.error(err)

        if job_ids:
            total = len(job_ids)
            max_polls = 150  # 5 minutes at 2-second intervals
            with st.status(f"Processing {total} job(s)...", expanded=True) as status:
                progress_bar = st.progress(0)
                progress_text = st.empty()
                polls = 0
                timed_out = False
                client = api_client()
                while True:
                    try:
                        resp = client.get("/jobs")
                        all_jobs = resp.json() if resp.status_code == 200 else []
                    except httpx.HTTPError:
                        all_jobs = []
                    statuses = {j["id"]: j["status"] for j in all_jobs}
                    completed = sum(
                        1 for jid in job_ids if statuses.get(jid) == "completed"
                    )
                    failed = sum(1 for jid in job_ids if statuses.get(jid) == "failed")
                    done = completed + failed
                    progress_bar.progress(done / total)
                    if failed:
                        progress_text.text(f"{done}/{total} done — {failed} failed")
                    else:
                        progress_text.text(f"{done}/{total} completed")
                    if done >= total:
                        break
                    polls += 1
                    if polls >= max_polls:
                        timed_out = True
                        break
                    time.sleep(2)

                if timed_out:
                    status.update(
                        label="Polling timed out — check job dashboard", state="error"
                    )
                elif failed:
                    status.update(label="Jobs finished with errors", state="error")
                else:
                    status.update(label="All jobs completed", state="complete")
            st.rerun()

# Job Dashboard
st.subheader("Jobs")

col_refresh, col_delete_all, col_filter = st.columns([1, 1, 2])
if col_refresh.button("Refresh"):
    st.rerun()


@st.dialog("Confirm Delete All")
def confirm_delete_all():
    st.write(
        "This will delete all jobs and their files. Jobs currently processing will be kept."
    )
    col_yes, col_no = st.columns(2)
    if col_yes.button("Delete", type="primary", key="confirm_delete"):
        try:
            client = api_client()
            resp = client.delete("/jobs")
            count = resp.json().get("deleted", 0)
            st.session_state.pop("search_results", None)
            st.session_state.pop("ask_result", None)
            st.toast(f"Deleted {count} job(s).")
            st.rerun()
        except httpx.HTTPError as e:
            st.error(str(e))
    if col_no.button("Cancel", key="cancel_delete"):
        st.rerun()


if col_delete_all.button("Delete All"):
    confirm_delete_all()

status_filter = col_filter.selectbox(
    "Status filter",
    ["all", "pending", "processing", "completed", "failed"],
)

try:
    client = api_client()
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
        with st.expander(f"{job['file_stem']} — {job['status'].capitalize()}"):
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
                    client = api_client()
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
                        client = api_client()
                        client.delete(f"/jobs/{job['id']}")
                        st.session_state.pop("search_results", None)
                        st.session_state.pop("ask_result", None)
                        st.rerun()
                    except httpx.HTTPError as e:
                        st.error(str(e))

    # Download All
    if len(completed) > 1:
        try:
            client = api_client()
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

    # Shared lookup for search and ask results
    if completed:
        job_lookup = {j["id"]: j for j in completed}

    # Search
    if completed:
        st.subheader("Search")
        filter_file_id = document_filter(completed, key_prefix="search_")

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
                    client = api_client()
                    search_resp = client.post(
                        "/search",
                        json={
                            "query": query,
                            "top_k": top_k,
                            "min_score": min_score,
                            "filter_file_id": filter_file_id,
                        },
                    )
                    if search_resp.status_code == 200:
                        st.session_state.search_results = search_resp.json()["results"]
                    else:
                        st.error(search_resp.json().get("detail", "Search failed"))
                except httpx.HTTPError as e:
                    st.error(str(e))

        if "search_results" in st.session_state:
            search_results = st.session_state.search_results
            if search_results:
                for rank, sr in enumerate(search_results):
                    j = job_lookup.get(sr["file_id"])
                    if j:
                        st.caption(
                            f"{j['file_stem']} · Page {sr['page_index'] + 1} · {sr['score']:.4f}"
                        )
            else:
                st.info("No results above the score threshold.")

        # Ask
        st.subheader("Ask")
        ask_filter_file_id = document_filter(completed, key_prefix="ask_")

        ask_col_topk, ask_col_minscore = st.columns(2)
        ask_top_k = ask_col_topk.number_input(
            "Top K", min_value=1, max_value=10, value=3, key="ask_top_k"
        )
        ask_min_score = ask_col_minscore.number_input(
            "Min score", min_value=0.0, value=0.0, step=0.1, key="ask_min_score"
        )

        ask_query = st.text_input("Question", key="ask_query")
        if st.button("Ask", key="ask_button"):
            if not ask_query:
                st.warning("Enter a question.")
            else:
                try:
                    client = api_client()
                    ask_resp = client.post(
                        "/ask",
                        json={
                            "query": ask_query,
                            "top_k": ask_top_k,
                            "min_score": ask_min_score,
                            "filter_file_id": ask_filter_file_id,
                        },
                    )
                    if ask_resp.status_code == 200:
                        st.session_state.ask_result = ask_resp.json()
                    elif ask_resp.status_code == 503:
                        st.warning(
                            "Answer generation is not configured. "
                            "Set GENERATION_API_URL and GENERATION_MODEL to enable."
                        )
                    else:
                        st.error(ask_resp.json().get("detail", "Ask failed"))
                except httpx.HTTPError as e:
                    st.error(str(e))

        if "ask_result" in st.session_state:
            ask_result = st.session_state.ask_result
            st.markdown(ask_result["answer"])
            if ask_result["sources"]:
                st.caption("Sources:")
                for sr in ask_result["sources"]:
                    j = job_lookup.get(sr["file_id"])
                    if j:
                        st.caption(
                            f"  {j['file_stem']} · Page {sr['page_index'] + 1} · {sr['score']:.4f}"
                        )

elif not uploaded_files:
    st.info("Upload files to get started.")

# Footer
try:
    client = api_client()
    health = client.get("/health").json()
    device = health.get("device", "unknown").upper()
    queue_depth = health.get("queue_depth", 0)
    st.caption(f"Device: {device} · Queue: {queue_depth}")
except httpx.HTTPError:
    st.caption("API server not connected")
