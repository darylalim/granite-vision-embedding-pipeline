import os
import time
from datetime import timedelta  # noqa: F401

import httpx
import pandas as pd  # noqa: F401
import streamlit as st

from core.constants import DPI_OPTIONS, IMAGE_EXTENSIONS, MODEL_ID

API_URL = os.environ.get("API_URL", "http://localhost:8000")

STATUS_ICONS = {
    "completed": "\U0001f7e2",  # green circle
    "failed": "\U0001f534",  # red circle
    "pending": "\U0001f7e1",  # yellow circle
    "processing": "\U0001f535",  # blue circle
}


@st.cache_resource
def api_client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=120.0)


def document_filter(completed: list[dict]) -> str | None:
    """Render a document filter selectbox and return the selected file ID."""
    options = ["All documents"] + [j["file_stem"] for j in completed]
    ids: list[str | None] = [None] + [j["id"] for j in completed]
    idx = st.selectbox(
        "Document filter",
        range(len(options)),
        format_func=lambda i: options[i],
    )
    return ids[idx]


# --- Page Config ---
st.set_page_config(page_title="Granite Vision Embedding Pipeline", layout="wide")
st.title("Granite Vision Embedding Pipeline")

# --- Connection Check ---
try:
    _health = api_client().get("/health").json()
except httpx.HTTPError:
    st.error(
        "Cannot connect to API server. "
        "Start it with: `uv run uvicorn api.app:create_app --factory --port 8000`"
    )
    st.stop()

# --- Sidebar ---
st.sidebar.caption(f"**Model:** {MODEL_ID}")
st.sidebar.caption(f"**Device:** {_health.get('device', 'unknown').upper()}")
st.sidebar.caption(f"**Queue:** {_health.get('queue_depth', 0)} pending")

# --- Tabs ---
tab_upload, tab_jobs, tab_query = st.tabs(["Upload", "Jobs", "Query"])

with tab_upload:
    uploaded_files = st.file_uploader(
        "Drop PDFs or images here (PNG, JPG, WebP) — up to 50 MB each",
        type=["pdf", *IMAGE_EXTENSIONS],
        accept_multiple_files=True,
    )

    with st.expander("Advanced options"):
        dpi_label = st.radio(
            "Render DPI",
            DPI_OPTIONS,
            index=1,
            horizontal=True,
            help="Higher DPI = better quality but slower processing",
        )
    dpi = DPI_OPTIONS[dpi_label]

    if uploaded_files:
        total_size_mb = sum(len(f.getvalue()) for f in uploaded_files) / 1_048_576
        st.caption(f"{len(uploaded_files)} file(s) \u00b7 {total_size_mb:.1f} MB")

    uploading = st.session_state.get("uploading", False)

    if uploaded_files and st.button("Submit Jobs", type="primary", disabled=uploading):
        st.session_state["uploading"] = True
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
            max_polls = 150
            with st.status(f"Processing {total} job(s)...", expanded=True) as status:
                progress_bar = st.progress(0)
                progress_text = st.empty()
                polls = 0
                timed_out = False
                while True:
                    try:
                        resp = client.get("/jobs")
                        all_jobs = resp.json() if resp.status_code == 200 else []
                    except httpx.HTTPError:
                        all_jobs = []
                    statuses = {j["id"]: j["status"] for j in all_jobs}
                    n_completed = sum(
                        1 for jid in job_ids if statuses.get(jid) == "completed"
                    )
                    n_failed = sum(
                        1 for jid in job_ids if statuses.get(jid) == "failed"
                    )
                    done = n_completed + n_failed
                    progress_bar.progress(done / total)
                    if n_failed:
                        progress_text.text(
                            f"{done}/{total} done \u2014 {n_failed} failed"
                        )
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
                        label="Polling timed out \u2014 check Jobs tab",
                        state="error",
                    )
                elif n_failed:
                    status.update(label="Jobs finished with errors", state="error")
                else:
                    status.update(label="All jobs completed", state="complete")
                    st.toast(f"Completed {total} job(s).")

        st.session_state["uploading"] = False
        st.rerun()

with tab_jobs:
    st.info("Jobs tab — coming soon.")

with tab_query:
    st.info("Query tab — coming soon.")
