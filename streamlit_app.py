import os
import time
from datetime import timedelta

import httpx
import pandas as pd
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


@st.dialog("Confirm Delete All")
def confirm_delete_all():
    st.write(
        "This will delete all jobs and their files. "
        "Jobs currently processing will be kept."
    )
    col_yes, col_no = st.columns(2)
    if col_yes.button("Delete", type="primary", key="confirm_delete"):
        try:
            client = api_client()
            resp = client.delete("/jobs")
            count = resp.json().get("deleted", 0)
            st.session_state.pop("search_results", None)
            st.session_state.pop("ask_result", None)
            st.session_state.pop("selected_job_id", None)
            st.toast(f"Deleted {count} job(s).")
            st.rerun()
        except httpx.HTTPError as e:
            st.error(str(e))
    if col_no.button("Cancel", key="cancel_delete"):
        st.rerun()


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
        if job_ids:
            st.rerun()

with tab_jobs:
    # --- Controls (outside fragment) ---
    col_filter, col_delete_all = st.columns([3, 1])
    status_filter = col_filter.selectbox(
        "Status filter",
        ["all", "pending", "processing", "completed", "failed"],
    )
    if col_delete_all.button("Delete All"):
        confirm_delete_all()

    # --- Auto-refreshing fragment ---
    @st.fragment(run_every=timedelta(seconds=5))
    def jobs_fragment():
        client = api_client()

        # Always fetch all jobs for metrics
        try:
            all_resp = client.get("/jobs")
            all_jobs = all_resp.json() if all_resp.status_code == 200 else []
        except httpx.HTTPError:
            all_jobs = []

        all_completed = [j for j in all_jobs if j["status"] == "completed"]

        # Metrics (always based on all completed jobs)
        if all_completed:
            total_pages = sum(j.get("page_count") or 0 for j in all_completed)
            total_duration_ns = sum(j.get("duration_ns") or 0 for j in all_completed)
            col1, col2, col3 = st.columns(3)
            col1.metric("Duration", f"{total_duration_ns / 1_000_000_000:.2f} s")
            col2.metric("Pages", total_pages)
            col3.metric("Documents", len(all_completed))

        # Download All
        if len(all_completed) > 1:
            all_results = []
            for j in all_completed:
                try:
                    result_resp = client.get(f"/jobs/{j['id']}/result")
                    if result_resp.status_code == 200:
                        all_results.append(result_resp.text)
                except httpx.HTTPError:
                    pass
            if all_results:
                all_json = "[" + ",".join(all_results) + "]"
                st.download_button(
                    "Download All JSON",
                    data=all_json,
                    file_name="all_embeddings.json",
                    mime="application/json",
                    key="download_all",
                )

        # Filtered job list for dataframe
        if status_filter == "all":
            filtered_jobs = all_jobs
        else:
            filtered_jobs = [j for j in all_jobs if j["status"] == status_filter]

        if not filtered_jobs:
            st.info("No jobs found.")
            return

        # Build dataframe
        rows = []
        for j in filtered_jobs:
            icon = STATUS_ICONS.get(j["status"], "")
            duration_s = (j.get("duration_ns") or 0) / 1_000_000_000
            rows.append(
                {
                    "Name": j["file_stem"],
                    "Status": f"{icon} {j['status'].capitalize()}",
                    "Type": j["file_type"],
                    "DPI": j["dpi"],
                    "Pages": j.get("page_count") or "",
                    "Duration": f"{duration_s:.2f} s" if duration_s > 0 else "",
                    "_id": j["id"],
                }
            )

        df = pd.DataFrame(rows)
        display_df = df.drop(columns=["_id"])

        event = st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun",
            key="jobs_table",
        )

        # Persist selection
        selected_rows = event.selection.rows if event.selection else []
        if selected_rows:
            selected_idx = selected_rows[0]
            st.session_state["selected_job_id"] = df.iloc[selected_idx]["_id"]
        else:
            st.session_state.pop("selected_job_id", None)

    jobs_fragment()

    # --- Detail Panel (outside fragment) ---
    selected_job_id = st.session_state.get("selected_job_id")
    if selected_job_id:
        try:
            client = api_client()
            resp = client.get(f"/jobs/{selected_job_id}")
            if resp.status_code == 200:
                job = resp.json()
                st.subheader(f"{job['file_stem']}")
                st.caption(
                    f"Status: {job['status']} \u00b7 Type: {job['file_type']} "
                    f"\u00b7 DPI: {job['dpi']}"
                )
                if job.get("page_count"):
                    duration_s = (job.get("duration_ns") or 0) / 1_000_000_000
                    st.caption(
                        f"Pages: {job['page_count']} \u00b7 Duration: {duration_s:.2f} s"
                    )
                if job.get("error"):
                    st.error(job["error"])

                col_dl, col_del = st.columns(2)

                if job["status"] == "completed":
                    try:
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
                            client.delete(f"/jobs/{job['id']}")
                            st.session_state.pop("selected_job_id", None)
                            st.session_state.pop("search_results", None)
                            st.session_state.pop("ask_result", None)
                            st.toast(f"Deleted {job['file_stem']}.")
                            st.rerun()
                        except httpx.HTTPError as e:
                            st.error(str(e))
            else:
                # Job no longer exists
                st.session_state.pop("selected_job_id", None)
        except httpx.HTTPError:
            st.session_state.pop("selected_job_id", None)

with tab_query:
    st.info("Query tab — coming soon.")
