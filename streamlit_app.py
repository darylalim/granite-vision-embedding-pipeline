import os
import time  # noqa: F401
from datetime import timedelta  # noqa: F401

import httpx
import pandas as pd  # noqa: F401
import streamlit as st

from core.constants import DPI_OPTIONS, IMAGE_EXTENSIONS, MODEL_ID  # noqa: F401

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
    st.info("Upload tab — coming next.")

with tab_jobs:
    st.info("Jobs tab — coming soon.")

with tab_query:
    st.info("Query tab — coming soon.")
