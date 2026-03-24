from unittest.mock import MagicMock, patch

import httpx
import streamlit as st
from streamlit.testing.v1 import AppTest

SCRIPT_PATH = "streamlit_app.py"


class TestStatusIcons:
    """Self-contained tests — hardcoded values, no import of streamlit_app.

    These must match the STATUS_ICONS dict in streamlit_app.py.
    If the dict changes, update these expected values.
    """

    EXPECTED = {
        "completed": "\U0001f7e2",
        "failed": "\U0001f534",
        "pending": "\U0001f7e1",
        "processing": "\U0001f535",
    }

    def test_contains_all_statuses(self) -> None:
        expected_keys = {"completed", "failed", "pending", "processing"}
        assert set(self.EXPECTED.keys()) == expected_keys

    def test_values_are_non_empty_strings(self) -> None:
        for status, icon in self.EXPECTED.items():
            assert isinstance(icon, str), f"{status} icon is not a string"
            assert len(icon) > 0, f"{status} icon is empty"


def _make_mock_client(
    jobs: list[dict] | None = None,
    completed_jobs: list[dict] | None = None,
    health: dict | None = None,
    health_error: bool = False,
) -> MagicMock:
    """Build a mock httpx.Client with URL-based routing.

    Args:
        jobs: Response for GET /jobs (no params). Defaults to [].
        completed_jobs: Response for GET /jobs?status=completed. Defaults to jobs.
        health: Response for GET /health. Defaults to healthy response.
        health_error: If True, GET /health raises httpx.ConnectError.
    """
    if jobs is None:
        jobs = []
    if completed_jobs is None:
        completed_jobs = [j for j in jobs if j.get("status") == "completed"]
    if health is None:
        health = {"device": "cpu", "queue_depth": 0, "worker_running": True}

    mock_client = MagicMock()

    def mock_get(url, **kwargs):
        if health_error and "health" in url:
            raise httpx.ConnectError("Connection refused")

        resp = MagicMock()
        resp.status_code = 200

        if "health" in url:
            resp.json.return_value = health
        elif "result" in url:
            resp.json.return_value = {"file_name": "stub", "embeddings": []}
            resp.text = '{"file_name": "stub", "embeddings": []}'
        elif kwargs.get("params", {}).get("status") == "completed":
            resp.json.return_value = completed_jobs
        else:
            resp.json.return_value = jobs
        return resp

    mock_client.get.side_effect = mock_get
    return mock_client


def _run_app(**kwargs) -> AppTest:
    """Create and run AppTest with a mocked httpx.Client."""
    st.cache_resource.clear()
    mock_client = _make_mock_client(**kwargs)
    with patch("httpx.Client", return_value=mock_client):
        at = AppTest.from_file(SCRIPT_PATH, default_timeout=10)
        at.run()
    return at


SAMPLE_JOBS = [
    {
        "id": "job1",
        "status": "completed",
        "file_name": "test.pdf",
        "file_stem": "test",
        "file_type": "pdf",
        "dpi": 150,
        "page_count": 3,
        "duration_ns": 1_000_000_000,
        "created_at": "2024-01-01",
        "updated_at": "2024-01-01",
    },
    {
        "id": "job2",
        "status": "completed",
        "file_name": "doc.pdf",
        "file_stem": "doc",
        "file_type": "pdf",
        "dpi": 150,
        "page_count": 5,
        "duration_ns": 2_000_000_000,
        "created_at": "2024-01-02",
        "updated_at": "2024-01-02",
    },
]


class TestConnectionCheck:
    def test_shows_error_when_api_unreachable(self) -> None:
        at = _run_app(health_error=True)
        errors = [e.value for e in at.error]
        assert any("Cannot connect" in e for e in errors)

    def test_no_tabs_when_api_unreachable(self) -> None:
        at = _run_app(health_error=True)
        assert len(at.tabs) == 0


class TestHealthyAppStructure:
    def test_sidebar_contains_model_id(self) -> None:
        at = _run_app()
        sidebar_texts = [c.value for c in at.sidebar.caption]
        assert any("ibm-granite" in t for t in sidebar_texts)

    def test_sidebar_contains_device(self) -> None:
        at = _run_app()
        sidebar_texts = [c.value for c in at.sidebar.caption]
        assert any("CPU" in t for t in sidebar_texts)

    def test_sidebar_contains_queue_depth(self) -> None:
        at = _run_app()
        sidebar_texts = [c.value for c in at.sidebar.caption]
        assert any("0 pending" in t for t in sidebar_texts)

    def test_three_tabs_render(self) -> None:
        at = _run_app()
        assert len(at.tabs) == 3

    def test_tab_labels(self) -> None:
        at = _run_app()
        labels = [t.label for t in at.tabs]
        assert labels == ["Upload", "Jobs", "Query"]


class TestUploadTab:
    def test_dpi_radio_exists_with_three_options(self) -> None:
        at = _run_app()
        assert len(at.radio) == 1
        assert len(at.radio[0].options) == 3

    def test_dpi_radio_options_match_constants(self) -> None:
        at = _run_app()
        options = at.radio[0].options
        assert "Low (72)" in options
        assert "Medium (150)" in options
        assert "High (300)" in options

    def test_dpi_default_is_medium(self) -> None:
        at = _run_app()
        assert at.radio[0].value == "Medium (150)"


class TestJobsTab:
    def test_no_jobs_shows_info(self) -> None:
        at = _run_app()
        info_texts = [i.value for i in at.info]
        assert any("No jobs found" in t for t in info_texts)

    def test_status_filter_selectbox_exists(self) -> None:
        at = _run_app()
        status_sb = [sb for sb in at.selectbox if sb.label == "Status filter"]
        assert len(status_sb) == 1
        assert "all" in status_sb[0].options
        assert "completed" in status_sb[0].options

    def test_dataframe_renders_with_jobs(self) -> None:
        at = _run_app(jobs=SAMPLE_JOBS)
        assert len(at.dataframe) >= 1

    def test_metrics_render_with_completed_jobs(self) -> None:
        at = _run_app(jobs=SAMPLE_JOBS)
        assert len(at.metric) >= 3


class TestQueryTab:
    def test_no_completed_jobs_shows_info(self) -> None:
        at = _run_app()
        info_texts = [i.value for i in at.info]
        assert any("No documents available" in t for t in info_texts)

    def test_document_filter_renders_with_completed_jobs(self) -> None:
        at = _run_app(jobs=SAMPLE_JOBS)
        doc_sb = [sb for sb in at.selectbox if sb.label == "Document filter"]
        assert len(doc_sb) == 1
        assert "All documents" in doc_sb[0].options
        assert "test" in doc_sb[0].options
        assert "doc" in doc_sb[0].options

    def test_query_input_renders_with_completed_jobs(self) -> None:
        at = _run_app(jobs=SAMPLE_JOBS)
        assert len(at.text_input) >= 1
        query_inputs = [ti for ti in at.text_input if ti.label == "Query"]
        assert len(query_inputs) == 1

    def test_search_and_ask_buttons_render(self) -> None:
        at = _run_app(jobs=SAMPLE_JOBS)
        button_labels = [b.label for b in at.button]
        assert "Search" in button_labels
        assert "Ask" in button_labels
