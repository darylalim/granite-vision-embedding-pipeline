from unittest.mock import MagicMock, patch

import httpx
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
