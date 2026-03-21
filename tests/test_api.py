from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

PDF_DATA_DIR = Path(__file__).parent / "data" / "pdf"
IMAGE_DATA_DIR = Path(__file__).parent / "data" / "images"


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Create a TestClient with temp directories and mocked worker."""
    uploads = tmp_path / "uploads"
    results = tmp_path / "results"
    db_path = tmp_path / "test.db"
    uploads.mkdir()
    results.mkdir()

    with (
        patch.dict("os.environ", {
            "UPLOAD_DIR": str(uploads),
            "RESULT_DIR": str(results),
            "DATABASE_PATH": str(db_path),
        }),
        patch("api.app.EmbeddingWorker") as MockWorker,
    ):
        mock_worker = MagicMock()
        mock_worker.is_running = True
        MockWorker.return_value = mock_worker

        import api.app
        api.app._db = None
        api.app._worker = None

        from api.app import create_app
        app = create_app()
        with TestClient(app) as tc:
            tc._mock_worker = mock_worker
            yield tc


class TestHealth:
    def test_returns_health(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "device" in data
        assert "queue_depth" in data
        assert "worker_running" in data


class TestUploadJob:
    def test_upload_pdf(self, client: TestClient) -> None:
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        resp = client.post(
            "/jobs",
            files={"file": ("test.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    def test_upload_image(self, client: TestClient) -> None:
        img_data = (IMAGE_DATA_DIR / "red.png").read_bytes()
        resp = client.post(
            "/jobs",
            files={"file": ("red.png", img_data, "image/png")},
            data={"dpi": "150"},
        )
        assert resp.status_code == 201

    def test_rejects_invalid_file_type(self, client: TestClient) -> None:
        resp = client.post(
            "/jobs",
            files={"file": ("test.txt", b"hello", "text/plain")},
            data={"dpi": "150"},
        )
        assert resp.status_code == 400

    def test_rejects_oversized_file(self, client: TestClient) -> None:
        big_data = b"x" * (50 * 1024 * 1024 + 1)
        resp = client.post(
            "/jobs",
            files={"file": ("big.pdf", big_data, "application/pdf")},
            data={"dpi": "150"},
        )
        assert resp.status_code == 400


class TestListJobs:
    def test_list_empty(self, client: TestClient) -> None:
        resp = client.get("/jobs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_after_upload(self, client: TestClient) -> None:
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        client.post("/jobs", files={"file": ("test.pdf", pdf_data, "application/pdf")}, data={"dpi": "150"})
        resp = client.get("/jobs")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_filter_by_status(self, client: TestClient) -> None:
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        client.post("/jobs", files={"file": ("test.pdf", pdf_data, "application/pdf")}, data={"dpi": "150"})
        resp = client.get("/jobs?status=completed")
        assert resp.status_code == 200
        assert len(resp.json()) == 0


class TestGetJob:
    def test_get_existing(self, client: TestClient) -> None:
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = client.post("/jobs", files={"file": ("test.pdf", pdf_data, "application/pdf")}, data={"dpi": "150"})
        job_id = create_resp.json()["job_id"]
        resp = client.get(f"/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == job_id

    def test_get_nonexistent(self, client: TestClient) -> None:
        resp = client.get("/jobs/nonexistent")
        assert resp.status_code == 404


class TestDeleteJob:
    def test_delete_pending(self, client: TestClient) -> None:
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = client.post("/jobs", files={"file": ("test.pdf", pdf_data, "application/pdf")}, data={"dpi": "150"})
        job_id = create_resp.json()["job_id"]
        resp = client.delete(f"/jobs/{job_id}")
        assert resp.status_code == 204

    def test_delete_nonexistent(self, client: TestClient) -> None:
        resp = client.delete("/jobs/nonexistent")
        assert resp.status_code == 404

    def test_delete_processing_returns_409(self, client: TestClient) -> None:
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = client.post("/jobs", files={"file": ("test.pdf", pdf_data, "application/pdf")}, data={"dpi": "150"})
        job_id = create_resp.json()["job_id"]
        # Manually set to processing
        from api.database import update_job
        from api.app import _get_db
        db = _get_db()
        update_job(db, job_id, status="processing")
        resp = client.delete(f"/jobs/{job_id}")
        assert resp.status_code == 409


class TestGetResult:
    def test_returns_404_for_pending(self, client: TestClient) -> None:
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = client.post("/jobs", files={"file": ("test.pdf", pdf_data, "application/pdf")}, data={"dpi": "150"})
        job_id = create_resp.json()["job_id"]
        resp = client.get(f"/jobs/{job_id}/result")
        assert resp.status_code == 404


class TestSearch:
    def test_returns_empty_with_no_completed_jobs(self, client: TestClient) -> None:
        resp = client.post("/search", json={"query": "test"})
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    def test_enqueues_search_to_worker(self, client: TestClient) -> None:
        from concurrent.futures import Future
        future: Future = Future()
        future.set_result([("job1", 0, 0.9)])
        client._mock_worker.enqueue_search.return_value = future

        # Create and manually complete a job
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = client.post("/jobs", files={"file": ("test.pdf", pdf_data, "application/pdf")}, data={"dpi": "150"})
        job_id = create_resp.json()["job_id"]
        from api.app import _get_db
        from api.database import update_job
        db = _get_db()
        update_job(db, job_id, status="completed", page_count=1, duration_ns=100, result_path="r.json", tensor_path="r.pt")

        resp = client.post("/search", json={"query": "charts", "top_k": 5})
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 1
        assert results[0]["file_id"] == "job1"
        client._mock_worker.enqueue_search.assert_called_once()
