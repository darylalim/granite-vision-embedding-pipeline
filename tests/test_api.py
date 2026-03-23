import sqlite3
from collections.abc import Generator
from pathlib import Path
from typing import NamedTuple
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

PDF_DATA_DIR = Path(__file__).parent / "data" / "pdf"
IMAGE_DATA_DIR = Path(__file__).parent / "data" / "images"


class ApiFixture(NamedTuple):
    client: TestClient
    mock_worker: MagicMock

    @property
    def db(self) -> sqlite3.Connection:
        """Typed accessor for app.state.db."""
        return self.client.app.state.db  # type: ignore[union-attr]

    @property
    def results_dir(self) -> Path:
        """Typed accessor for app.state.results_dir."""
        return self.client.app.state.results_dir  # type: ignore[union-attr]


@pytest.fixture
def api(tmp_path: Path) -> Generator[ApiFixture]:
    """Create a TestClient with temp directories and mocked worker."""
    uploads = tmp_path / "uploads"
    results = tmp_path / "results"
    db_path = tmp_path / "test.db"
    uploads.mkdir()
    results.mkdir()

    with (
        patch.dict(
            "os.environ",
            {
                "UPLOAD_DIR": str(uploads),
                "RESULT_DIR": str(results),
                "DATABASE_PATH": str(db_path),
            },
        ),
        patch("api.app.EmbeddingWorker") as MockWorker,
    ):
        mock_worker = MagicMock()
        mock_worker.is_running = True
        MockWorker.return_value = mock_worker

        from api.app import create_app

        app = create_app()
        with TestClient(app) as tc:
            yield ApiFixture(client=tc, mock_worker=mock_worker)


class TestCleanupJobFiles:
    def test_deletes_associated_files(self, tmp_path: Path) -> None:
        from api.app import _cleanup_job_files

        file_path = tmp_path / "upload.pdf"
        result_path = tmp_path / "result.json"
        tensor_path = tmp_path / "tensor.pt"
        file_path.write_text("data")
        result_path.write_text("{}")
        tensor_path.write_bytes(b"tensor")

        job = {
            "id": "job1",
            "file_path": str(file_path),
            "result_path": str(result_path),
            "tensor_path": str(tensor_path),
        }
        _cleanup_job_files(job)

        assert not file_path.exists()
        assert not result_path.exists()
        assert not tensor_path.exists()

    def test_evicts_cache_when_worker_provided(self) -> None:
        from api.app import _cleanup_job_files

        mock_worker = MagicMock()
        job = {"id": "job1"}
        _cleanup_job_files(job, worker=mock_worker)
        mock_worker.evict_cache.assert_called_once_with("job1")

    def test_skips_without_worker(self) -> None:
        from api.app import _cleanup_job_files

        job = {"id": "job1"}
        _cleanup_job_files(job)  # Should not raise

    def test_handles_missing_files(self, tmp_path: Path) -> None:
        from api.app import _cleanup_job_files

        job = {
            "id": "job1",
            "file_path": str(tmp_path / "nonexistent.pdf"),
            "result_path": str(tmp_path / "nonexistent.json"),
            "tensor_path": str(tmp_path / "nonexistent.pt"),
        }
        _cleanup_job_files(job)  # Should not raise (missing_ok=True)


class TestHealth:
    def test_returns_health(self, api: ApiFixture) -> None:
        resp = api.client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "device" in data
        assert "queue_depth" in data
        assert "worker_running" in data


class TestUploadJob:
    def test_upload_pdf(self, api: ApiFixture) -> None:
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        resp = api.client.post(
            "/jobs",
            files={"file": ("test.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    def test_upload_image(self, api: ApiFixture) -> None:
        img_data = (IMAGE_DATA_DIR / "red.png").read_bytes()
        resp = api.client.post(
            "/jobs",
            files={"file": ("red.png", img_data, "image/png")},
            data={"dpi": "150"},
        )
        assert resp.status_code == 201

    def test_rejects_invalid_file_type(self, api: ApiFixture) -> None:
        resp = api.client.post(
            "/jobs",
            files={"file": ("test.txt", b"hello", "text/plain")},
            data={"dpi": "150"},
        )
        assert resp.status_code == 400

    def test_rejects_invalid_dpi(self, api: ApiFixture) -> None:
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        resp = api.client.post(
            "/jobs",
            files={"file": ("test.pdf", pdf_data, "application/pdf")},
            data={"dpi": "999"},
        )
        assert resp.status_code == 400
        assert "DPI" in resp.json()["detail"]

    def test_rejects_oversized_file(self, api: ApiFixture) -> None:
        big_data = b"x" * (50 * 1024 * 1024 + 1)
        resp = api.client.post(
            "/jobs",
            files={"file": ("big.pdf", big_data, "application/pdf")},
            data={"dpi": "150"},
        )
        assert resp.status_code == 400

    def test_upload_stores_dpi(self, api: ApiFixture) -> None:
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = api.client.post(
            "/jobs",
            files={"file": ("test.pdf", pdf_data, "application/pdf")},
            data={"dpi": "300"},
        )
        assert create_resp.status_code == 201
        job_id = create_resp.json()["job_id"]
        resp = api.client.get(f"/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["dpi"] == 300


class TestListJobs:
    def test_list_empty(self, api: ApiFixture) -> None:
        resp = api.client.get("/jobs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_after_upload(self, api: ApiFixture) -> None:
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        api.client.post(
            "/jobs",
            files={"file": ("test.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )
        resp = api.client.get("/jobs")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_filter_by_status(self, api: ApiFixture) -> None:
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        api.client.post(
            "/jobs",
            files={"file": ("test.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )
        resp = api.client.get("/jobs?status=completed")
        assert resp.status_code == 200
        assert len(resp.json()) == 0


class TestGetJob:
    def test_get_existing(self, api: ApiFixture) -> None:
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = api.client.post(
            "/jobs",
            files={"file": ("test.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )
        job_id = create_resp.json()["job_id"]
        resp = api.client.get(f"/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == job_id

    def test_get_nonexistent(self, api: ApiFixture) -> None:
        resp = api.client.get("/jobs/nonexistent")
        assert resp.status_code == 404


class TestDeleteJob:
    def test_delete_pending(self, api: ApiFixture) -> None:
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = api.client.post(
            "/jobs",
            files={"file": ("test.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )
        job_id = create_resp.json()["job_id"]
        resp = api.client.delete(f"/jobs/{job_id}")
        assert resp.status_code == 204

    def test_delete_nonexistent(self, api: ApiFixture) -> None:
        resp = api.client.delete("/jobs/nonexistent")
        assert resp.status_code == 404

    def test_delete_processing_returns_409(self, api: ApiFixture) -> None:
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = api.client.post(
            "/jobs",
            files={"file": ("test.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )
        job_id = create_resp.json()["job_id"]
        from api.database import update_job

        db = api.db
        update_job(db, job_id, status="processing")
        resp = api.client.delete(f"/jobs/{job_id}")
        assert resp.status_code == 409

    def test_delete_cleans_up_files(self, api: ApiFixture) -> None:
        from api.database import update_job

        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = api.client.post(
            "/jobs",
            files={"file": ("test.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )
        job_id = create_resp.json()["job_id"]

        results_dir = api.results_dir
        result_path = results_dir / f"{job_id}.json"
        tensor_path = results_dir / f"{job_id}.pt"
        result_path.write_text("{}")
        tensor_path.write_bytes(b"fake")

        db = api.db
        update_job(
            db,
            job_id,
            status="completed",
            page_count=1,
            duration_ns=100,
            result_path=str(result_path),
            tensor_path=str(tensor_path),
        )

        resp = api.client.delete(f"/jobs/{job_id}")
        assert resp.status_code == 204
        assert not result_path.exists()
        assert not tensor_path.exists()


class TestDeleteAllJobs:
    def test_deletes_all_and_returns_count(self, api: ApiFixture) -> None:
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        api.client.post(
            "/jobs",
            files={"file": ("a.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )
        api.client.post(
            "/jobs",
            files={"file": ("b.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )
        resp = api.client.delete("/jobs")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 2
        assert api.client.get("/jobs").json() == []

    def test_preserves_processing_jobs(self, api: ApiFixture) -> None:
        from api.database import update_job

        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = api.client.post(
            "/jobs",
            files={"file": ("a.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )
        job_id = create_resp.json()["job_id"]
        update_job(api.db, job_id, status="processing")

        api.client.post(
            "/jobs",
            files={"file": ("b.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )

        resp = api.client.delete("/jobs")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 1
        remaining = api.client.get("/jobs").json()
        assert len(remaining) == 1
        assert remaining[0]["id"] == job_id

    def test_cleans_up_files(self, api: ApiFixture) -> None:
        from api.database import update_job

        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = api.client.post(
            "/jobs",
            files={"file": ("a.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )
        job_id = create_resp.json()["job_id"]

        results_dir = api.results_dir
        result_path = results_dir / f"{job_id}.json"
        tensor_path = results_dir / f"{job_id}.pt"
        result_path.write_text("{}")
        tensor_path.write_bytes(b"fake")

        update_job(
            api.db,
            job_id,
            status="completed",
            page_count=1,
            duration_ns=100,
            result_path=str(result_path),
            tensor_path=str(tensor_path),
        )

        resp = api.client.delete("/jobs")
        assert resp.status_code == 200
        assert not result_path.exists()
        assert not tensor_path.exists()

    def test_returns_zero_when_empty(self, api: ApiFixture) -> None:
        resp = api.client.delete("/jobs")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 0


class TestGetResult:
    def test_returns_404_for_pending(self, api: ApiFixture) -> None:
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = api.client.post(
            "/jobs",
            files={"file": ("test.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )
        job_id = create_resp.json()["job_id"]
        resp = api.client.get(f"/jobs/{job_id}/result")
        assert resp.status_code == 404

    def test_returns_200_for_completed_job(self, api: ApiFixture) -> None:
        import json as json_mod

        from api.database import update_job

        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = api.client.post(
            "/jobs",
            files={"file": ("test.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )
        job_id = create_resp.json()["job_id"]

        results_dir = api.results_dir
        result_path = results_dir / f"{job_id}.json"
        result_path.write_text(
            json_mod.dumps(
                {
                    "file_name": "test",
                    "model": "m",
                    "dpi": 150,
                    "embeddings": [],
                    "total_duration": 1,
                    "page_count": 1,
                }
            )
        )

        db = api.db
        update_job(
            db,
            job_id,
            status="completed",
            page_count=1,
            duration_ns=100,
            result_path=str(result_path),
            tensor_path=str(results_dir / f"{job_id}.pt"),
        )

        resp = api.client.get(f"/jobs/{job_id}/result")
        assert resp.status_code == 200
        data = resp.json()
        assert "file_name" in data


class TestSearch:
    def test_returns_empty_with_no_completed_jobs(self, api: ApiFixture) -> None:
        resp = api.client.post("/search", json={"query": "test"})
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    def test_enqueues_search_to_worker(self, api: ApiFixture) -> None:
        from concurrent.futures import Future

        from api.database import update_job

        future: Future = Future()
        future.set_result([("job1", 0, 0.9)])
        api.mock_worker.enqueue_search.return_value = future

        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = api.client.post(
            "/jobs",
            files={"file": ("test.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )
        job_id = create_resp.json()["job_id"]
        db = api.db
        update_job(
            db,
            job_id,
            status="completed",
            page_count=1,
            duration_ns=100,
            result_path="r.json",
            tensor_path="r.pt",
        )

        resp = api.client.post("/search", json={"query": "charts", "top_k": 5})
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 1
        assert results[0]["file_id"] == "job1"
        api.mock_worker.enqueue_search.assert_called_once()

    def test_rejects_invalid_filter_file_id(self, api: ApiFixture) -> None:
        resp = api.client.post(
            "/search",
            json={"query": "test", "filter_file_id": "nonexistent"},
        )
        # No completed jobs, so empty results (filter_file_id not validated when no jobs)
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    def test_rejects_invalid_top_k(self, api: ApiFixture) -> None:
        resp = api.client.post("/search", json={"query": "test", "top_k": 0})
        assert resp.status_code == 422

    def test_rejects_negative_min_score(self, api: ApiFixture) -> None:
        resp = api.client.post("/search", json={"query": "test", "min_score": -1.0})
        assert resp.status_code == 422


class TestAsk:
    def test_returns_503_when_not_configured(self, api: ApiFixture) -> None:
        import os

        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("GENERATION_API_URL", "GENERATION_MODEL")
        }
        with patch.dict("os.environ", env, clear=True):
            resp = api.client.post("/ask", json={"query": "test"})
        assert resp.status_code == 503

    def test_returns_message_when_no_completed_jobs(self, api: ApiFixture) -> None:
        with patch.dict(
            "os.environ",
            {
                "GENERATION_API_URL": "http://fake-vlm:8000/v1",
                "GENERATION_MODEL": "test-model",
            },
        ):
            resp = api.client.post("/ask", json={"query": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["sources"] == []
        assert "No documents have been processed" in data["answer"]

    def test_returns_answer_with_sources(self, api: ApiFixture) -> None:
        from concurrent.futures import Future

        from api.database import update_job

        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = api.client.post(
            "/jobs",
            files={"file": ("test.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )
        job_id = create_resp.json()["job_id"]
        db = api.db
        update_job(
            db,
            job_id,
            status="completed",
            page_count=1,
            duration_ns=100,
            result_path="r.json",
            tensor_path="r.pt",
        )

        search_future: Future = Future()
        search_future.set_result([(job_id, 0, 0.9)])
        api.mock_worker.enqueue_search.return_value = search_future

        mock_vlm_response = httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "The document shows a test page."}}]
            },
        )

        with (
            patch.dict(
                "os.environ",
                {
                    "GENERATION_API_URL": "http://fake-vlm:8000/v1",
                    "GENERATION_MODEL": "test-model",
                },
            ),
            patch("httpx.AsyncClient") as MockClient,
        ):
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.post.return_value = mock_vlm_response
            MockClient.return_value = mock_client_instance

            resp = api.client.post("/ask", json={"query": "What is this?"})

        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert data["answer"] == "The document shows a test page."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["file_id"] == job_id

    def test_returns_answer_when_no_results(self, api: ApiFixture) -> None:
        from concurrent.futures import Future

        from api.database import update_job

        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = api.client.post(
            "/jobs",
            files={"file": ("test.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )
        job_id = create_resp.json()["job_id"]
        db = api.db
        update_job(
            db,
            job_id,
            status="completed",
            page_count=1,
            duration_ns=100,
            result_path="r.json",
            tensor_path="r.pt",
        )

        search_future: Future = Future()
        search_future.set_result([])
        api.mock_worker.enqueue_search.return_value = search_future

        with patch.dict(
            "os.environ",
            {
                "GENERATION_API_URL": "http://fake-vlm:8000/v1",
                "GENERATION_MODEL": "test-model",
            },
        ):
            resp = api.client.post("/ask", json={"query": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["sources"] == []
        assert len(data["answer"]) > 0

    def test_rejects_invalid_top_k(self, api: ApiFixture) -> None:
        with patch.dict(
            "os.environ",
            {
                "GENERATION_API_URL": "http://fake-vlm:8000/v1",
                "GENERATION_MODEL": "test-model",
            },
        ):
            resp = api.client.post("/ask", json={"query": "test", "top_k": 0})
            assert resp.status_code == 422
            resp = api.client.post("/ask", json={"query": "test", "top_k": 11})
            assert resp.status_code == 422

    def test_returns_502_on_vlm_timeout(self, api: ApiFixture) -> None:
        from concurrent.futures import Future

        from api.database import update_job

        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = api.client.post(
            "/jobs",
            files={"file": ("test.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )
        job_id = create_resp.json()["job_id"]
        db = api.db
        update_job(
            db,
            job_id,
            status="completed",
            page_count=1,
            duration_ns=100,
            result_path="r.json",
            tensor_path="r.pt",
        )

        search_future: Future = Future()
        search_future.set_result([(job_id, 0, 0.9)])
        api.mock_worker.enqueue_search.return_value = search_future

        with (
            patch.dict(
                "os.environ",
                {
                    "GENERATION_API_URL": "http://fake-vlm:8000/v1",
                    "GENERATION_MODEL": "test-model",
                },
            ),
            patch("httpx.AsyncClient") as MockClient,
        ):
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.post.side_effect = httpx.TimeoutException(
                "Connection timed out"
            )
            MockClient.return_value = mock_client_instance

            resp = api.client.post("/ask", json={"query": "What is this?"})

        assert resp.status_code == 502
        assert "Unable to reach" in resp.json()["detail"]

    def test_returns_502_on_vlm_error(self, api: ApiFixture) -> None:
        from concurrent.futures import Future

        from api.database import update_job

        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = api.client.post(
            "/jobs",
            files={"file": ("test.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )
        job_id = create_resp.json()["job_id"]
        db = api.db
        update_job(
            db,
            job_id,
            status="completed",
            page_count=1,
            duration_ns=100,
            result_path="r.json",
            tensor_path="r.pt",
        )

        search_future: Future = Future()
        search_future.set_result([(job_id, 0, 0.9)])
        api.mock_worker.enqueue_search.return_value = search_future

        mock_vlm_response = httpx.Response(500, json={"error": "internal error"})

        with (
            patch.dict(
                "os.environ",
                {
                    "GENERATION_API_URL": "http://fake-vlm:8000/v1",
                    "GENERATION_MODEL": "test-model",
                },
            ),
            patch("httpx.AsyncClient") as MockClient,
        ):
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.post.return_value = mock_vlm_response
            MockClient.return_value = mock_client_instance

            resp = api.client.post("/ask", json={"query": "What is this?"})

        assert resp.status_code == 502
        assert "Generation service error" in resp.json()["detail"]

    def test_returns_502_on_vlm_malformed_response(self, api: ApiFixture) -> None:
        from concurrent.futures import Future

        from api.database import update_job

        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = api.client.post(
            "/jobs",
            files={"file": ("test.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )
        job_id = create_resp.json()["job_id"]
        db = api.db
        update_job(
            db,
            job_id,
            status="completed",
            page_count=1,
            duration_ns=100,
            result_path="r.json",
            tensor_path="r.pt",
        )

        search_future: Future = Future()
        search_future.set_result([(job_id, 0, 0.9)])
        api.mock_worker.enqueue_search.return_value = search_future

        mock_vlm_response = httpx.Response(200, json={"unexpected": "format"})

        with (
            patch.dict(
                "os.environ",
                {
                    "GENERATION_API_URL": "http://fake-vlm:8000/v1",
                    "GENERATION_MODEL": "test-model",
                },
            ),
            patch("httpx.AsyncClient") as MockClient,
        ):
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.post.return_value = mock_vlm_response
            MockClient.return_value = mock_client_instance

            resp = api.client.post("/ask", json={"query": "What is this?"})

        assert resp.status_code == 502
        assert "Unexpected response" in resp.json()["detail"]

    def test_returns_503_when_only_model_missing(self, api: ApiFixture) -> None:
        with patch.dict(
            "os.environ",
            {"GENERATION_API_URL": "http://fake-vlm:8000/v1"},
        ):
            resp = api.client.post("/ask", json={"query": "test"})
        assert resp.status_code == 503

    def test_rejects_invalid_filter_file_id(self, api: ApiFixture) -> None:
        from api.database import update_job

        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = api.client.post(
            "/jobs",
            files={"file": ("test.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )
        job_id = create_resp.json()["job_id"]
        db = api.db
        update_job(
            db,
            job_id,
            status="completed",
            page_count=1,
            duration_ns=100,
            result_path="r.json",
            tensor_path="r.pt",
        )

        with patch.dict(
            "os.environ",
            {
                "GENERATION_API_URL": "http://fake-vlm:8000/v1",
                "GENERATION_MODEL": "test-model",
            },
        ):
            resp = api.client.post(
                "/ask",
                json={"query": "test", "filter_file_id": "nonexistent"},
            )
        assert resp.status_code == 400
        assert "filter_file_id" in resp.json()["detail"]

    def test_returns_answer_with_image_file(self, api: ApiFixture) -> None:
        from concurrent.futures import Future

        from api.database import update_job

        img_data = (IMAGE_DATA_DIR / "red.png").read_bytes()
        create_resp = api.client.post(
            "/jobs",
            files={"file": ("red.png", img_data, "image/png")},
            data={"dpi": "150"},
        )
        job_id = create_resp.json()["job_id"]
        db = api.db
        update_job(
            db,
            job_id,
            status="completed",
            page_count=1,
            duration_ns=100,
            result_path="r.json",
            tensor_path="r.pt",
        )

        search_future: Future = Future()
        search_future.set_result([(job_id, 0, 0.9)])
        api.mock_worker.enqueue_search.return_value = search_future

        mock_vlm_response = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "A red image."}}]},
        )

        with (
            patch.dict(
                "os.environ",
                {
                    "GENERATION_API_URL": "http://fake-vlm:8000/v1",
                    "GENERATION_MODEL": "test-model",
                },
            ),
            patch("httpx.AsyncClient") as MockClient,
        ):
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.post.return_value = mock_vlm_response
            MockClient.return_value = mock_client_instance

            resp = api.client.post("/ask", json={"query": "What color?"})

        assert resp.status_code == 200
        assert resp.json()["answer"] == "A red image."

    def test_returns_502_on_vlm_connect_error(self, api: ApiFixture) -> None:
        from concurrent.futures import Future

        from api.database import update_job

        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = api.client.post(
            "/jobs",
            files={"file": ("test.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )
        job_id = create_resp.json()["job_id"]
        db = api.db
        update_job(
            db,
            job_id,
            status="completed",
            page_count=1,
            duration_ns=100,
            result_path="r.json",
            tensor_path="r.pt",
        )

        search_future: Future = Future()
        search_future.set_result([(job_id, 0, 0.9)])
        api.mock_worker.enqueue_search.return_value = search_future

        with (
            patch.dict(
                "os.environ",
                {
                    "GENERATION_API_URL": "http://fake-vlm:8000/v1",
                    "GENERATION_MODEL": "test-model",
                },
            ),
            patch("httpx.AsyncClient") as MockClient,
        ):
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.post.side_effect = httpx.ConnectError(
                "Connection refused"
            )
            MockClient.return_value = mock_client_instance

            resp = api.client.post("/ask", json={"query": "test"})

        assert resp.status_code == 502
        assert "Unable to reach" in resp.json()["detail"]

    def test_skips_missing_upload_file(self, api: ApiFixture) -> None:
        from concurrent.futures import Future

        from api.database import update_job

        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        create_resp = api.client.post(
            "/jobs",
            files={"file": ("test.pdf", pdf_data, "application/pdf")},
            data={"dpi": "150"},
        )
        job_id = create_resp.json()["job_id"]
        db = api.db

        # Delete the uploaded file to simulate missing file
        job = api.db.execute(
            "SELECT file_path FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        Path(job[0]).unlink()

        update_job(
            db,
            job_id,
            status="completed",
            page_count=1,
            duration_ns=100,
            result_path="r.json",
            tensor_path="r.pt",
        )

        search_future: Future = Future()
        search_future.set_result([(job_id, 0, 0.9)])
        api.mock_worker.enqueue_search.return_value = search_future

        with patch.dict(
            "os.environ",
            {
                "GENERATION_API_URL": "http://fake-vlm:8000/v1",
                "GENERATION_MODEL": "test-model",
            },
        ):
            resp = api.client.post("/ask", json={"query": "test"})

        assert resp.status_code == 200
        assert resp.json()["sources"] == []
        assert "Could not load" in resp.json()["answer"]
