import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from api.database import create_job, get_connection, get_job, init_db, update_job
from api.worker import EmbeddingWorker

PDF_DATA_DIR = Path(__file__).parent / "data" / "pdf"
IMAGE_DATA_DIR = Path(__file__).parent / "data" / "images"


def _make_mock_model(return_value: torch.Tensor) -> MagicMock:
    model = MagicMock()
    model.device = "cpu"
    model.return_value = return_value
    return model


def _make_mock_processor() -> MagicMock:
    mock_val = MagicMock(spec=torch.Tensor)
    mock_val.to.return_value = mock_val
    processor = MagicMock()
    processor.process_images.return_value = {"pixel_values": mock_val}
    processor.process_queries.return_value = {"input_ids": mock_val}
    return processor


@pytest.fixture
def db(tmp_path: Path) -> sqlite3.Connection:
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    return conn


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    uploads = tmp_path / "uploads"
    results = tmp_path / "results"
    uploads.mkdir()
    results.mkdir()
    return uploads, results


class TestProcessJob:
    @patch("api.worker.load_model")
    def test_processes_pdf_to_completed(
        self, mock_load: MagicMock, db: sqlite3.Connection, dirs: tuple[Path, Path]
    ) -> None:
        uploads, results = dirs
        mock_model = _make_mock_model(torch.randn(1, 4, 128))
        mock_processor = _make_mock_processor()
        mock_load.return_value = (mock_model, mock_processor)

        # Copy PDF fixture to uploads
        pdf_data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        upload_path = uploads / "test.pdf"
        upload_path.write_bytes(pdf_data)

        job_id = create_job(
            db,
            file_name="test.pdf",
            file_stem="test",
            file_path=str(upload_path),
            file_type="pdf",
            dpi=150,
        )

        worker = EmbeddingWorker(db, uploads_dir=uploads, results_dir=results)
        worker.process_job(get_job(db, job_id))

        job = get_job(db, job_id)
        assert job["status"] == "completed"
        assert job["page_count"] == 1
        assert job["duration_ns"] > 0
        assert Path(job["result_path"]).exists()
        assert Path(job["tensor_path"]).exists()

    @patch("api.worker.load_model")
    def test_processes_image_to_completed(
        self, mock_load: MagicMock, db: sqlite3.Connection, dirs: tuple[Path, Path]
    ) -> None:
        uploads, results = dirs
        mock_model = _make_mock_model(torch.randn(1, 4, 128))
        mock_processor = _make_mock_processor()
        mock_load.return_value = (mock_model, mock_processor)

        # Copy image fixture to uploads
        img_data = (IMAGE_DATA_DIR / "red.png").read_bytes()
        upload_path = uploads / "red.png"
        upload_path.write_bytes(img_data)

        job_id = create_job(
            db,
            file_name="red.png",
            file_stem="red",
            file_path=str(upload_path),
            file_type="image",
            dpi=150,
        )

        worker = EmbeddingWorker(db, uploads_dir=uploads, results_dir=results)
        worker.process_job(get_job(db, job_id))

        job = get_job(db, job_id)
        assert job["status"] == "completed"
        assert job["page_count"] == 1

    @patch("api.worker.load_model")
    def test_marks_corrupt_file_as_failed(
        self, mock_load: MagicMock, db: sqlite3.Connection, dirs: tuple[Path, Path]
    ) -> None:
        uploads, results = dirs
        mock_model = _make_mock_model(torch.randn(1, 4, 128))
        mock_processor = _make_mock_processor()
        mock_load.return_value = (mock_model, mock_processor)

        upload_path = uploads / "bad.png"
        upload_path.write_bytes(b"not an image")

        job_id = create_job(
            db,
            file_name="bad.png",
            file_stem="bad",
            file_path=str(upload_path),
            file_type="image",
            dpi=150,
        )

        worker = EmbeddingWorker(db, uploads_dir=uploads, results_dir=results)
        worker.process_job(get_job(db, job_id))

        job = get_job(db, job_id)
        assert job["status"] == "failed"
        assert job["error"] is not None


class TestStartupRecovery:
    @patch("api.worker.load_model")
    def test_resets_processing_to_pending(
        self, mock_load: MagicMock, db: sqlite3.Connection, dirs: tuple[Path, Path]
    ) -> None:
        uploads, results = dirs
        mock_model = _make_mock_model(torch.randn(1, 4, 128))
        mock_processor = _make_mock_processor()
        mock_load.return_value = (mock_model, mock_processor)

        job_id = create_job(
            db,
            file_name="a.pdf",
            file_stem="a",
            file_path="uploads/a.pdf",
            file_type="pdf",
            dpi=150,
        )
        update_job(db, job_id, status="processing")

        worker = EmbeddingWorker(db, uploads_dir=uploads, results_dir=results)
        worker.startup_recovery()

        job = get_job(db, job_id)
        assert job["status"] == "pending"


class TestTensorCache:
    @patch("api.worker.load_model")
    def test_cache_evicts_lru(
        self, mock_load: MagicMock, db: sqlite3.Connection, dirs: tuple[Path, Path]
    ) -> None:
        uploads, results = dirs
        mock_model = _make_mock_model(torch.randn(1, 4, 128))
        mock_processor = _make_mock_processor()
        mock_load.return_value = (mock_model, mock_processor)

        worker = EmbeddingWorker(
            db, uploads_dir=uploads, results_dir=results, cache_max=2
        )
        worker._tensor_cache["a"] = torch.randn(1, 4, 128)
        worker._tensor_cache["b"] = torch.randn(1, 4, 128)

        # Adding "c" should evict "a" (LRU)
        worker._tensor_cache["c"] = torch.randn(1, 4, 128)
        worker._enforce_cache_limit()

        assert "a" not in worker._tensor_cache
        assert "b" in worker._tensor_cache
        assert "c" in worker._tensor_cache

    @patch("api.worker.load_model")
    def test_cache_remove_on_delete(
        self, mock_load: MagicMock, db: sqlite3.Connection, dirs: tuple[Path, Path]
    ) -> None:
        uploads, results = dirs
        mock_model = _make_mock_model(torch.randn(1, 4, 128))
        mock_processor = _make_mock_processor()
        mock_load.return_value = (mock_model, mock_processor)

        worker = EmbeddingWorker(db, uploads_dir=uploads, results_dir=results)
        worker._tensor_cache["a"] = torch.randn(1, 4, 128)
        worker.evict_cache("a")
        assert "a" not in worker._tensor_cache


class TestSearchDispatch:
    @patch("api.worker.load_model")
    def test_enqueue_and_drain_returns_results(
        self, mock_load: MagicMock, db: sqlite3.Connection, dirs: tuple[Path, Path]
    ) -> None:
        uploads, results = dirs
        mock_model = _make_mock_model(torch.tensor([[0.1, 0.2]]))
        mock_processor = _make_mock_processor()
        mock_processor.score.return_value = torch.tensor([[0.8]])
        mock_load.return_value = (mock_model, mock_processor)

        # Create a completed job with a .pt file
        job_id = create_job(
            db,
            file_name="a.pdf",
            file_stem="a",
            file_path="uploads/a.pdf",
            file_type="pdf",
            dpi=150,
        )
        tensor = torch.randn(1, 4, 128)
        tensor_path = results / f"{job_id}.pt"
        torch.save(tensor, tensor_path)
        update_job(
            db,
            job_id,
            status="completed",
            page_count=1,
            duration_ns=100,
            result_path=str(results / f"{job_id}.json"),
            tensor_path=str(tensor_path),
        )

        worker = EmbeddingWorker(db, uploads_dir=uploads, results_dir=results)
        future = worker.enqueue_search(
            {
                "query": "test",
                "top_k": 5,
                "min_score": 0.0,
                "job_ids": [job_id],
            }
        )

        # Drain the queue manually (simulates worker loop)
        worker._drain_search_queue()

        result = future.result(timeout=5)
        assert len(result) == 1
        assert result[0][0] == job_id

    @patch("api.worker.load_model")
    def test_enqueue_returns_empty_for_no_jobs(
        self, mock_load: MagicMock, db: sqlite3.Connection, dirs: tuple[Path, Path]
    ) -> None:
        uploads, results = dirs
        mock_model = _make_mock_model(torch.randn(1, 4, 128))
        mock_processor = _make_mock_processor()
        mock_load.return_value = (mock_model, mock_processor)

        worker = EmbeddingWorker(db, uploads_dir=uploads, results_dir=results)
        future = worker.enqueue_search(
            {
                "query": "test",
                "job_ids": [],
            }
        )
        worker._drain_search_queue()

        result = future.result(timeout=5)
        assert result == []
