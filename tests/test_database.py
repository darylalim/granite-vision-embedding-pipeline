import sqlite3
from pathlib import Path

import pytest

from api.database import (
    create_job,
    delete_job,
    get_connection,
    get_job,
    init_db,
    list_jobs,
    next_pending_job,
    reset_processing_jobs,
    update_job,
)


@pytest.fixture
def db(tmp_path: Path) -> sqlite3.Connection:
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    return conn


class TestInitDb:
    def test_creates_jobs_table(self, db: sqlite3.Connection) -> None:
        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'"
        )
        assert cursor.fetchone() is not None

    def test_enables_wal_mode(self, db: sqlite3.Connection) -> None:
        cursor = db.execute("PRAGMA journal_mode")
        assert cursor.fetchone()[0] == "wal"

    def test_idempotent(self, db: sqlite3.Connection) -> None:
        init_db(db)
        cursor = db.execute("SELECT count(*) FROM jobs")
        assert cursor.fetchone()[0] == 0


class TestCreateJob:
    def test_inserts_pending_job(self, db: sqlite3.Connection) -> None:
        job_id = create_job(
            db,
            file_name="test.pdf",
            file_stem="test",
            file_path="uploads/abc.pdf",
            file_type="pdf",
            dpi=150,
        )
        job = get_job(db, job_id)
        assert job is not None
        assert job["status"] == "pending"
        assert job["file_name"] == "test.pdf"
        assert job["file_stem"] == "test"
        assert job["file_type"] == "pdf"
        assert job["dpi"] == 150

    def test_generates_unique_ids(self, db: sqlite3.Connection) -> None:
        id1 = create_job(
            db,
            file_name="a.pdf",
            file_stem="a",
            file_path="uploads/a.pdf",
            file_type="pdf",
            dpi=150,
        )
        id2 = create_job(
            db,
            file_name="b.pdf",
            file_stem="b",
            file_path="uploads/b.pdf",
            file_type="pdf",
            dpi=150,
        )
        assert id1 != id2

    def test_accepts_optional_job_id(self, db: sqlite3.Connection) -> None:
        job_id = create_job(
            db,
            file_name="a.pdf",
            file_stem="a",
            file_path="uploads/a.pdf",
            file_type="pdf",
            dpi=150,
            job_id="custom123",
        )
        assert job_id == "custom123"
        job = get_job(db, "custom123")
        assert job is not None


class TestGetJob:
    def test_returns_none_for_missing(self, db: sqlite3.Connection) -> None:
        assert get_job(db, "nonexistent") is None


class TestListJobs:
    def test_returns_all_jobs(self, db: sqlite3.Connection) -> None:
        create_job(
            db,
            file_name="a.pdf",
            file_stem="a",
            file_path="uploads/a.pdf",
            file_type="pdf",
            dpi=150,
        )
        create_job(
            db,
            file_name="b.pdf",
            file_stem="b",
            file_path="uploads/b.pdf",
            file_type="pdf",
            dpi=150,
        )
        jobs = list_jobs(db)
        assert len(jobs) == 2

    def test_filters_by_status(self, db: sqlite3.Connection) -> None:
        job_id = create_job(
            db,
            file_name="a.pdf",
            file_stem="a",
            file_path="uploads/a.pdf",
            file_type="pdf",
            dpi=150,
        )
        update_job(
            db,
            job_id,
            status="completed",
            page_count=1,
            duration_ns=100,
            result_path="results/a.json",
            tensor_path="results/a.pt",
        )
        create_job(
            db,
            file_name="b.pdf",
            file_stem="b",
            file_path="uploads/b.pdf",
            file_type="pdf",
            dpi=150,
        )
        pending = list_jobs(db, status="pending")
        assert len(pending) == 1
        assert pending[0]["file_name"] == "b.pdf"

    def test_ordered_by_created_at(self, db: sqlite3.Connection) -> None:
        create_job(
            db,
            file_name="first.pdf",
            file_stem="first",
            file_path="uploads/first.pdf",
            file_type="pdf",
            dpi=150,
        )
        create_job(
            db,
            file_name="second.pdf",
            file_stem="second",
            file_path="uploads/second.pdf",
            file_type="pdf",
            dpi=150,
        )
        jobs = list_jobs(db)
        assert jobs[0]["file_name"] == "first.pdf"

    def test_returns_empty_for_no_jobs(self, db: sqlite3.Connection) -> None:
        assert list_jobs(db) == []


class TestUpdateJob:
    def test_updates_to_completed(self, db: sqlite3.Connection) -> None:
        job_id = create_job(
            db,
            file_name="a.pdf",
            file_stem="a",
            file_path="uploads/a.pdf",
            file_type="pdf",
            dpi=150,
        )
        update_job(
            db,
            job_id,
            status="completed",
            page_count=3,
            duration_ns=5000,
            result_path="results/a.json",
            tensor_path="results/a.pt",
        )
        job = get_job(db, job_id)
        assert job["status"] == "completed"
        assert job["page_count"] == 3
        assert job["duration_ns"] == 5000

    def test_updates_to_failed(self, db: sqlite3.Connection) -> None:
        job_id = create_job(
            db,
            file_name="a.pdf",
            file_stem="a",
            file_path="uploads/a.pdf",
            file_type="pdf",
            dpi=150,
        )
        update_job(db, job_id, status="failed", error="corrupt file")
        job = get_job(db, job_id)
        assert job["status"] == "failed"
        assert job["error"] == "corrupt file"

    def test_updates_to_processing(self, db: sqlite3.Connection) -> None:
        job_id = create_job(
            db,
            file_name="a.pdf",
            file_stem="a",
            file_path="uploads/a.pdf",
            file_type="pdf",
            dpi=150,
        )
        update_job(db, job_id, status="processing")
        job = get_job(db, job_id)
        assert job["status"] == "processing"


class TestDeleteJob:
    def test_removes_job(self, db: sqlite3.Connection) -> None:
        job_id = create_job(
            db,
            file_name="a.pdf",
            file_stem="a",
            file_path="uploads/a.pdf",
            file_type="pdf",
            dpi=150,
        )
        delete_job(db, job_id)
        assert get_job(db, job_id) is None

    def test_delete_nonexistent_is_noop(self, db: sqlite3.Connection) -> None:
        delete_job(db, "nonexistent")


class TestResetProcessingJobs:
    def test_resets_processing_to_pending(self, db: sqlite3.Connection) -> None:
        job_id = create_job(
            db,
            file_name="a.pdf",
            file_stem="a",
            file_path="uploads/a.pdf",
            file_type="pdf",
            dpi=150,
        )
        update_job(db, job_id, status="processing")
        count = reset_processing_jobs(db)
        assert count == 1
        job = get_job(db, job_id)
        assert job["status"] == "pending"


class TestNextPendingJob:
    def test_returns_oldest_pending(self, db: sqlite3.Connection) -> None:
        create_job(
            db,
            file_name="first.pdf",
            file_stem="first",
            file_path="uploads/first.pdf",
            file_type="pdf",
            dpi=150,
        )
        create_job(
            db,
            file_name="second.pdf",
            file_stem="second",
            file_path="uploads/second.pdf",
            file_type="pdf",
            dpi=150,
        )
        job = next_pending_job(db)
        assert job is not None
        assert job["file_name"] == "first.pdf"

    def test_returns_none_when_empty(self, db: sqlite3.Connection) -> None:
        assert next_pending_job(db) is None
