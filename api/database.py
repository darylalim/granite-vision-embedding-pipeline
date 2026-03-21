import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def get_connection(
    db_path: Path | str, *, check_same_thread: bool = True
) -> sqlite3.Connection:
    """Create a SQLite connection with WAL mode and busy timeout."""
    conn = sqlite3.connect(str(db_path), check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create the jobs table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id            TEXT PRIMARY KEY,
            status        TEXT NOT NULL,
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL,
            file_name     TEXT NOT NULL,
            file_stem     TEXT NOT NULL,
            file_path     TEXT NOT NULL,
            file_type     TEXT NOT NULL,
            dpi           INTEGER NOT NULL,
            page_count    INTEGER,
            duration_ns   INTEGER,
            result_path   TEXT,
            tensor_path   TEXT,
            error         TEXT
        )
    """)
    conn.commit()


def create_job(
    conn: sqlite3.Connection,
    *,
    file_name: str,
    file_stem: str,
    file_path: str,
    file_type: str,
    dpi: int,
    job_id: str | None = None,
) -> str:
    """Insert a new pending job and return its ID."""
    if job_id is None:
        job_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO jobs (id, status, created_at, updated_at, file_name, file_stem, file_path, file_type, dpi)
           VALUES (?, 'pending', ?, ?, ?, ?, ?, ?, ?)""",
        (job_id, now, now, file_name, file_stem, file_path, file_type, dpi),
    )
    conn.commit()
    return job_id


def get_job(conn: sqlite3.Connection, job_id: str) -> dict | None:
    """Fetch a single job by ID, or None if not found."""
    cursor = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
    row = cursor.fetchone()
    return dict(row) if row else None


def list_jobs(conn: sqlite3.Connection, status: str | None = None) -> list[dict]:
    """List jobs ordered by created_at, optionally filtered by status."""
    if status:
        cursor = conn.execute(
            "SELECT * FROM jobs WHERE status = ? ORDER BY created_at", (status,)
        )
    else:
        cursor = conn.execute("SELECT * FROM jobs ORDER BY created_at")
    return [dict(row) for row in cursor.fetchall()]


def update_job(
    conn: sqlite3.Connection,
    job_id: str,
    *,
    status: str,
    page_count: int | None = None,
    duration_ns: int | None = None,
    result_path: str | None = None,
    tensor_path: str | None = None,
    error: str | None = None,
) -> None:
    """Update a job's status and optional metadata fields.

    Uses COALESCE so None values preserve existing data. This means fields
    cannot be reset to NULL once set — acceptable for the one-way lifecycle
    (pending -> processing -> completed/failed).
    """
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """UPDATE jobs SET status = ?, updated_at = ?, page_count = COALESCE(?, page_count),
           duration_ns = COALESCE(?, duration_ns), result_path = COALESCE(?, result_path),
           tensor_path = COALESCE(?, tensor_path), error = COALESCE(?, error)
           WHERE id = ?""",
        (status, now, page_count, duration_ns, result_path, tensor_path, error, job_id),
    )
    conn.commit()


def delete_job(conn: sqlite3.Connection, job_id: str) -> None:
    """Delete a job by ID."""
    conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    conn.commit()


def reset_processing_jobs(conn: sqlite3.Connection) -> int:
    """Reset any jobs stuck in 'processing' back to 'pending'. Returns count reset."""
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "UPDATE jobs SET status = 'pending', updated_at = ? WHERE status = 'processing'",
        (now,),
    )
    conn.commit()
    return cursor.rowcount


def next_pending_job(conn: sqlite3.Connection) -> dict | None:
    """Fetch the oldest pending job, or None."""
    cursor = conn.execute(
        "SELECT * FROM jobs WHERE status = 'pending' ORDER BY created_at LIMIT 1"
    )
    row = cursor.fetchone()
    return dict(row) if row else None
