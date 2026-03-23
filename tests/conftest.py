import sqlite3
from pathlib import Path

import pytest

from api.database import get_connection, init_db


@pytest.fixture
def db(tmp_path: Path) -> sqlite3.Connection:
    """Create a fresh test database."""
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    return conn


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Create temporary uploads and results directories."""
    uploads = tmp_path / "uploads"
    results = tmp_path / "results"
    uploads.mkdir()
    results.mkdir()
    return uploads, results
