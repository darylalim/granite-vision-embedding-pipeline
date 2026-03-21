import asyncio
import os
import sqlite3
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from api.database import (
    create_job,
    delete_job,
    get_job,
    init_db,
    list_jobs,
)
from api.models import (
    HealthResponse,
    JobCreateResponse,
    JobResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from api.worker import EmbeddingWorker
from core.constants import IMAGE_EXTENSIONS, MAX_UPLOAD_BYTES
from core.embedding import get_device

_db = None
_worker: EmbeddingWorker | None = None


def _get_db():
    global _db
    if _db is None:
        db_path = os.environ.get("DATABASE_PATH", "data/jobs.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        _db = sqlite3.connect(str(db_path), check_same_thread=False)
        _db.row_factory = sqlite3.Row
        _db.execute("PRAGMA journal_mode=WAL")
        _db.execute("PRAGMA busy_timeout=5000")
        init_db(_db)
    return _db


def _get_dirs() -> tuple[Path, Path]:
    uploads = Path(os.environ.get("UPLOAD_DIR", "uploads"))
    results = Path(os.environ.get("RESULT_DIR", "results"))
    uploads.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    return uploads, results


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _worker
        db = _get_db()
        uploads_dir, results_dir = _get_dirs()
        _worker = EmbeddingWorker(db, uploads_dir=uploads_dir, results_dir=results_dir)
        _worker.start()
        yield
        _worker.stop()
        _worker = None

    app = FastAPI(title="Granite Vision Embedding API", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    async def health():
        db = _get_db()
        pending = list_jobs(db, status="pending")
        return HealthResponse(
            device=get_device(),
            queue_depth=len(pending),
            worker_running=_worker.is_running if _worker else False,
        )

    @app.post("/jobs", response_model=JobCreateResponse, status_code=201)
    async def upload_job(file: UploadFile = File(...), dpi: int = Form(150)):
        ext = file.filename.rsplit(".", 1)[-1].lower() if file.filename else ""
        valid_types = IMAGE_EXTENSIONS | {"pdf"}
        if ext not in valid_types:
            raise HTTPException(400, detail=f"Invalid file type: .{ext}")

        content = await file.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(400, detail="File exceeds 50 MB limit")

        file_stem = file.filename.rsplit(".", 1)[0] if file.filename else "upload"
        file_type = "image" if ext in IMAGE_EXTENSIONS else "pdf"

        uploads_dir, _ = _get_dirs()
        db = _get_db()

        # Generate job ID and save file before creating the DB row
        job_id = uuid.uuid4().hex
        upload_path = uploads_dir / f"{job_id}.{ext}"
        upload_path.write_bytes(content)

        create_job(
            db,
            file_name=file.filename or "unknown",
            file_stem=file_stem,
            file_path=str(upload_path),
            file_type=file_type,
            dpi=dpi,
            job_id=job_id,
        )

        return JobCreateResponse(job_id=job_id, status="pending")

    @app.get("/jobs", response_model=list[JobResponse])
    async def list_all_jobs(status: str | None = None):
        db = _get_db()
        jobs = list_jobs(db, status=status)
        return [JobResponse(**{k: v for k, v in j.items() if k in JobResponse.model_fields}) for j in jobs]

    @app.get("/jobs/{job_id}", response_model=JobResponse)
    async def get_single_job(job_id: str):
        db = _get_db()
        job = get_job(db, job_id)
        if not job:
            raise HTTPException(404, detail="Job not found")
        return JobResponse(**{k: v for k, v in job.items() if k in JobResponse.model_fields})

    @app.delete("/jobs/{job_id}", status_code=204)
    async def delete_single_job(job_id: str):
        db = _get_db()
        job = get_job(db, job_id)
        if not job:
            raise HTTPException(404, detail="Job not found")
        if job["status"] == "processing":
            raise HTTPException(409, detail="Cannot delete a processing job")

        # Clean up files
        for path_key in ("file_path", "result_path", "tensor_path"):
            path_str = job.get(path_key)
            if path_str:
                Path(path_str).unlink(missing_ok=True)

        if _worker:
            _worker.evict_cache(job_id)
        delete_job(db, job_id)

    @app.get("/jobs/{job_id}/result")
    async def get_result(job_id: str):
        db = _get_db()
        job = get_job(db, job_id)
        if not job or not job.get("result_path"):
            raise HTTPException(404, detail="Result not available")
        result_path = Path(job["result_path"])
        if not result_path.exists():
            raise HTTPException(404, detail="Result file not found")
        return FileResponse(result_path, media_type="application/json")

    @app.post("/search", response_model=SearchResponse)
    async def search_embeddings(req: SearchRequest):
        if not _worker:
            raise HTTPException(503, detail="Worker not running")

        db = _get_db()
        completed_jobs = list_jobs(db, status="completed")
        job_ids = [j["id"] for j in completed_jobs]

        if not job_ids:
            return SearchResponse(results=[])

        params = {
            "query": req.query,
            "top_k": req.top_k,
            "min_score": req.min_score,
            "filter_file_id": req.filter_file_id,
            "job_ids": job_ids,
        }
        future = _worker.enqueue_search(params)
        results = await asyncio.wrap_future(future)
        return SearchResponse(
            results=[
                SearchResult(file_id=fid, page_index=pidx, score=score)
                for fid, pidx, score in results
            ]
        )

    return app
