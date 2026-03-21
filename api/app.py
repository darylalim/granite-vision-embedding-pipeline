import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from api.database import (
    create_job,
    delete_job,
    get_connection,
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
from core.constants import DPI_OPTIONS, IMAGE_EXTENSIONS, MAX_UPLOAD_BYTES
from core.embedding import get_device

VALID_DPI = set(DPI_OPTIONS.values())
VALID_EXTENSIONS = IMAGE_EXTENSIONS | {"pdf"}


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db_path = os.environ.get("DATABASE_PATH", "data/jobs.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        db = get_connection(db_path, check_same_thread=False)
        init_db(db)
        app.state.db = db

        uploads = Path(os.environ.get("UPLOAD_DIR", "uploads"))
        results = Path(os.environ.get("RESULT_DIR", "results"))
        uploads.mkdir(parents=True, exist_ok=True)
        results.mkdir(parents=True, exist_ok=True)
        app.state.uploads_dir = uploads
        app.state.results_dir = results

        worker = EmbeddingWorker(db, uploads_dir=uploads, results_dir=results)
        worker.start()
        app.state.worker = worker
        yield
        worker.stop()

    app = FastAPI(title="Granite Vision Embedding API", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    async def health():
        db = app.state.db
        pending = list_jobs(db, status="pending")
        worker = app.state.worker
        return HealthResponse(
            device=get_device(),
            queue_depth=len(pending),
            worker_running=worker.is_running if worker else False,
        )

    @app.post("/jobs", response_model=JobCreateResponse, status_code=201)
    async def upload_job(file: UploadFile = File(...), dpi: int = Form(150)):
        ext = file.filename.rsplit(".", 1)[-1].lower() if file.filename else ""
        if ext not in VALID_EXTENSIONS:
            raise HTTPException(400, detail=f"Invalid file type: .{ext}")

        if dpi not in VALID_DPI:
            raise HTTPException(
                400, detail=f"Invalid DPI: {dpi}. Must be one of {sorted(VALID_DPI)}"
            )

        content = await file.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(400, detail="File exceeds 50 MB limit")

        file_stem = file.filename.rsplit(".", 1)[0] if file.filename else "upload"
        file_type = "image" if ext in IMAGE_EXTENSIONS else "pdf"

        db = app.state.db
        job_id = uuid.uuid4().hex
        upload_path = app.state.uploads_dir / f"{job_id}.{ext}"
        upload_path.write_bytes(content)

        try:
            create_job(
                db,
                file_name=file.filename or "unknown",
                file_stem=file_stem,
                file_path=str(upload_path),
                file_type=file_type,
                dpi=dpi,
                job_id=job_id,
            )
        except Exception:
            upload_path.unlink(missing_ok=True)
            raise

        return JobCreateResponse(job_id=job_id, status="pending")

    @app.get("/jobs", response_model=list[JobResponse])
    async def list_all_jobs(status: str | None = None):
        db = app.state.db
        jobs = list_jobs(db, status=status)
        return [
            JobResponse(**{k: v for k, v in j.items() if k in JobResponse.model_fields})
            for j in jobs
        ]

    @app.get("/jobs/{job_id}", response_model=JobResponse)
    async def get_single_job(job_id: str):
        db = app.state.db
        job = get_job(db, job_id)
        if not job:
            raise HTTPException(404, detail="Job not found")
        return JobResponse(
            **{k: v for k, v in job.items() if k in JobResponse.model_fields}
        )

    @app.delete("/jobs/{job_id}", status_code=204)
    async def delete_single_job(job_id: str):
        db = app.state.db
        job = get_job(db, job_id)
        if not job:
            raise HTTPException(404, detail="Job not found")
        if job["status"] == "processing":
            raise HTTPException(409, detail="Cannot delete a processing job")

        for path_key in ("file_path", "result_path", "tensor_path"):
            path_str = job.get(path_key)
            if path_str:
                Path(path_str).unlink(missing_ok=True)

        worker = app.state.worker
        if worker:
            worker.evict_cache(job_id)
        delete_job(db, job_id)

    @app.get("/jobs/{job_id}/result")
    async def get_result(job_id: str):
        db = app.state.db
        job = get_job(db, job_id)
        if not job or not job.get("result_path"):
            raise HTTPException(404, detail="Result not available")
        result_path = Path(job["result_path"])
        if not result_path.exists():
            raise HTTPException(404, detail="Result file not found")
        return FileResponse(result_path, media_type="application/json")

    @app.post("/search", response_model=SearchResponse)
    async def search_embeddings(req: SearchRequest):
        worker = app.state.worker
        if not worker:
            raise HTTPException(503, detail="Worker not running")

        db = app.state.db
        completed_jobs = list_jobs(db, status="completed")
        job_ids = [j["id"] for j in completed_jobs]

        if not job_ids:
            return SearchResponse(results=[])

        if req.filter_file_id and req.filter_file_id not in job_ids:
            raise HTTPException(
                400,
                detail=f"filter_file_id '{req.filter_file_id}' is not a completed job",
            )

        params = {
            "query": req.query,
            "top_k": req.top_k,
            "min_score": req.min_score,
            "filter_file_id": req.filter_file_id,
            "job_ids": job_ids,
        }
        future = worker.enqueue_search(params)
        results = await asyncio.wrap_future(future)
        return SearchResponse(
            results=[
                SearchResult(file_id=fid, page_index=pidx, score=score)
                for fid, pidx, score in results
            ]
        )

    return app
