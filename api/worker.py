import json
import queue
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from pathlib import Path

import torch

from api.database import (
    get_job,
    next_pending_job,
    reset_processing_jobs,
    update_job,
)
from core.constants import MODEL_ID
from core.embedding import embed, get_device, load_image, load_model
from core.rendering import render_pages
from core.search import filter_results, search_multi


class EmbeddingWorker:
    def __init__(
        self,
        db,
        *,
        uploads_dir: Path,
        results_dir: Path,
        cache_max: int = 500,
    ) -> None:
        self._db = db
        self._uploads_dir = uploads_dir
        self._results_dir = results_dir
        self._cache_max = cache_max
        self._tensor_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._search_queue: queue.Queue[tuple[dict, Future]] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        device = get_device()
        self._model, self._processor = load_model(device)

    def startup_recovery(self) -> None:
        """Reset any jobs stuck in 'processing' to 'pending'."""
        reset_processing_jobs(self._db)

    def start(self) -> None:
        """Start the worker thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the worker to stop and wait for it."""
        self._stop_event.set()
        if self._thread:
            self._thread.join()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def enqueue_search(self, params: dict) -> Future:
        """Enqueue a search request and return a Future for the result."""
        future: Future = Future()
        self._search_queue.put((params, future))
        return future

    def evict_cache(self, job_id: str) -> None:
        """Remove a tensor from the cache. Thread-safe."""
        with self._cache_lock:
            self._tensor_cache.pop(job_id, None)

    def _enforce_cache_limit(self) -> None:
        """Evict LRU entries until cache is within limit. Caller must hold _cache_lock."""
        while len(self._tensor_cache) > self._cache_max:
            self._tensor_cache.popitem(last=False)

    def _run(self) -> None:
        """Main worker loop."""
        self.startup_recovery()
        while not self._stop_event.is_set():
            self._drain_search_queue()
            job = next_pending_job(self._db)
            if job:
                self.process_job(job)
            else:
                self._stop_event.wait(timeout=1.0)

    def _drain_search_queue(self) -> None:
        """Process all pending search requests."""
        while not self._search_queue.empty():
            try:
                params, future = self._search_queue.get_nowait()
            except queue.Empty:
                break
            try:
                result = self._execute_search(params)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

    def _execute_search(self, params: dict) -> list[tuple[str, int, float]]:
        """Run a search query using cached tensors."""
        query = params["query"]
        top_k = params.get("top_k", 5)
        min_score = params.get("min_score", 0.0)
        filter_file_id = params.get("filter_file_id")
        job_ids = params.get("job_ids", [])

        embeddings: dict[str, torch.Tensor] = {}
        for job_id in job_ids:
            tensor = self._get_or_load_tensor(job_id)
            if tensor is not None:
                embeddings[job_id] = tensor

        if not embeddings:
            return []

        raw = search_multi(
            query,
            self._model,
            self._processor,
            embeddings,
            filter_file_id=filter_file_id,
        )
        return filter_results(raw, top_k=top_k, min_score=min_score)

    def _get_or_load_tensor(self, job_id: str) -> torch.Tensor | None:
        """Load tensor from cache or .pt file."""
        with self._cache_lock:
            if job_id in self._tensor_cache:
                self._tensor_cache.move_to_end(job_id)
                return self._tensor_cache[job_id]

        job = get_job(self._db, job_id)
        if not job or not job.get("tensor_path"):
            return None

        tensor_path = Path(job["tensor_path"])
        if not tensor_path.exists():
            return None

        tensor = torch.load(tensor_path, weights_only=True)
        with self._cache_lock:
            self._tensor_cache[job_id] = tensor
            self._enforce_cache_limit()
        return tensor

    def process_job(self, job: dict) -> None:
        """Process a single embedding job."""
        job_id = job["id"]
        update_job(self._db, job_id, status="processing")

        try:
            total_start = time.perf_counter_ns()
            file_path = Path(job["file_path"])

            if job["file_type"] == "image":
                image = load_image(file_path)
                pages = [image]
            else:
                pdf_data = file_path.read_bytes()
                pages = render_pages(pdf_data, dpi=job["dpi"])
                if not pages:
                    raise ValueError("PDF contains no pages")

            page_embeddings = embed(pages, self._model, self._processor)
            total_duration = time.perf_counter_ns() - total_start
            page_embeddings = page_embeddings.cpu()

            # Save results
            result_path = self._results_dir / f"{job_id}.json"
            tensor_path = self._results_dir / f"{job_id}.pt"

            result_json = json.dumps(
                {
                    "file_name": job["file_stem"],
                    "model": MODEL_ID,
                    "dpi": job["dpi"],
                    "embeddings": page_embeddings.tolist(),
                    "total_duration": total_duration,
                    "page_count": len(pages),
                }
            )
            result_path.write_text(result_json)
            torch.save(page_embeddings, tensor_path)

            # Cache the tensor
            with self._cache_lock:
                self._tensor_cache[job_id] = page_embeddings
                self._enforce_cache_limit()

            update_job(
                self._db,
                job_id,
                status="completed",
                page_count=len(pages),
                duration_ns=total_duration,
                result_path=str(result_path),
                tensor_path=str(tensor_path),
            )
        except Exception as e:
            update_job(self._db, job_id, status="failed", error=str(e))
