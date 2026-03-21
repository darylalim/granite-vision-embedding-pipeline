from pydantic import BaseModel


class JobResponse(BaseModel):
    id: str
    status: str
    created_at: str
    updated_at: str
    file_name: str
    file_stem: str
    file_type: str
    dpi: int
    page_count: int | None = None
    duration_ns: int | None = None
    error: str | None = None


class JobCreateResponse(BaseModel):
    job_id: str
    status: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    min_score: float = 0.0
    filter_file_id: str | None = None


class SearchResult(BaseModel):
    file_id: str
    page_index: int
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]


class HealthResponse(BaseModel):
    device: str
    queue_depth: int
    worker_running: bool
