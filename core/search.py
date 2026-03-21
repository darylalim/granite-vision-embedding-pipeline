import torch

from core.types import EmbeddingProcessor


def search_multi(
    query: str,
    model: torch.nn.Module,
    processor: EmbeddingProcessor,
    embeddings: dict[str, torch.Tensor],
    filter_file_id: str | None = None,
) -> list[tuple[str, int, float]]:
    """Score a text query across multiple documents and return ranked results."""
    if filter_file_id is not None:
        if filter_file_id not in embeddings:
            return []
        docs = {filter_file_id: embeddings[filter_file_id]}
    else:
        docs = embeddings
    batch = processor.process_queries([query])
    batch = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    with torch.inference_mode():
        query_embedding = model(**batch)
    ranked: list[tuple[str, int, float]] = []
    for file_id, page_embeddings in docs.items():
        scores = processor.score(
            query_embedding, page_embeddings, device=str(model.device)
        )
        for page_idx in range(scores.shape[1]):
            ranked.append((file_id, page_idx, scores[0][page_idx].item()))
    ranked.sort(key=lambda x: x[2], reverse=True)
    return ranked


def filter_results(
    results: list[tuple[str, int, float]],
    top_k: int = 5,
    min_score: float = 0.0,
) -> list[tuple[str, int, float]]:
    """Apply score threshold then top-K to ranked search results."""
    filtered = [r for r in results if r[2] >= min_score]
    return filtered[:top_k]
