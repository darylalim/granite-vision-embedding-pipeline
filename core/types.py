from typing import Any, Protocol

import torch
from PIL import Image


class EmbeddingProcessor(Protocol):
    def process_images(self, images: list[Image.Image]) -> dict[str, Any]: ...
    def process_queries(self, queries: list[str]) -> dict[str, Any]: ...
    def score(
        self, qs: torch.Tensor, ps: torch.Tensor, *, device: str
    ) -> torch.Tensor: ...
