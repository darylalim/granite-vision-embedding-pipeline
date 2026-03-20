from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from core.constants import MODEL_ID
from core.types import EmbeddingProcessor


def get_device() -> str:
    """Detect best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(device: str) -> tuple[Any, Any]:
    """Load embedding model and processor."""
    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model, processor


def load_image(path: Path) -> Image.Image:
    """Load an image file and convert to RGB.

    Lets UnidentifiedImageError and OSError propagate to caller.
    """
    return Image.open(path).convert("RGB")


def embed(
    images: list[Image.Image], model: torch.nn.Module, processor: EmbeddingProcessor
) -> torch.Tensor:
    """Generate per-page multi-vector embeddings from images."""
    batch = processor.process_images(images)
    batch = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    with torch.inference_mode():
        embeddings = model(**batch)
    return embeddings
