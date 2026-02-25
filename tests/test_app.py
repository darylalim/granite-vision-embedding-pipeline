from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from streamlit_app import embed, get_device, render_pages

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class TestGetDevice:
    @patch("streamlit_app.torch")
    def test_prefers_mps(self, mock_torch: MagicMock) -> None:
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.cuda.is_available.return_value = True
        assert get_device() == "mps"

    @patch("streamlit_app.torch")
    def test_falls_back_to_cuda(self, mock_torch: MagicMock) -> None:
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = True
        assert get_device() == "cuda"

    @patch("streamlit_app.torch")
    def test_falls_back_to_cpu(self, mock_torch: MagicMock) -> None:
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False
        assert get_device() == "cpu"


class TestRenderPages:
    def test_renders_fixture_pdf(self) -> None:
        pages = render_pages(str(FIXTURE_DIR / "test.pdf"))
        assert len(pages) >= 1
        assert all(isinstance(p, Image.Image) for p in pages)
        assert all(p.mode == "RGB" for p in pages)

    def test_returns_empty_for_no_pages(self, tmp_path: Path) -> None:
        empty_pdf = tmp_path / "empty.pdf"
        empty_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
        pages = render_pages(str(empty_pdf))
        assert pages == []


class TestEmbed:
    def test_returns_per_page_embeddings(self) -> None:
        num_pages = 2
        num_patches = 4
        embedding_dim = 128

        mock_processor = MagicMock()
        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_batch
        mock_processor.process_images.return_value = mock_batch

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.return_value = torch.randn(num_pages, num_patches, embedding_dim)

        images = [Image.new("RGB", (64, 64)) for _ in range(num_pages)]
        embeddings = embed(images, mock_model, mock_processor)

        assert len(embeddings) == num_pages
        assert all(len(page) == num_patches for page in embeddings)
        assert all(len(vec) == embedding_dim for page in embeddings for vec in page)

    def test_calls_process_images(self) -> None:
        mock_processor = MagicMock()
        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_batch
        mock_processor.process_images.return_value = mock_batch

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.return_value = torch.randn(1, 4, 128)

        images = [Image.new("RGB", (64, 64))]
        embed(images, mock_model, mock_processor)

        mock_processor.process_images.assert_called_once_with(images)
        mock_batch.to.assert_called_once_with("cpu")
