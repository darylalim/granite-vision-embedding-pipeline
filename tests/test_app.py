from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from PIL import Image
from pytest import approx

from streamlit_app import embed, get_device, render_pages, search

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
        data = (FIXTURE_DIR / "test.pdf").read_bytes()
        pages = render_pages(data)
        assert len(pages) >= 1
        assert all(isinstance(p, Image.Image) for p in pages)
        assert all(p.mode == "RGB" for p in pages)

    def test_returns_empty_for_no_pages(self) -> None:
        pages = render_pages(b"%PDF-1.4\n%%EOF\n")
        assert pages == []

    def test_returns_empty_for_invalid_data(self) -> None:
        pages = render_pages(b"not a pdf")
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

        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (num_pages, num_patches, embedding_dim)

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


class TestSearch:
    def test_returns_ranked_results(self) -> None:
        mock_processor = MagicMock()
        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_batch
        mock_processor.process_texts.return_value = mock_batch

        query_embedding = torch.tensor([[0.1, 0.2, 0.3]])
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.return_value = query_embedding

        scores = torch.tensor([[0.5, 0.9, 0.2]])
        mock_processor.score.return_value = scores

        image_embeddings = torch.randn(3, 128)

        results = search("test query", mock_model, mock_processor, image_embeddings)

        assert len(results) == 3
        assert results[0] == (1, approx(0.9))
        assert results[1] == (0, approx(0.5))
        assert results[2] == (2, approx(0.2))
        mock_processor.process_texts.assert_called_once_with(["test query"])

    def test_calls_process_texts_and_score(self) -> None:
        mock_processor = MagicMock()
        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_batch
        mock_processor.process_texts.return_value = mock_batch

        query_embedding = torch.tensor([[0.1, 0.2, 0.3]])
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.return_value = query_embedding

        scores = torch.tensor([[0.5, 0.9]])
        mock_processor.score.return_value = scores

        image_embeddings = torch.randn(2, 128)

        search("find charts", mock_model, mock_processor, image_embeddings)

        mock_processor.process_texts.assert_called_once_with(["find charts"])
        mock_batch.to.assert_called_once_with("cpu")
        mock_processor.score.assert_called_once()
