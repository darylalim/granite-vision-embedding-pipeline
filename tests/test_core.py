from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image, UnidentifiedImageError
from pytest import approx

from core.constants import DPI_OPTIONS, GENERATION_MAX_TOKENS, IMAGE_EXTENSIONS, MAX_UPLOAD_BYTES
from core.embedding import embed, get_device, load_image
from core.rendering import render_page, render_pages
from core.search import filter_results, search_multi

PDF_DATA_DIR = Path(__file__).parent / "data" / "pdf"
IMAGE_DATA_DIR = Path(__file__).parent / "data" / "images"


def _make_mock_model(return_value: torch.Tensor) -> MagicMock:
    model = MagicMock()
    model.device = "cpu"
    model.return_value = return_value
    return model


def _make_image_processor() -> MagicMock:
    mock_val = MagicMock(spec=torch.Tensor)
    mock_val.to.return_value = mock_val
    processor = MagicMock()
    processor.process_images.return_value = {"pixel_values": mock_val}
    return processor


def _make_query_processor() -> MagicMock:
    mock_val = MagicMock(spec=torch.Tensor)
    mock_val.to.return_value = mock_val
    processor = MagicMock()
    processor.process_queries.return_value = {"input_ids": mock_val}
    return processor


class TestDpiOptions:
    def test_contains_three_options(self) -> None:
        assert len(DPI_OPTIONS) == 3

    def test_values_are_72_150_300(self) -> None:
        assert sorted(DPI_OPTIONS.values()) == [72, 150, 300]

    def test_labels_match_values(self) -> None:
        for label, value in DPI_OPTIONS.items():
            assert str(value) in label


class TestImageExtensions:
    def test_contains_expected_types(self) -> None:
        assert IMAGE_EXTENSIONS == {"png", "jpg", "jpeg", "webp"}


class TestMaxUploadBytes:
    def test_equals_50_mb(self) -> None:
        assert MAX_UPLOAD_BYTES == 50 * 1024 * 1024


class TestGenerationMaxTokens:
    def test_equals_1024(self) -> None:
        assert GENERATION_MAX_TOKENS == 1024


class TestLoadImage:
    def test_loads_png_as_rgb(self) -> None:
        img = load_image(IMAGE_DATA_DIR / "red.png")
        assert img.mode == "RGB"
        assert img.size == (64, 64)

    def test_loads_jpg_as_rgb(self) -> None:
        img = load_image(IMAGE_DATA_DIR / "blue.jpg")
        assert img.mode == "RGB"
        assert img.size == (64, 64)

    def test_loads_webp_as_rgb(self) -> None:
        img = load_image(IMAGE_DATA_DIR / "green.webp")
        assert img.mode == "RGB"
        assert img.size == (64, 64)

    def test_embed_accepts_image_fixture(self) -> None:
        mock_processor = _make_image_processor()
        mock_model = _make_mock_model(torch.randn(1, 4, 128))

        img = load_image(IMAGE_DATA_DIR / "red.png")
        result = embed([img], mock_model, mock_processor)

        assert isinstance(result, torch.Tensor)
        mock_processor.process_images.assert_called_once_with([img])

    def test_missing_file_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_image(IMAGE_DATA_DIR / "nonexistent.png")

    def test_corrupt_data_raises_unidentified_image(self, tmp_path: Path) -> None:
        corrupt_file = tmp_path / "corrupt.png"
        corrupt_file.write_bytes(b"not an image")
        with pytest.raises(UnidentifiedImageError):
            load_image(corrupt_file)


class TestGetDevice:
    @patch("core.embedding.torch")
    def test_prefers_mps(self, mock_torch: MagicMock) -> None:
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.cuda.is_available.return_value = True
        assert get_device() == "mps"

    @patch("core.embedding.torch")
    def test_falls_back_to_cuda(self, mock_torch: MagicMock) -> None:
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = True
        assert get_device() == "cuda"

    @patch("core.embedding.torch")
    def test_falls_back_to_cpu(self, mock_torch: MagicMock) -> None:
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False
        assert get_device() == "cpu"


class TestRenderPages:
    def test_renders_single_page_pdf(self) -> None:
        data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        pages = render_pages(data)
        assert len(pages) == 1
        assert isinstance(pages[0], Image.Image)
        assert pages[0].mode == "RGB"

    def test_renders_multi_page_pdf(self) -> None:
        data = (PDF_DATA_DIR / "multi_page.pdf").read_bytes()
        pages = render_pages(data)
        assert len(pages) == 3
        assert all(isinstance(p, Image.Image) for p in pages)
        assert all(p.mode == "RGB" for p in pages)

    def test_raises_for_empty_pdf(self) -> None:
        with pytest.raises(ValueError, match="Corrupt or unreadable PDF"):
            render_pages(b"%PDF-1.4\n%%EOF\n")

    def test_raises_for_invalid_data(self) -> None:
        with pytest.raises(ValueError, match="Corrupt or unreadable PDF"):
            render_pages(b"not a pdf")

    def test_higher_dpi_produces_larger_images(self) -> None:
        data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        pages_72 = render_pages(data, dpi=72)
        pages_150 = render_pages(data, dpi=150)
        assert pages_150[0].width > pages_72[0].width
        assert pages_150[0].height > pages_72[0].height

    def test_default_dpi_is_150(self) -> None:
        data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        pages_default = render_pages(data)
        pages_150 = render_pages(data, dpi=150)
        assert pages_default[0].size == pages_150[0].size


class TestRenderPage:
    def test_renders_first_page(self) -> None:
        data = (PDF_DATA_DIR / "multi_page.pdf").read_bytes()
        page = render_page(data, page_index=0)
        assert isinstance(page, Image.Image)
        assert page.mode == "RGB"

    def test_renders_last_page(self) -> None:
        data = (PDF_DATA_DIR / "multi_page.pdf").read_bytes()
        page = render_page(data, page_index=2)
        assert isinstance(page, Image.Image)
        assert page.mode == "RGB"

    def test_raises_for_out_of_bounds_index(self) -> None:
        data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        with pytest.raises(IndexError):
            render_page(data, page_index=1)

    def test_raises_for_corrupt_pdf(self) -> None:
        with pytest.raises(ValueError, match="Corrupt or unreadable PDF"):
            render_page(b"not a pdf", page_index=0)

    def test_respects_dpi(self) -> None:
        data = (PDF_DATA_DIR / "single_page.pdf").read_bytes()
        page_72 = render_page(data, page_index=0, dpi=72)
        page_300 = render_page(data, page_index=0, dpi=300)
        assert page_300.width > page_72.width


class TestEmbed:
    def test_returns_per_page_embeddings(self) -> None:
        num_pages = 2
        num_patches = 4
        embedding_dim = 128

        mock_processor = _make_image_processor()
        mock_model = _make_mock_model(
            torch.randn(num_pages, num_patches, embedding_dim)
        )

        images = [Image.new("RGB", (64, 64)) for _ in range(num_pages)]
        embeddings = embed(images, mock_model, mock_processor)

        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (num_pages, num_patches, embedding_dim)

    def test_calls_process_images(self) -> None:
        mock_processor = _make_image_processor()
        mock_model = _make_mock_model(torch.randn(1, 4, 128))

        images = [Image.new("RGB", (64, 64))]
        embed(images, mock_model, mock_processor)

        mock_processor.process_images.assert_called_once_with(images)
        mock_processor.process_images.return_value[
            "pixel_values"
        ].to.assert_called_once_with("cpu")


class TestFilterResults:
    def test_filters_below_min_score(self) -> None:
        results: list[tuple[str, int, float]] = [
            ("id_a", 0, 0.9),
            ("id_a", 1, 0.3),
            ("id_b", 0, 0.6),
        ]
        filtered = filter_results(results, top_k=10, min_score=0.5)
        assert len(filtered) == 2
        assert filtered[0] == ("id_a", 0, approx(0.9))
        assert filtered[1] == ("id_b", 0, approx(0.6))

    def test_limits_to_top_k(self) -> None:
        results: list[tuple[str, int, float]] = [
            ("id_a", 0, 0.9),
            ("id_b", 0, 0.8),
            ("id_a", 1, 0.7),
        ]
        filtered = filter_results(results, top_k=2)
        assert len(filtered) == 2
        assert filtered[0] == ("id_a", 0, approx(0.9))
        assert filtered[1] == ("id_b", 0, approx(0.8))

    def test_threshold_applied_before_top_k(self) -> None:
        results: list[tuple[str, int, float]] = [
            ("id_a", 0, 0.9),
            ("id_b", 0, 0.8),
            ("id_a", 1, 0.3),
            ("id_b", 1, 0.1),
        ]
        filtered = filter_results(results, top_k=5, min_score=0.5)
        assert len(filtered) == 2

    def test_returns_empty_for_empty_input(self) -> None:
        assert filter_results([], top_k=5, min_score=0.0) == []

    def test_default_values(self) -> None:
        results: list[tuple[str, int, float]] = [
            (f"id_{i}", 0, 0.9 - i * 0.1) for i in range(7)
        ]
        filtered = filter_results(results)
        assert len(filtered) == 5
        assert filtered[0][2] == approx(0.9)

    def test_top_k_zero_returns_empty(self) -> None:
        results: list[tuple[str, int, float]] = [
            ("id_a", 0, 0.9),
            ("id_b", 0, 0.8),
        ]
        assert filter_results(results, top_k=0) == []


class TestSearchMulti:
    def test_returns_cross_document_ranked_results(self) -> None:
        mock_processor = _make_query_processor()
        mock_model = _make_mock_model(torch.tensor([[0.1, 0.2, 0.3]]))

        # Doc A: 2 pages scoring [0.3, 0.8], Doc B: 1 page scoring [0.6]
        mock_processor.score.side_effect = [
            torch.tensor([[0.3, 0.8]]),
            torch.tensor([[0.6]]),
        ]

        embeddings: dict[str, torch.Tensor] = {
            "id_a": torch.randn(2, 4, 128),
            "id_b": torch.randn(1, 4, 128),
        }

        ranked = search_multi("test query", mock_model, mock_processor, embeddings)

        assert len(ranked) == 3
        assert ranked[0] == ("id_a", 1, approx(0.8))
        assert ranked[1] == ("id_b", 0, approx(0.6))
        assert ranked[2] == ("id_a", 0, approx(0.3))

    def test_filters_to_single_document(self) -> None:
        mock_processor = _make_query_processor()
        mock_model = _make_mock_model(torch.tensor([[0.1, 0.2, 0.3]]))
        mock_processor.score.return_value = torch.tensor([[0.5]])

        embeddings: dict[str, torch.Tensor] = {
            "id_a": torch.randn(1, 4, 128),
            "id_b": torch.randn(1, 4, 128),
        }

        ranked = search_multi(
            "test query", mock_model, mock_processor, embeddings, filter_file_id="id_b"
        )

        assert len(ranked) == 1
        assert ranked[0][0] == "id_b"
        assert mock_processor.score.call_count == 1

    def test_encodes_query_once_for_multiple_docs(self) -> None:
        mock_processor = _make_query_processor()
        mock_model = _make_mock_model(torch.tensor([[0.1, 0.2, 0.3]]))
        mock_processor.score.side_effect = [
            torch.tensor([[0.3]]),
            torch.tensor([[0.6]]),
        ]

        embeddings: dict[str, torch.Tensor] = {
            "id_a": torch.randn(1, 4, 128),
            "id_b": torch.randn(1, 4, 128),
        }

        search_multi("test", mock_model, mock_processor, embeddings)

        mock_processor.process_queries.assert_called_once_with(["test"])
        assert mock_model.call_count == 1

    def test_returns_empty_for_empty_embeddings(self) -> None:
        mock_processor = _make_query_processor()
        mock_model = _make_mock_model(torch.tensor([[0.1, 0.2, 0.3]]))

        result = search_multi("test query", mock_model, mock_processor, {})

        assert result == []

    def test_returns_empty_for_missing_filter_file_id(self) -> None:
        mock_processor = _make_query_processor()
        mock_model = _make_mock_model(torch.tensor([[0.1, 0.2, 0.3]]))

        embeddings: dict[str, torch.Tensor] = {
            "id_a": torch.randn(1, 4, 128),
        }

        result = search_multi(
            "test", mock_model, mock_processor, embeddings, filter_file_id="nonexistent"
        )

        assert result == []
