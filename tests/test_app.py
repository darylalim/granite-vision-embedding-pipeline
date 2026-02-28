from pathlib import Path
from unittest.mock import MagicMock, patch

import io

import pytest
import torch
from PIL import Image, UnidentifiedImageError
from pytest import approx

from streamlit_app import (
    DPI_OPTIONS,
    IMAGE_EXTENSIONS,
    EmbedResults,
    cleanup_stale_results,
    embed,
    filter_results,
    get_device,
    render_pages,
    search,
    search_multi,
)

PDF_DATA_DIR = Path(__file__).parent / "data" / "pdf"
IMAGE_DATA_DIR = Path(__file__).parent / "data" / "images"


def _make_result(
    file_id: str,
    file_stem: str,
    *,
    page_embeddings: torch.Tensor | None = None,
) -> EmbedResults:
    return {
        "file_id": file_id,
        "pages": [],
        "page_embeddings": page_embeddings
        if page_embeddings is not None
        else torch.empty(0),
        "total_duration": 0,
        "file_stem": file_stem,
        "dpi": 150,
        "json": "{}",
    }


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


class TestLoadImage:
    def test_loads_png_as_rgb(self) -> None:
        img = Image.open(IMAGE_DATA_DIR / "red.png").convert("RGB")
        assert img.mode == "RGB"
        assert img.size == (64, 64)

    def test_loads_jpg_as_rgb(self) -> None:
        img = Image.open(IMAGE_DATA_DIR / "blue.jpg").convert("RGB")
        assert img.mode == "RGB"
        assert img.size == (64, 64)

    def test_loads_webp_as_rgb(self) -> None:
        img = Image.open(IMAGE_DATA_DIR / "green.webp").convert("RGB")
        assert img.mode == "RGB"
        assert img.size == (64, 64)

    def test_embed_accepts_image_fixture(self) -> None:
        mock_processor = MagicMock()
        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_batch
        mock_processor.process_images.return_value = mock_batch

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.return_value = torch.randn(1, 4, 128)

        img = Image.open(IMAGE_DATA_DIR / "red.png").convert("RGB")
        result = embed([img], mock_model, mock_processor)

        assert isinstance(result, torch.Tensor)
        mock_processor.process_images.assert_called_once_with([img])

    def test_missing_file_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            Image.open(IMAGE_DATA_DIR / "nonexistent.png")

    def test_corrupt_data_raises_unidentified_image(self) -> None:
        with pytest.raises(UnidentifiedImageError):
            Image.open(io.BytesIO(b"not an image")).convert("RGB")


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

    def test_returns_empty_for_no_pages(self) -> None:
        pages = render_pages(b"%PDF-1.4\n%%EOF\n")
        assert pages == []

    def test_returns_empty_for_invalid_data(self) -> None:
        pages = render_pages(b"not a pdf")
        assert pages == []

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
        mock_processor.process_queries.return_value = mock_batch

        query_embedding = torch.tensor([[0.1, 0.2, 0.3]])
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.return_value = query_embedding

        scores = torch.tensor([[0.5, 0.9, 0.2]])
        mock_processor.score_multi_vector.return_value = scores

        image_embeddings = torch.randn(3, 128)

        results = search("test query", mock_model, mock_processor, image_embeddings)

        assert len(results) == 3
        assert results[0] == (1, approx(0.9))
        assert results[1] == (0, approx(0.5))
        assert results[2] == (2, approx(0.2))
        mock_processor.process_queries.assert_called_once_with(["test query"])

    def test_calls_process_queries_and_score_multi_vector(self) -> None:
        mock_processor = MagicMock()
        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_batch
        mock_processor.process_queries.return_value = mock_batch

        query_embedding = torch.tensor([[0.1, 0.2, 0.3]])
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.return_value = query_embedding

        scores = torch.tensor([[0.5, 0.9]])
        mock_processor.score_multi_vector.return_value = scores

        image_embeddings = torch.randn(2, 128)

        search("find charts", mock_model, mock_processor, image_embeddings)

        mock_processor.process_queries.assert_called_once_with(["find charts"])
        mock_batch.to.assert_called_once_with("cpu")
        mock_processor.score_multi_vector.assert_called_once()


class TestCleanupStaleResults:
    def test_removes_stale_entries(self) -> None:
        results: dict[str, EmbedResults] = {
            "id1": _make_result("id1", "doc1"),
            "id2": _make_result("id2", "doc2"),
        }
        cleanup_stale_results({"id1"}, results)
        assert "id1" in results
        assert "id2" not in results

    def test_keeps_all_when_none_stale(self) -> None:
        results: dict[str, EmbedResults] = {
            "id1": _make_result("id1", "doc1"),
        }
        cleanup_stale_results({"id1"}, results)
        assert len(results) == 1

    def test_removes_all_when_all_stale(self) -> None:
        results: dict[str, EmbedResults] = {
            "id1": _make_result("id1", "doc1"),
        }
        cleanup_stale_results(set(), results)
        assert len(results) == 0

    def test_handles_empty_results(self) -> None:
        results: dict[str, EmbedResults] = {}
        cleanup_stale_results({"id1"}, results)
        assert len(results) == 0


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


class TestSearchMulti:
    def test_returns_cross_document_ranked_results(self) -> None:
        mock_processor = MagicMock()
        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_batch
        mock_processor.process_queries.return_value = mock_batch

        query_embedding = torch.tensor([[0.1, 0.2, 0.3]])
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.return_value = query_embedding

        # Doc A: 2 pages scoring [0.3, 0.8], Doc B: 1 page scoring [0.6]
        mock_processor.score_multi_vector.side_effect = [
            torch.tensor([[0.3, 0.8]]),
            torch.tensor([[0.6]]),
        ]

        results: dict[str, EmbedResults] = {
            "id_a": _make_result(
                "id_a", "doc_a", page_embeddings=torch.randn(2, 4, 128)
            ),
            "id_b": _make_result(
                "id_b", "doc_b", page_embeddings=torch.randn(1, 4, 128)
            ),
        }

        ranked = search_multi("test query", mock_model, mock_processor, results)

        assert len(ranked) == 3
        assert ranked[0] == ("id_a", 1, approx(0.8))
        assert ranked[1] == ("id_b", 0, approx(0.6))
        assert ranked[2] == ("id_a", 0, approx(0.3))

    def test_filters_to_single_document(self) -> None:
        mock_processor = MagicMock()
        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_batch
        mock_processor.process_queries.return_value = mock_batch

        query_embedding = torch.tensor([[0.1, 0.2, 0.3]])
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.return_value = query_embedding

        mock_processor.score_multi_vector.return_value = torch.tensor([[0.5]])

        results: dict[str, EmbedResults] = {
            "id_a": _make_result(
                "id_a", "doc_a", page_embeddings=torch.randn(1, 4, 128)
            ),
            "id_b": _make_result(
                "id_b", "doc_b", page_embeddings=torch.randn(1, 4, 128)
            ),
        }

        ranked = search_multi(
            "test query", mock_model, mock_processor, results, filter_file_id="id_b"
        )

        assert len(ranked) == 1
        assert ranked[0][0] == "id_b"
        assert mock_processor.score_multi_vector.call_count == 1

    def test_encodes_query_once_for_multiple_docs(self) -> None:
        mock_processor = MagicMock()
        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_batch
        mock_processor.process_queries.return_value = mock_batch

        query_embedding = torch.tensor([[0.1, 0.2, 0.3]])
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.return_value = query_embedding

        mock_processor.score_multi_vector.side_effect = [
            torch.tensor([[0.3]]),
            torch.tensor([[0.6]]),
        ]

        results: dict[str, EmbedResults] = {
            "id_a": _make_result(
                "id_a", "doc_a", page_embeddings=torch.randn(1, 4, 128)
            ),
            "id_b": _make_result(
                "id_b", "doc_b", page_embeddings=torch.randn(1, 4, 128)
            ),
        }

        search_multi("test", mock_model, mock_processor, results)

        mock_processor.process_queries.assert_called_once_with(["test"])
        assert mock_model.call_count == 1
