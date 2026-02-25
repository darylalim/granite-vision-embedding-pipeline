from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
try:
    from docling.datamodel.accelerator_options import AcceleratorDevice
    from docling.datamodel.pipeline_options import TableFormerMode
    from docling.document_converter import DocumentConverter
except ModuleNotFoundError:
    pass
from PIL import Image
from streamlit_app import build_pipeline_options, convert, embed, get_device, render_pages

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class TestGetDevice:
    @patch("streamlit_app.torch")
    def test_prefers_mps(self, mock_torch: MagicMock) -> None:
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.cuda.is_available.return_value = True
        device, accel = get_device()
        assert device == "mps"
        assert accel == AcceleratorDevice.MPS

    @patch("streamlit_app.torch")
    def test_falls_back_to_cuda(self, mock_torch: MagicMock) -> None:
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = True
        device, accel = get_device()
        assert device == "cuda"
        assert accel == AcceleratorDevice.CUDA

    @patch("streamlit_app.torch")
    def test_falls_back_to_cpu(self, mock_torch: MagicMock) -> None:
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False
        device, accel = get_device()
        assert device == "cpu"
        assert accel == AcceleratorDevice.CPU


class TestBuildPipelineOptions:
    def test_accurate_mode(self) -> None:
        opts = build_pipeline_options(
            tableformer_mode="Accurate",
            use_structure_prediction=False,
            code_understanding=False,
            formula_understanding=False,
            picture_classification=False,
            accelerator_device=AcceleratorDevice.CPU,
        )
        assert opts.table_structure_options.mode == TableFormerMode.ACCURATE
        assert opts.table_structure_options.do_cell_matching is True

    def test_fast_mode(self) -> None:
        opts = build_pipeline_options(
            tableformer_mode="Fast",
            use_structure_prediction=False,
            code_understanding=False,
            formula_understanding=False,
            picture_classification=False,
            accelerator_device=AcceleratorDevice.CPU,
        )
        assert opts.table_structure_options.mode == TableFormerMode.FAST

    def test_structure_prediction_disables_cell_matching(self) -> None:
        opts = build_pipeline_options(
            tableformer_mode="Accurate",
            use_structure_prediction=True,
            code_understanding=False,
            formula_understanding=False,
            picture_classification=False,
            accelerator_device=AcceleratorDevice.CPU,
        )
        assert opts.table_structure_options.do_cell_matching is False

    def test_enrichment_flags(self) -> None:
        opts = build_pipeline_options(
            tableformer_mode="Accurate",
            use_structure_prediction=False,
            code_understanding=True,
            formula_understanding=True,
            picture_classification=False,
            accelerator_device=AcceleratorDevice.CPU,
        )
        assert opts.do_code_enrichment is True
        assert opts.do_formula_enrichment is True

    def test_picture_classification(self) -> None:
        opts = build_pipeline_options(
            tableformer_mode="Accurate",
            use_structure_prediction=False,
            code_understanding=False,
            formula_understanding=False,
            picture_classification=True,
            accelerator_device=AcceleratorDevice.CPU,
        )
        assert opts.generate_picture_images is True
        assert opts.images_scale == 2
        assert opts.do_picture_classification is True


class TestConvert:
    def test_converts_pdf_to_markdown(self) -> None:
        doc_converter = DocumentConverter()
        md = convert(str(FIXTURE_DIR / "test.pdf"), doc_converter)
        assert "test PDF document" in md
        assert "embedding pipeline" in md


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
