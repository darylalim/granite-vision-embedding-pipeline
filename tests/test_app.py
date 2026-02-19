from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.pipeline_options import TableFormerMode
from docling.document_converter import DocumentConverter
from transformers import BatchEncoding

from streamlit_app import build_pipeline_options, convert, embed, get_device

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
    def test_returns_normalized_vector_and_token_count(self) -> None:
        embedding_dim = 128
        seq_len = 5

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = BatchEncoding(
            {
                "input_ids": torch.ones(1, seq_len, dtype=torch.long),
                "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
            }
        )

        hidden_states = torch.randn(1, seq_len, embedding_dim)
        mock_model = MagicMock()
        mock_model.return_value = (hidden_states,)

        vector, count = embed("test text", mock_model, mock_tokenizer, "cpu")

        assert count == seq_len
        assert len(vector) == embedding_dim
        norm = sum(x**2 for x in vector) ** 0.5
        assert abs(norm - 1.0) < 1e-5

    def test_tokenizer_receives_plain_string(self) -> None:
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = BatchEncoding(
            {
                "input_ids": torch.ones(1, 3, dtype=torch.long),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
            }
        )

        mock_model = MagicMock()
        mock_model.return_value = (torch.randn(1, 3, 64),)

        embed("hello world", mock_model, mock_tokenizer, "cpu")

        args, kwargs = mock_tokenizer.call_args
        assert args == ("hello world",)
        assert kwargs == {"padding": True, "truncation": True, "return_tensors": "pt"}

    def test_moves_inputs_to_device(self) -> None:
        batch = BatchEncoding(
            {
                "input_ids": torch.ones(1, 3, dtype=torch.long),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
            }
        )
        mock_tokenizer = MagicMock(return_value=batch)

        mock_model = MagicMock()
        mock_model.return_value = (torch.randn(1, 3, 64),)

        with patch.object(BatchEncoding, "to", return_value=batch) as mock_to:
            embed("text", mock_model, mock_tokenizer, "cpu")
            mock_to.assert_called_once_with("cpu")
