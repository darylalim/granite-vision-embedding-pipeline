from unittest.mock import MagicMock

import torch


def make_mock_model(return_value: torch.Tensor) -> MagicMock:
    """Create a mock embedding model that returns the given tensor."""
    model = MagicMock()
    model.device = "cpu"
    model.return_value = return_value
    return model


def make_image_processor() -> MagicMock:
    """Create a mock processor for image embedding."""
    mock_val = MagicMock(spec=torch.Tensor)
    mock_val.to.return_value = mock_val
    processor = MagicMock()
    processor.process_images.return_value = {"pixel_values": mock_val}
    return processor


def make_query_processor() -> MagicMock:
    """Create a mock processor for query embedding."""
    mock_val = MagicMock(spec=torch.Tensor)
    mock_val.to.return_value = mock_val
    processor = MagicMock()
    processor.process_queries.return_value = {"input_ids": mock_val}
    return processor


def make_mock_processor() -> MagicMock:
    """Create a combined image + query processor mock."""
    mock_val = MagicMock(spec=torch.Tensor)
    mock_val.to.return_value = mock_val
    processor = MagicMock()
    processor.process_images.return_value = {"pixel_values": mock_val}
    processor.process_queries.return_value = {"input_ids": mock_val}
    return processor
