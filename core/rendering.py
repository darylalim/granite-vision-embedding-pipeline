import fitz
from PIL import Image


def _make_matrix(dpi: int) -> fitz.Matrix:
    """Build a fitz transformation matrix for the given DPI."""
    scale = dpi / 72
    return fitz.Matrix(scale, scale)


def render_pages(data: bytes, dpi: int = 150) -> list[Image.Image]:
    """Render PDF pages as PIL Images.

    Returns an empty list for empty PDFs (no pages).
    Raises ValueError for corrupt or unreadable PDF data.
    """
    try:
        matrix = _make_matrix(dpi)
        with fitz.open(stream=data, filetype="pdf") as doc:
            return [
                Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                for page in doc
                for pix in [page.get_pixmap(matrix=matrix)]
            ]
    except (fitz.FileDataError, fitz.EmptyFileError):
        raise ValueError("Corrupt or unreadable PDF")


def render_page(data: bytes, page_index: int, dpi: int = 150) -> Image.Image:
    """Render a single PDF page by index as a PIL Image.

    Raises ValueError for corrupt PDF data.
    Raises IndexError if page_index is out of range.
    """
    try:
        matrix = _make_matrix(dpi)
        with fitz.open(stream=data, filetype="pdf") as doc:
            if page_index < 0 or page_index >= len(doc):
                raise IndexError(
                    f"Page index {page_index} out of range for {len(doc)}-page PDF"
                )
            page = doc[page_index]
            pix = page.get_pixmap(matrix=matrix)
            return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    except (fitz.FileDataError, fitz.EmptyFileError):
        raise ValueError("Corrupt or unreadable PDF")
