"""Loads PDF data."""

from pathlib import Path

import fitz  # PyMuPDF

from ai_backend.types import PathLike

PDF_MIME_TYPES = "application/pdf"
CONVERTED_IMAGE_MIME_TYPE = "image/jpeg"


def _encode_pdf_bytes_to_images_bytes(pdf_bytes: bytes, dpi: int = 200) -> list[bytes]:
    """
    Convert a PDF file to a list of image bytes.

    Args:
        pdf_bytes: PDF file content as bytes
        dpi: Resolution for rendering (default: 200)

    Returns:
        List of image bytes, one per page

    Example:
        >>> with open("document.pdf", "rb") as f:
        ...     pdf_bytes = f.read()
        >>> images = pdf_bytes_to_images(pdf_bytes)
        >>> print(f"Converted {len(images)} pages")
    """
    images: list[bytes] = []

    # Open PDF from bytes
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

    try:
        # Convert each page to an image
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]

            # Render page to pixmap at specified DPI
            # zoom factor: dpi/72 (72 is the default DPI)
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Convert pixmap to PIL Image
            img_data = pix.tobytes("jpg")
            images.append(img_data)

    finally:
        pdf_document.close()

    return images


def _encode_pdf_path_to_image_bytes(pdf_path: PathLike, dpi: int = 200) -> list[bytes]:
    """
    Convert a PDF file to a list of image bytes.

    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for rendering (default: 200)

    Returns:
        List of image bytes, one per page
    """
    pdf_path = Path(pdf_path)
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    return _encode_pdf_bytes_to_images_bytes(pdf_bytes, dpi)


def encode_pdf_to_images_bytes(
    pdf_input: bytes | PathLike, dpi: int = 200
) -> list[bytes]:
    """
    Convert a PDF file (from bytes or file path) to a list of image bytes.

    Args:
        pdf_input: PDF file content as bytes or path to the PDF file
        dpi: Resolution for rendering (default: 200)

    Returns:
        List of image bytes, one per page
    """
    if isinstance(pdf_input, bytes):
        return _encode_pdf_bytes_to_images_bytes(pdf_input, dpi)
    elif isinstance(pdf_input, (str, Path)):
        return _encode_pdf_path_to_image_bytes(pdf_input, dpi)
    else:
        raise ValueError("Unsupported input type for PDF encoding.")
