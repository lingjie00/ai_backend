"""Loads image data."""

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from ai_backend.types import PathLike

ENCODING = "utf-8"


def _encode_image_bytes_to_base64(image_bytes: bytes) -> str:
    """Encodes image bytes to a base64 string."""
    return base64.b64encode(image_bytes).decode(ENCODING)


def _encode_image_Image_to_base64(image: Image.Image) -> str:
    """Encodes a PIL Image object to a base64 string."""

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return _encode_image_bytes_to_base64(buffered.getvalue())


def _encode_image_file_to_base64(image_path: PathLike) -> str:
    """Encodes an image file to a base64 string."""
    image_path = Path(image_path)

    with open(image_path, "rb") as image_file:
        encoded_string = _encode_image_bytes_to_base64(image_file.read())
    return encoded_string


def _decode_base64_to_image_bytes(encoded_string: str) -> bytes:
    """Decodes a base64 string back to image bytes."""
    return base64.b64decode(encoded_string.encode(ENCODING))


def decode_base64_to_Image(encoded_string: str) -> Image.Image:
    """Decodes a base64 string back to a PIL Image object."""
    image_bytes = _decode_base64_to_image_bytes(encoded_string)
    return Image.open(BytesIO(image_bytes))


def encode_image_to_base64(inputs: Any) -> str:
    """Encodes various image input types to a base64 string."""
    if isinstance(inputs, bytes):
        return _encode_image_bytes_to_base64(inputs)
    elif isinstance(inputs, Image.Image):
        return _encode_image_Image_to_base64(inputs)
    elif isinstance(inputs, (str, Path)):
        return _encode_image_file_to_base64(inputs)
    else:
        raise ValueError("Unsupported input type for image encoding.")


def optimize_image(
    image: Image.Image, max_dimension: int = 1024, format: str = "JPEG"
) -> Image.Image:
    """Optimizes image by resizing it to fit within the specified maximum dimension."""
    if max(image.size) > max_dimension:
        ratio = max_dimension / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    # convert to RGB
    if format.upper() == "JPEG" and image.mode in ("RGBA", "LA"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode in ("RGBA", "LA"):
            background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        else:
            background.paste(image)
        image = background
    elif format.upper() == "JPEG" and image.mode != "RGB":
        image = image.convert("RGB")  # Ensure compatibility with JPEG format
    return image
