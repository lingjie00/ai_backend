"""Loads image data."""

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageDraw, ImageFont

from ai_backend.message.image_model import BoundingBox, ImageSize
from ai_backend.types import PathLike

ENCODING = "utf-8"
OPTIMIZED_IMAGE_TYPE = "jpeg"
OPTIMIZED_MIME_TYPE = f"image/{OPTIMIZED_IMAGE_TYPE}"


def _encode_image_bytes_to_base64(image_bytes: bytes) -> str:
    """Encodes image bytes to a base64 string."""
    return base64.b64encode(image_bytes).decode(ENCODING)


def _encode_image_Image_to_base64(image: Image.Image) -> str:
    """Encodes a PIL Image object to a base64 string."""

    # 1. Prepare for JPEG by removing transparency/Alpha
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        # Create white background to replace transparency
        background = Image.new("RGB", image.size, (255, 255, 255))
        # Use the last channel (-1) as the mask to handle both RGBA and LA
        background.paste(image, mask=image.split()[-1])
        image = background
    elif image.mode != "RGB":
        # Handles CMYK, L, etc.
        image = image.convert("RGB")

    buffered = BytesIO()
    image.save(buffered, format=OPTIMIZED_IMAGE_TYPE.upper())
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


def annotate_image_with_bounding_box(
    image: Image.Image,
    bounding_box: BoundingBox,
    label: str = "",
    color: str = "red",
    linewidth_scale: float = 0.005,
    linewidth_min: int = 2,
    fontsize_scale: float = 0.05,
    fontsize_min: int = 20,
    font_shift_scale: float = 0.05,
) -> Image.Image:
    """Draw optional bounding box overlay with question label on an image.

    bounding_box: BoundingBox object with normalized coordinates (0-1)
    label: Optional label to display on the bounding box
    color: Color of the bounding box and label
    linewidth_scale: Scale factor for the bounding box line width
    linewidth_min: Minimum line width for the bounding box
    fontsize_scale: Scale factor for the label font size
    fontsize_min: Minimum font size for the label
    font_shift_scale: Scale factor for shifting the label above the bounding box
    """
    x_min, y_min, x_max, y_max = (
        bounding_box.x_min,
        bounding_box.y_min,
        bounding_box.x_max,
        bounding_box.y_max,
    )

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size

    # Convert normalized coordinates to pixel coordinates
    left = int(x_min * width)
    top = int(y_min * height)
    right = int(x_max * width)
    bottom = int(y_max * height)

    left, right = sorted((max(0, left), min(width, right)))
    top, bottom = sorted((max(0, top), min(height, bottom)))
    if left >= right or top >= bottom:
        raise ValueError("Invalid bounding box with non-positive area")

    line_width = max(linewidth_min, int(min(width, height) * linewidth_scale))
    draw.rectangle([(left, top), (right, bottom)], outline=color, width=line_width)

    if label:
        shift = line_width + font_shift_scale * min(height, width)
        text_y = max(0, top - shift)
        font_size = max(fontsize_min, int(min(width, height) * fontsize_scale))
        try:
            font = ImageFont.load_default(size=font_size)
        except TypeError:
            font = ImageFont.load_default()
        draw.text(
            (left, text_y),
            label,
            fill=color,
            font=font,
        )

    return annotated


def normalized_to_crop_tuple(norm_box: BoundingBox, img_size: ImageSize):
    """
    norm_box: BoundingBox object with x_min, x_max, y_min, y_max (0-1)
    img_size: ImageSize object
    """
    width, height = img_size.width, img_size.height

    # Calculate absolute pixels
    left = norm_box.x_min * width
    upper = norm_box.y_min * height
    right = norm_box.x_max * width
    lower = norm_box.y_max * height

    # PIL crop() requires integers
    return (int(left), int(upper), int(right), int(lower))


def normalize_bbox(
    box: dict[Literal["left", "top", "width", "height"], int],
    img_size: ImageSize,
) -> BoundingBox:
    """
    Converts pixel coordinates from streamlit_cropper to normalized [0, 1] range.
    box: {"left": int, "top": int, "width": int, "height": int}
    img_size: ImageSize object
    """
    img_w, img_h = img_size.width, img_size.height

    x_min = box["left"] / img_w
    x_max = (box["left"] + box["width"]) / img_w
    y_min = box["top"] / img_h
    y_max = (box["top"] + box["height"]) / img_h

    return BoundingBox.model_validate(
        {
            "x_min": round(x_min, 4),
            "y_min": round(y_min, 4),
            "x_max": round(x_max, 4),
            "y_max": round(y_max, 4),
        }
    )
