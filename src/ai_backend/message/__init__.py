"""Loads different file types into a standardized format for use in AI interactions."""

import mimetypes
import re
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages.content import ImageContentBlock, create_image_block

from ai_backend.message.image_loader import (
    OPTIMIZED_MIME_TYPE,
    annotate_image_with_bounding_box,
    decode_base64_to_Image,
    encode_image_to_base64,
    normalize_bbox,
    normalized_to_crop_tuple,
    optimize_image,
)
from ai_backend.message.image_model import BoundingBox, ImageData, ImageSize
from ai_backend.message.pdf_loader import (
    CONVERTED_IMAGE_MIME_TYPE,
    encode_pdf_to_images_bytes,
)


def is_base64_regex(s: str) -> bool:
    # Pattern: Optional data URI prefix, then base64 chars, ending with optional padding
    # Length must be a multiple of 4
    pattern = r"^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$"
    return bool(re.match(pattern, s))


class MessageLoader:
    """Utility class for loading and encoding files from various sources."""

    @staticmethod
    def optimize_image_data(
        image_data: ImageData, max_dimension: int = 1024
    ) -> ImageData:
        """Optimizes the image data by resizing it if it exceeds the max dimension."""
        image = decode_base64_to_Image(image_data.base64_content)
        optimized_image = optimize_image(image, max_dimension)
        optimized_base64 = encode_image_to_base64(optimized_image)
        return ImageData(
            base64_content=optimized_base64,
            filename=image_data.filename,
            mime_type=OPTIMIZED_MIME_TYPE,
            page_number=image_data.page_number,
            selected=image_data.selected,
            processed=image_data.processed,
        )

    @staticmethod
    def convert_pdf_to_image_data(
        pdf_input: Any, filename: str = "", dpi: int = 300
    ) -> list[ImageData]:
        """Converts PDF input to a list of ImageData objects, one per page."""
        image_bytes_list = encode_pdf_to_images_bytes(pdf_input, dpi)
        image_data_list = []
        for i, image_bytes in enumerate(image_bytes_list):
            image_data = MessageLoader.convert_image_to_image_data(
                image_bytes,
                filename=filename if filename else "",
                mime_type=CONVERTED_IMAGE_MIME_TYPE,
                page_number=i + 1,
            )
            image_data_list.append(image_data)
        return image_data_list

    @staticmethod
    def convert_image_to_image_data(
        inputs: Any,
        filename: str = "",
        mime_type: str = "",
        page_number: int = 1,
    ) -> ImageData:
        """Converts various image input types to an ImageData object."""
        not_base64 = not is_base64_regex(str(inputs))
        if isinstance(inputs, (Path, str)) and not_base64:
            filepath = Path(inputs)
            filename = filepath.name if filename == "" else filename
            try:
                mime_type = mimetypes.guess_type(filepath)[0] or ""
            except Exception as e:
                if mime_type == "":
                    raise ValueError(
                        f"Could not determine MIME type for the image at {filepath}. "
                        "Please provide the MIME type explicitly. "
                        f"Error: {e}"
                    )

        if not_base64:
            base64_content = encode_image_to_base64(inputs)
        else:
            base64_content = str(inputs)

        if mime_type == "":
            raise ValueError("Could not determine MIME type for the image.")

        return ImageData(
            base64_content=base64_content,
            filename=filename,
            mime_type=mime_type,
            page_number=page_number,
        )

    @staticmethod
    def convert_image_data_to_langchain_content(
        image_data: ImageData, max_dimension: int = 1024, optimize: bool = True
    ) -> ImageContentBlock:
        """Converts ImageData to a format suitable for LangChain content."""
        if optimize:
            image_data = MessageLoader.optimize_image_data(image_data, max_dimension)
        else:
            image_data = image_data
        encoded_string = image_data.base64_content
        filename = image_data.filename
        mime_type = image_data.mime_type
        if not mime_type:
            raise ValueError("MIME type is required to convert image data.")
        return create_image_block(
            base64=encoded_string,
            filename=filename,
            mime_type=mime_type,
        )

    @staticmethod
    def annotate_image_with_bounding_box(
        image_data: ImageData, bounding_box: BoundingBox
    ) -> ImageData:
        """Annotates the image with a bounding box and returns new ImageData."""
        image = decode_base64_to_Image(image_data.base64_content)
        annotated_image = annotate_image_with_bounding_box(image, bounding_box)
        annotated_base64 = encode_image_to_base64(annotated_image)
        return ImageData(
            base64_content=annotated_base64,
            filename=image_data.filename,
            mime_type=image_data.mime_type,
            page_number=image_data.page_number,
            selected=image_data.selected,
            processed=image_data.processed,
        )

    @staticmethod
    def crop_image_with_bounding_box(
        image_data: ImageData, bounding_box: BoundingBox
    ) -> ImageData:
        """Crops the image using the bounding box and returns new ImageData."""
        image = decode_base64_to_Image(image_data.base64_content)
        image_size = ImageSize(width=image.width, height=image.height)
        crop_tuple = normalized_to_crop_tuple(
            bounding_box,
            image_size,
        )
        cropped_image = image.crop(crop_tuple)
        cropped_base64 = encode_image_to_base64(cropped_image)
        return ImageData(
            base64_content=cropped_base64,
            filename=image_data.filename,
            mime_type=image_data.mime_type,
            page_number=image_data.page_number,
            selected=image_data.selected,
            processed=image_data.processed,
        )

    @staticmethod
    def crop_image_with_bounding_box_dict(
        image_data: ImageData,
        bounding_box: dict[Literal["left", "top", "width", "height"], int],
    ) -> ImageData:
        """Crops the image using the bounding box dict and returns new ImageData."""
        image = decode_base64_to_Image(image_data.base64_content)
        image_size = ImageSize(width=image.width, height=image.height)
        normalized_box = normalize_bbox(bounding_box, image_size)
        crop_tuple = normalized_to_crop_tuple(normalized_box, image_size)
        cropped_image = image.crop(crop_tuple)
        cropped_base64 = encode_image_to_base64(cropped_image)
        return ImageData(
            base64_content=cropped_base64,
            filename=image_data.filename,
            mime_type=image_data.mime_type,
            page_number=image_data.page_number,
            selected=image_data.selected,
            processed=image_data.processed,
        )


__all__ = [
    "BoundingBox",
    "ImageData",
    "MessageLoader",
]
