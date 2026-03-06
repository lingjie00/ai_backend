"""Defines the data model for image-related objects."""

import base64
from io import BytesIO
from typing import Any

from PIL import Image
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box covering the question and related student working region."""

    x_min: float = Field(
        description=("Left coordinate in normalized range [0, 1]."),
        ge=0.0,
        le=1.0,
    )
    y_min: float = Field(
        description=("Top coordinate in normalized range [0, 1]."),
        ge=0.0,
        le=1.0,
    )
    x_max: float = Field(
        description=("Right coordinate in normalized range [0, 1]."),
        ge=0.0,
        le=1.0,
    )
    y_max: float = Field(
        description=("Bottom coordinate in normalized range [0, 1]."),
        ge=0.0,
        le=1.0,
    )


class ImageData(BaseModel):
    """Model representing an image uploaded by the user."""

    filename: str = Field(
        default_factory=str, description="Original filename of the uploaded image"
    )
    page_number: int = Field(
        default_factory=lambda: 1,
        description=(
            "Page number the image is associated with, if applicable. "
            "Defaults to 1 for single-page documents."
        ),
        ge=1,
    )
    base64_content: str = Field(
        ..., description="Binary content of the image in string format", repr=False
    )
    mime_type: str = Field(..., description="MIME type of the image")
    selected: bool = Field(
        default_factory=lambda: True,
        description="Indicates if the image is selected for processing",
    )
    processed: bool = Field(
        default_factory=lambda: False,
        description="Indicates if the image has been processed",
    )

    def model_post_init(self, context: Any) -> None:
        self.mime_type = self.mime_type.lower()
        return super().model_post_init(context)

    def to_bytes(self) -> bytes:
        """Converts the base64 content to bytes."""
        return base64.b64decode(self.base64_content.encode("utf-8"))

    def to_bytesio(self) -> BytesIO:
        """Converts the base64 content to a BytesIO object."""
        return BytesIO(self.to_bytes())

    def to_pil_image(self) -> Image.Image:
        """Converts the base64 content to a PIL Image object."""
        return Image.open(self.to_bytesio())

    def show(self) -> None:
        """Displays the image using PIL."""
        image = self.to_pil_image()
        image.show()
