"""Defines the data model for image-related objects."""

import base64
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel, Field


class ImageSize(BaseModel):
    """Model representing the dimensions of an image."""

    width: int = Field(..., description="Width of the image in pixels", ge=1)
    height: int = Field(..., description="Height of the image in pixels", ge=1)


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
    file_path: Path | None = Field(
        default=None,
        description=(
            "Path to the image file on disk. Set when the image is saved to disk "
            "for large file handling."
        ),
    )

    # Threshold in bytes for base64 content above which save_to_disk() is recommended
    LARGE_FILE_THRESHOLD: int = 5 * 1024 * 1024  # 5MB

    @property
    def size(self) -> ImageSize:
        """Returns the dimensions of the image."""
        image = self.to_pil_image()
        return ImageSize(width=image.width, height=image.height)

    @property
    def is_large(self) -> bool:
        """Check if the base64 content exceeds the large file threshold."""
        return len(self.base64_content) > self.LARGE_FILE_THRESHOLD

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
        if self.file_path and self.file_path.exists():
            return Image.open(self.file_path)
        return Image.open(self.to_bytesio())

    def save_to_disk(self, directory: str | Path | None = None) -> "ImageData":
        """Save the image to disk and return a new ImageData with file_path set.

        Useful for large images to avoid keeping large base64 strings in memory
        when passing to models that support file paths.

        Args:
            directory: Directory to save the file in. Uses a temp directory if None.

        Returns:
            New ImageData with file_path set.
        """
        if directory is None:
            directory = Path(tempfile.mkdtemp())
        else:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)

        ext = self.mime_type.split("/")[-1] if "/" in self.mime_type else "bin"
        filename = self.filename or f"image.{ext}"
        file_path = directory / filename
        file_path.write_bytes(self.to_bytes())
        return self.model_copy(update={"file_path": file_path})

    def show(self) -> None:
        """Displays the image using PIL."""
        image = self.to_pil_image()
        image.show()
