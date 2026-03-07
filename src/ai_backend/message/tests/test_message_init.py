import tempfile
import unittest
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from ai_backend.message import (
    BoundingBox,
    ImageData,
    MessageLoader,
    is_base64_regex,
)
from ai_backend.message.image_loader import (
    OPTIMIZED_MIME_TYPE,
    annotate_image_with_bounding_box,
    decode_base64_to_Image,
    normalize_bbox,
    normalized_to_crop_tuple,
)
from ai_backend.message.image_model import ImageSize


class TestMessage(unittest.TestCase):
    def test_imports(self):
        self.assertTrue(callable(BoundingBox))
        self.assertTrue(callable(ImageData))
        self.assertTrue(callable(MessageLoader))

    def test_is_base64_regex(self):
        self.assertTrue(is_base64_regex("SGVsbG8gd29ybGQ="))
        self.assertFalse(is_base64_regex("Hello world"))

    def test_convert_image_to_image_data(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as temp_image_file:
            image = Image.new("RGB", (100, 100), color="red")
            image.save(temp_image_file.name)
            image_data = MessageLoader.convert_image_to_image_data(temp_image_file.name)
            self.assertIsInstance(image_data, ImageData)
            self.assertIsNotNone(image_data.base64_content)
            self.assertEqual(image_data.filename, Path(temp_image_file.name).name)
            self.assertEqual(image_data.mime_type, OPTIMIZED_MIME_TYPE)

    def test_convert_pdf_to_image_data(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_pdf_file:
            doc = fitz.open()
            doc.new_page()
            doc.save(temp_pdf_file.name)
            image_data_list = MessageLoader.convert_pdf_to_image_data(
                temp_pdf_file.name
            )
            self.assertIsInstance(image_data_list, list)
            self.assertIsInstance(image_data_list[0], ImageData)
            self.assertEqual(len(image_data_list), 1)

    def test_optimize_image_data(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as temp_image_file:
            image = Image.new("RGB", (2000, 2000), color="red")
            image.save(temp_image_file.name)
            image_data = MessageLoader.convert_image_to_image_data(temp_image_file.name)
            optimized_image_data = MessageLoader.optimize_image_data(
                image_data, max_dimension=500
            )
            self.assertIsInstance(optimized_image_data, ImageData)
            img = decode_base64_to_Image(optimized_image_data.base64_content)
            self.assertLessEqual(img.width, 500)
            self.assertLessEqual(img.height, 500)

    def test_convert_image_data_to_langchain_content(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as temp_image_file:
            image = Image.new("RGB", (100, 100), color="red")
            image.save(temp_image_file.name)
            image_data = MessageLoader.convert_image_to_image_data(temp_image_file.name)
            content = MessageLoader.convert_image_data_to_langchain_content(image_data)
            self.assertIn("type", content)
            self.assertEqual(content["type"], "image")

    def test_image_data_size(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as temp_image_file:
            image = Image.new("RGB", (64, 32), color="blue")
            image.save(temp_image_file.name)
            image_data = MessageLoader.convert_image_to_image_data(temp_image_file.name)
            self.assertEqual(image_data.size.width, 64)
            self.assertEqual(image_data.size.height, 32)

    def test_normalize_bbox(self):
        img_size = ImageSize(width=100, height=200)
        box = {"left": 10, "top": 20, "width": 30, "height": 40}
        bbox = normalize_bbox(box, img_size)
        self.assertAlmostEqual(bbox.x_min, 0.1)
        self.assertAlmostEqual(bbox.x_max, 0.4)
        self.assertAlmostEqual(bbox.y_min, 0.1)
        self.assertAlmostEqual(bbox.y_max, 0.3)

    def test_normalized_to_crop_tuple(self):
        img_size = ImageSize(width=100, height=200)
        bbox = BoundingBox.model_validate(
            {"x_min": 0.1, "y_min": 0.2, "x_max": 0.6, "y_max": 0.7}
        )
        crop = normalized_to_crop_tuple(bbox, img_size)
        self.assertEqual(crop, (10, 40, 60, 140))

    def test_annotate_image_with_bounding_box(self):
        image = Image.new("RGB", (100, 100), color="white")
        bbox = BoundingBox.model_validate(
            {"x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.9}
        )
        annotated = annotate_image_with_bounding_box(image, bbox, label="Q1")
        self.assertEqual(annotated.size, image.size)
        self.assertNotEqual(annotated.getpixel((10, 10)), (255, 255, 255))

    def test_annotate_image_with_invalid_bounding_box(self):
        image = Image.new("RGB", (100, 100), color="white")
        bbox = BoundingBox.model_validate(
            {"x_min": 0.5, "y_min": 0.2, "x_max": 0.5, "y_max": 0.8}
        )
        with self.assertRaises(ValueError):
            annotate_image_with_bounding_box(image, bbox)

    def test_message_loader_annotate_image_with_bounding_box(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as temp_image_file:
            image = Image.new("RGB", (100, 100), color="white")
            image.save(temp_image_file.name)
            image_data = MessageLoader.convert_image_to_image_data(temp_image_file.name)
            bbox = BoundingBox.model_validate(
                {"x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.9}
            )
            annotated = MessageLoader.annotate_image_with_bounding_box(
                image_data, bbox
            )
            annotated_image = decode_base64_to_Image(annotated.base64_content)
            self.assertEqual(annotated_image.size, (100, 100))
            self.assertNotEqual(
                annotated_image.getpixel((10, 10)), (255, 255, 255)
            )

    def test_message_loader_crop_image_with_bounding_box(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as temp_image_file:
            image = Image.new("RGB", (100, 200), color="green")
            image.save(temp_image_file.name)
            image_data = MessageLoader.convert_image_to_image_data(temp_image_file.name)
            bbox = BoundingBox.model_validate(
                {"x_min": 0.1, "y_min": 0.2, "x_max": 0.6, "y_max": 0.7}
            )
            cropped = MessageLoader.crop_image_with_bounding_box(image_data, bbox)
            cropped_image = decode_base64_to_Image(cropped.base64_content)
            self.assertEqual(cropped_image.size, (50, 100))

    def test_message_loader_crop_image_with_bounding_box_dict(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as temp_image_file:
            image = Image.new("RGB", (100, 200), color="green")
            image.save(temp_image_file.name)
            image_data = MessageLoader.convert_image_to_image_data(temp_image_file.name)
            bbox_dict = {"left": 10, "top": 20, "width": 30, "height": 40}
            cropped = MessageLoader.crop_image_with_bounding_box_dict(
                image_data, bbox_dict
            )
            cropped_image = decode_base64_to_Image(cropped.base64_content)
            self.assertEqual(cropped_image.size, (30, 40))


if __name__ == "__main__":
    unittest.main()
