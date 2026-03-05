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
from ai_backend.message.image_loader import decode_base64_to_Image


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
            self.assertEqual(image_data.mime_type, "image/png")

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


if __name__ == "__main__":
    unittest.main()
