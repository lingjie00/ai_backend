"""Example usage of converting multimodal inputs to structured input for LangChain."""

# %%
import tempfile

import fitz  # PyMuPDF
from PIL import Image

from ai_backend import MessageLoader

# Image data
with tempfile.NamedTemporaryFile(suffix=".png") as temp_image_file:
    # Create a simple image and save it to the temporary file
    image = Image.new("RGB", (100, 100), color="red")
    image.save(temp_image_file.name)

    image.show()  # This will open the image in the default image viewer

    # Use MessageLoader to convert the image file from disk to ImageData
    image_data = MessageLoader.convert_image_to_image_data(temp_image_file.name)
    print(image_data.model_dump_json(indent=2))

    # convert image data from memory to ImageData
    image_memory = Image.open(temp_image_file.name)
    image_data_memory = MessageLoader.convert_image_to_image_data(
        image_memory, mime_type=f"image/{image_memory.format}"
    )
    print(image_data_memory.model_dump_json(indent=2))

    # convert image data to content
    content = MessageLoader.convert_image_data_to_langchain_content(image_data)
    print(content)

    # we can also optimize the image data before converting to content
    optimized_content = MessageLoader.convert_image_data_to_langchain_content(
        image_data, max_dimension=5, optimize=True
    )
    print(optimized_content)


# PDF will also be converted to images and then to ImageData
with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_pdf_file:
    # Create a simple PDF with 2 pages
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello, this is a test PDF.")
    page = doc.new_page()
    page.insert_text((72, 72), "This is the second page of the PDF.")
    doc.save(temp_pdf_file.name)

    # Use MessageLoader to convert the PDF file from disk to ImageData
    image_data_list = MessageLoader.convert_pdf_to_image_data(temp_pdf_file.name)
    for image_data in image_data_list:
        image_data.show()  # Preview the image data
        print(image_data.model_dump_json(indent=2))
