# Multimodal Inputs Usage

This document provides an overview of how to use the `MessageLoader` class to handle multimodal inputs, such as images and PDFs, and convert them into a structured format for use with LangChain.

## `MessageLoader` Class

The `MessageLoader` class is a utility for loading and encoding files from various sources. It provides methods to convert images and PDFs into `ImageData` objects, which can then be used to create LangChain content.

### Converting Images

The `MessageLoader.convert_image_to_image_data` method can be used to convert an image from a file path or an in-memory object into an `ImageData` object.

```python
from PIL import Image
from ai_backend import MessageLoader

# Convert an image from a file path
image_data_from_file = MessageLoader.convert_image_to_image_data("path/to/your/image.png")

# Convert an in-memory image object
in_memory_image = Image.open("path/to/your/image.png")
image_data_from_memory = MessageLoader.convert_image_to_image_data(
    in_memory_image, mime_type=f"image/{in_memory_image.format}"
)
```

### Converting PDFs

The `MessageLoader.convert_pdf_to_image_data` method can be used to convert a PDF from a file path or an in-memory object into a list of `ImageData` objects, one for each page of the PDF.

```python
from ai_backend import MessageLoader

# Convert a PDF from a file path
image_data_list = MessageLoader.convert_pdf_to_image_data("path/to/your/document.pdf")
```

### Creating LangChain Content

Once you have an `ImageData` object, you can use the `MessageLoader.convert_image_data_to_langchain_content` method to convert it into a format suitable for use with LangChain.

```python
from ai_backend import MessageLoader

# Assuming you have an ImageData object named `image_data`
langchain_content = MessageLoader.convert_image_data_to_langchain_content(image_data)
```

You can also optimize the image data before converting it to LangChain content by setting the `optimize` parameter to `True`.

```python
from ai_backend import MessageLoader

# Optimize the image data and then convert it to LangChain content
optimized_content = MessageLoader.convert_image_data_to_langchain_content(
    image_data, max_dimension=1024, optimize=True
)
```
