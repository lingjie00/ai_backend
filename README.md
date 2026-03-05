# ai_backend

`ai_backend` is a Python package that provides a collection of commonly used backend utilities for building AI-powered applications. It simplifies tasks such as managing prompts, interacting with language models, and handling multimodal inputs.

## Features

- **Prompt Management**: Easily create, load, and manage prompts from YAML files.
- **Language Model Interaction**: A high-level client for interacting with language models, with built-in support for structured (Pydantic) and unstructured outputs.
- **Multimodal Input Handling**: Utilities to convert images and PDFs into a standardized format for use in language model prompts.

## Core Components

### `PromptLoader`

The `PromptLoader` class is a utility for managing AI prompts stored in YAML files. It provides an easy way to create, load, save, and use prompts with the `langchain` library.

**Example: Creating and using a prompt**

```python
from pathlib import Path
from ai_backend import PromptLoader
import tempfile

prompt_directory = Path(tempfile.gettempdir())
loader = PromptLoader(prompt_directory)

# Create a new prompt file
new_prompt_path = loader.create_prompt("example_prompt.yaml")

# Load it as a LangChain ChatPromptTemplate
prompt_template = loader.load_chat_prompt_template("example_prompt.yaml")
messages = prompt_template.invoke({"input_variable": "some value"}).to_messages()
```

### `LangChainClient`

The `LangChainClient` provides a high-level interface for interacting with large language models (LLMs) using the LangChain framework. It simplifies the process of loading prompts, configuring models, and handling different output formats.

**Example: Getting structured output**

```python
from ai_backend import LangChainClient, PromptLoader
from pydantic import BaseModel

# Define your desired output structure
class StoryBoard(BaseModel):
    title: str
    setting: str
    characters_name: str

# Initialize the client with your prompt and structured output model
loader = PromptLoader(".") # Assuming 'example_model.yaml' is in the current directory
client = LangChainClient(
    prompt_loader=loader,
    model_name="example_model",
    structured_output_model=StoryBoard,
    additional_prompts=[("user", "{context}")],
)

# Invoke the model and get a validated Pydantic object
context = {"context": "Write a short story about a brave knight."}
output = client.model.invoke(input=context)
output_class = StoryBoard.model_validate(output)
print(output_class.model_dump_json(indent=2))
```

### `MessageLoader`

The `MessageLoader` class is a utility for loading and encoding files from various sources, such as images and PDFs, and converting them into a structured format that can be used in prompts for multimodal models.

**Example: Converting an image for a prompt**

```python
from ai_backend import MessageLoader

# Convert an image file to an ImageData object
image_data = MessageLoader.convert_image_to_image_data("path/to/your/image.png")

# Convert the ImageData to a format suitable for LangChain
langchain_content = MessageLoader.convert_image_data_to_langchain_content(image_data)
```

**Example: Converting a PDF to images**

```python
from ai_backend import MessageLoader

# Convert each page of a PDF to an ImageData object
image_data_list = MessageLoader.convert_pdf_to_image_data("path/to/your/document.pdf")
```

## Getting Started

1.  Install the package:
    ```bash
    pip install .
    ```
2.  Explore the examples in the `example/` directory to see the different components in action.
3.  Review the detailed documentation in the `documentation/` directory for in-depth usage guides.
