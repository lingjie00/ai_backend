# ai_backend

`ai_backend` is a Python package that provides a collection of commonly used backend utilities for building AI-powered applications. It simplifies tasks such as managing prompts, interacting with language models, and handling multimodal inputs.

## Installation

```bash
# Base install (Gemini support included)
pip install .

# With OpenAI support
pip install ".[openai]"

# With Anthropic support
pip install ".[anthropic]"

# Dev dependencies
pip install ".[dev]"
```

Or using `uv`:

```bash
uv sync
uv sync --extra openai
uv sync --extra anthropic
```

## Environment Variables

Set the API key for your chosen provider:

| Provider      | Environment Variable   |
| ------------- | ---------------------- |
| Google Gemini | `GOOGLE_API_KEY`       |
| OpenAI        | `OPENAI_API_KEY`       |
| Anthropic     | `ANTHROPIC_API_KEY`    |
| Azure OpenAI  | `AZURE_OPENAI_API_KEY` |

## Features

- **Prompt Management**: Easily create, load, and manage prompts from YAML files.
- **Language Model Interaction**: A high-level client for interacting with language models (Gemini, OpenAI, Anthropic, Azure OpenAI), with built-in support for structured (Pydantic) and unstructured outputs, streaming, and async.
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

# Get metadata from a prompt
metadata = loader.get_prompt_metadata("example_prompt.yaml")
```

### `LangChainClient`

The `LangChainClient` provides a high-level interface for interacting with large language models (LLMs) using the LangChain framework. It simplifies the process of loading prompts, configuring models, and handling different output formats.

Supported providers: `gemini`, `openai`, `anthropic`, `azure_openai`

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

# Invoke the model
context = {"context": "Write a short story about a brave knight."}
result = client.invoke(context)

# Stream output
for chunk in client.stream(context):
    print(chunk)

# Async invoke
# result = await client.ainvoke(context)
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

# For large images, save to disk to reduce memory usage
if image_data.is_large:
    image_data = image_data.save_to_disk("/tmp/images")
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
