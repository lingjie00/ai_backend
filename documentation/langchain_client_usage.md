# Using the LangChainClient

The `LangChainClient` provides a high-level interface for interacting with large language models (LLMs) using the LangChain framework. It simplifies the process of loading prompts, configuring models, and handling different output formats, including structured data.

## Basic Setup

To begin, you need a `PromptLoader` instance and the name of a model configuration file. The model configuration file is a YAML file that specifies the provider, model name, and other parameters.

```python
from ai_backend import LangChainClient, PromptLoader

# The loader needs a directory where your prompt YAML files are stored.
# For this example, we assume 'example_model.yaml' is in the current directory.
loader = PromptLoader(".")

# Initialize the client
client = LangChainClient(
    prompt_loader=loader,
    model_name="example_model",
)

# The client exposes a 'chat_model' for standard chat interactions.
chat_output = client.chat_model.invoke({"context": "Tell me a joke."})
chat_output.pretty_print()
```

## Structured Output with Pydantic

A key feature of `LangChainClient` is its ability to produce structured output that conforms to a Pydantic model. This is incredibly useful for applications that require reliable, schema-compliant data from the LLM.

### 1. Define Your Pydantic Model

First, define the data structure you expect from the model.

```python
from pydantic import BaseModel

class StoryBoard(BaseModel):
    """Data model representing a storyboard for a story generation task."""
    title: str
    setting: str
    characters_name: str
```

### 2. Initialize the Client with the Model

When creating the `LangChainClient`, pass your Pydantic model to the `structured_output_model` argument. You can also include `additional_prompts` to customize the interaction.

```python
client = LangChainClient(
    prompt_loader=loader,
    model_name="example_model",
    structured_output_model=StoryBoard,
    # You can add more context or instructions to the prompt
    additional_prompts=[("user", "{context}")],
)

# The primary 'model' will now produce structured output
model = client.model
```

### 3. Invoke the Model

When you invoke the model, it will return a dictionary that can be validated against your Pydantic model.

```python
context = {"context": "Write a short story about a brave knight in a magical kingdom."}
output = model.invoke(input=context)

# Validate the output against your Pydantic model
output_class = StoryBoard.model_validate(output)

# Now you have a Pydantic model instance
print(output_class.model_dump_json(indent=2))
```

This produces a clean, structured JSON output:

```json
{
  "title": "The Knight of the Glimmering Shield",
  "setting": "The Whispering Woods of Eldoria",
  "characters_name": "Sir Kaelan"
}
```

By integrating Pydantic models, the `LangChainClient` ensures that you receive predictable and usable data from the language model, making it easier to build robust AI-powered applications.
