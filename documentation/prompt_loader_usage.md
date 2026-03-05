# Using the PromptLoader

The `PromptLoader` class is a utility for managing AI prompts stored in YAML files. It provides an easy way to create, load, save, and use prompts with the `langchain` library.

## Initialization

To get started, create an instance of `PromptLoader`, passing the directory where your prompts are stored.

```python
from pathlib import Path
from ai_backend import PromptLoader

# It's recommended to use a temporary directory for examples
import tempfile
prompt_directory = Path(tempfile.gettempdir())

loader = PromptLoader(prompt_directory)
```

## Creating a New Prompt

You can create a new prompt file with a default structure using the `create_prompt` method. This is useful for getting started with a new prompt.

```python
new_prompt_path = loader.create_prompt("example_prompt.yaml")

with new_prompt_path.open("r", encoding="utf-8") as f:
    print(f.read())
```

This will create an `example_prompt.yaml` file with the following content:

```yaml
metadata:
  name: The name of the prompt.
  description: A brief description of the prompt's purpose.
  version: "1.0"
  last_updated: "2024-01-01"
  author: The name of the person or organization that created the prompt.
  changelog:
    - version: "1.0"
      date: "2024-01-01"
      changes: Initial version of the prompt.
model_config:
  provider: The AI provider to use (e.g., 'openai', 'azure').
  model_name: The name of the AI model to use (e.g., 'gpt-3.5-turbo').
  temperature: A float value between 0 and 1 controlling outcome randomness.
  max_tokens: The maximum number of tokens to generate in the response.
examples:
  - input: An example input for the prompt.
    output: The expected output for the example input.
prompts:
  - role: system
    content: System instructions for the AI model.
```

## Loading and Modifying a Prompt

You can load a prompt from a YAML file into a Python dictionary, modify it, and save it back.

```python
import json

# Load the prompt
loaded_prompt = loader.load_prompt_yaml("example_prompt.yaml")
print(json.dumps(loaded_prompt, indent=2))

# Add a user message with a placeholder
loaded_prompt["prompts"].append(
    {
        "role": "user",
        "content": "What is the capital of {capital}?",
    }
)

# Save the modified prompt
loader.save_prompt_yaml("example_prompt.yaml", loaded_prompt)
```

## Creating a ChatPromptTemplate

The `PromptLoader` can directly create a `langchain` `ChatPromptTemplate` from a prompt file. This template can then be used to generate prompts for an AI model.

```python
prompt_template = loader.load_chat_prompt_template("example_prompt.yaml")
messages = prompt_template.invoke({"capital": "France"}).to_messages()

for message in messages:
    message.pretty_print()
```

This will output:

```
================================ System Message ================================

System instructions for the AI model.

================================== User Message ==================================

What is the capital of France?
```

## Using Placeholders for Additional Messages

You can also add placeholders for additional messages, which can be useful for injecting conversation history or other dynamic content.

```python
from langchain_core.prompts import MessagesPlaceholder

print("*" * 50 + "\nWith additional messages:\n")

additional_messages = [MessagesPlaceholder(variable_name="additional_messages")]
new_prompt_template = loader.load_chat_prompt_template(
    "example_prompt.yaml", additional_prompts=additional_messages
)

messages = new_prompt_template.invoke(
    {
        "capital": "France",
        "additional_messages": [{"role": "ai", "content": "I don't know"}],
    }
).to_messages()

for message in messages:
    message.pretty_print()
```

This will output:

```
**************************************************
With additional messages:

================================ System Message ================================

System instructions for the AI model.

================================== User Message ==================================

What is the capital of France?

=================================== AI Message ===================================

I don't know
```
