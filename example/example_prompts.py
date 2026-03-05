"""Provides the example prompts for testing and demonstration purposes."""

# %%
import json
import tempfile
from pathlib import Path

from langchain_core.prompts import MessagesPlaceholder

from ai_backend import PromptLoader

with tempfile.TemporaryDirectory() as temp_dir:
    loader = PromptLoader(Path(tempfile.gettempdir()))

    # save default prompt
    new_prompt_path = loader.create_prompt("example_prompt.yaml")

    with new_prompt_path.open("r", encoding="utf-8") as f:
        print(f.read())

    loaded_prompt = loader.load_prompt_yaml("example_prompt.yaml")
    print(json.dumps(loaded_prompt, indent=2))

    loaded_prompt["prompts"].append(
        {
            "role": "user",
            "content": "What is the capital of {capital}?",
        }
    )
    loader.save_prompt_yaml("example_prompt.yaml", loaded_prompt)

    prompt_template = loader.load_chat_prompt_template("example_prompt.yaml")
    messages = prompt_template.invoke({"capital": "France"}).to_messages()
    for message in messages:
        message.pretty_print()

    # We can add in message holders as well
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
