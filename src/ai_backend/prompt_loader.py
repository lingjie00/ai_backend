"""Loads and manages AI prompts from a specified directory."""
import datetime
import os
from pathlib import Path
from typing import Any, Dict, TypeAlias, Union, cast

import yaml
from langchain_core.prompts import ChatPromptTemplate

# Wide type for maximum compatibility
PathLike: TypeAlias = Union[str, os.PathLike[str]]

PROMPT_KEY = "prompts"
DEFAULT_PROMPT_STRUCTURE = {
    "metadata": {
        "name": "The name of the prompt.",
        "description": "A brief description of the prompt's purpose.",
        "version": "The version of the prompt format (e.g., '1.0').",
        "last_updated": f"{datetime.date.today()}",
        "author": "The name of the person or organization that created the prompt.",
        "changelog": [
            {
                "version": "1.0",
                "date": f"{datetime.date.today()}",
                "changes": "Initial version of the prompt.",
            }
        ],
    },
    "model_config": {
        "provider": "The AI provider to use (e.g., 'openai', 'azure').",
        "model_name": "The name of the AI model to use (e.g., 'gpt-3.5-turbo').",
        "temperature": "A float value between 0 and 1 controlling outcome randomness.",
        "max_tokens": "The maximum number of tokens to generate in the response.",
    },
    "examples": [
        {
            "input": "An example input for the prompt.",
            "output": "The expected output for the example input.",
        }
    ],
    PROMPT_KEY: [
        {"role": "system", "content": "System instructions for the AI model."},
    ],
}


class PromptLoader:
    """Class responsible for loading and managing AI prompts from YAML files.

    User can import this in the project and use it to load prompts from a
    specified directory. Each prompt is expected to be a YAML file that can be
    parsed into a dictionary.
    """

    def __init__(self, prompt_directory: PathLike) -> None:
        """Initialize the PromptLoader with a specified directory.

        Args:
            prompt_directory: Path to the directory containing prompt YAML files.
        """
        self.prompt_directory = Path(prompt_directory)

    def get_prompt_path(self, prompt_name: str) -> Path:
        """Get the full path to a prompt YAML file.

        Args:
            prompt_name: name of the YAML prompt file.

        Returns:
            Full Path object pointing to the prompt YAML file.
        """
        if not prompt_name.endswith(".yaml"):
            prompt_name += ".yaml"
        return self.prompt_directory / prompt_name

    def load_prompt_yaml(self, prompt_name: str) -> Dict[str, Any]:
        """Load a prompt YAML file from disk.

        Args:
            prompt_name: name of the YAML prompt file.

        Returns:
            Parsed YAML content as a dictionary.
        """
        if not prompt_name.endswith(".yaml"):
            prompt_name += ".yaml"
        with open(
            self.get_prompt_path(prompt_name), "r", encoding="utf-8"
        ) as prompt_file:
            return cast(Dict[str, Any], yaml.safe_load(prompt_file))

    def save_prompt_yaml(self, prompt_name: str, prompt_data: Dict[str, Any]) -> Path:
        """Save a prompt dictionary to a YAML file.

        Args:
            prompt_name: name of the YAML prompt file to save.
            prompt_data: dictionary containing the prompt data to save.
        """
        with open(
            self.get_prompt_path(prompt_name), "w", encoding="utf-8"
        ) as prompt_file:
            yaml.safe_dump(prompt_data, prompt_file, sort_keys=False)
        return self.get_prompt_path(prompt_name)

    def create_prompt(self, prompt_name: str) -> Path:
        """Create a new prompt YAML file with the default structure.

        Args:
            prompt_name: name of the YAML prompt file to create.
        """
        self.save_prompt_yaml(prompt_name, DEFAULT_PROMPT_STRUCTURE)
        return self.get_prompt_path(prompt_name)

    def load_chat_prompt_template(
        self, prompt_name: str, additional_prompts: list = list()
    ) -> ChatPromptTemplate:
        """Load a prompt YAML file and convert it to a ChatPromptTemplate.

        Args:
            prompt_name: name of the YAML prompt file to load.
            additional_prompts: list of additional prompts to include in the template.

        Returns:
            A ChatPromptTemplate object created from the loaded YAML prompt.
        """
        prompt_data = self.load_prompt_yaml(prompt_name)
        messages = prompt_data.get(PROMPT_KEY, [])
        if additional_prompts:
            messages.extend(additional_prompts)
        return ChatPromptTemplate.from_messages(messages)
