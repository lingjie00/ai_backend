"""Unit tests for the PromptLoader class."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from ai_backend.prompt_loader import (
    DEFAULT_PROMPT_STRUCTURE,
    PROMPT_KEY,
    PromptLoader,
)


class TestPromptLoader(unittest.TestCase):
    """Test suite for the PromptLoader class."""

    def setUp(self):
        """Set up a temporary directory for testing."""
        self.test_dir = Path("/tmp/test_prompts")
        self.test_dir.mkdir(exist_ok=True)
        self.loader = PromptLoader(self.test_dir)

    def tearDown(self):
        """Clean up the temporary directory."""
        for f in self.test_dir.glob("*.yaml"):
            f.unlink()
        self.test_dir.rmdir()

    def test_get_prompt_path(self):
        """Test getting the full path to a prompt file."""
        prompt_name = "test_prompt.yaml"
        expected_path = self.test_dir / prompt_name
        self.assertEqual(self.loader.get_prompt_path(prompt_name), expected_path)

    def test_get_prompt_path_with_autofix(self):
        """Test getting the full path with automatic '.yaml' extension."""
        prompt_name = "test_prompt"
        expected_path = self.test_dir / (prompt_name + ".yaml")
        self.assertEqual(self.loader.get_prompt_path(prompt_name), expected_path)

    def test_load_prompt_yaml(self):
        """Test loading a YAML file from disk."""
        prompt_name = "test_prompt.yaml"
        prompt_data = {"key": "value"}
        with open(self.test_dir / prompt_name, "w", encoding="utf-8") as f:
            yaml.safe_dump(prompt_data, f)

        loaded_data = self.loader.load_prompt_yaml(prompt_name)
        self.assertEqual(loaded_data, prompt_data)

    def test_load_prompt_yaml_with_autofix(self):
        """Test loading a YAML file from disk with automatic '.yaml' extension."""
        prompt_name = "test_prompt"
        prompt_data = {"key": "value"}
        with open(self.test_dir / (prompt_name + ".yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(prompt_data, f)

        loaded_data = self.loader.load_prompt_yaml(prompt_name)
        self.assertEqual(loaded_data, prompt_data)

    def test_save_prompt_yaml(self):
        """Test saving a prompt dictionary to a YAML file."""
        prompt_name = "test_save.yaml"
        prompt_data = {"save_key": "save_value"}
        self.loader.save_prompt_yaml(prompt_name, prompt_data)

        with open(self.test_dir / prompt_name, encoding="utf-8") as f:
            saved_data = yaml.safe_load(f)
        self.assertEqual(saved_data, prompt_data)

        # check if the key order is preserved
        self.loader.save_prompt_yaml("test_order.yaml", {"b": 1, "a": 2})
        with open(self.test_dir / "test_order.yaml", encoding="utf-8") as f:
            content = f.read()
            self.assertTrue(content.startswith("b: 1"))

    def test_save_prompt_yaml_with_autofix(self):
        """Test saving a prompt dictionary to a YAML file.

        It will create automatic '.yaml' extension."""
        prompt_name = "test_save"
        prompt_data = {"save_key": "save_value"}
        self.loader.save_prompt_yaml(prompt_name, prompt_data)

        with open(self.test_dir / (prompt_name + ".yaml"), encoding="utf-8") as f:
            saved_data = yaml.safe_load(f)
        self.assertEqual(saved_data, prompt_data)

    def test_create_prompt(self):
        """Test creating a new prompt YAML file with the default structure."""
        prompt_name = "new_prompt.yaml"
        self.loader.create_prompt(prompt_name)

        loaded_data = self.loader.load_prompt_yaml(prompt_name)
        self.assertEqual(loaded_data, DEFAULT_PROMPT_STRUCTURE)

    @patch("ai_backend.prompt_loader.ChatPromptTemplate")
    def test_load_chat_prompt_template(self, mock_chat_prompt_template: MagicMock):
        """Test loading a prompt and converting it to a ChatPromptTemplate."""
        prompt_name = "chat_prompt.yaml"
        prompt_content = {
            PROMPT_KEY: [{"role": "system", "content": "You are a helpful assistant."}]
        }
        self.loader.save_prompt_yaml(prompt_name, prompt_content)

        self.loader.load_chat_prompt_template(prompt_name)
        mock_chat_prompt_template.from_messages.assert_called_with(
            prompt_content[PROMPT_KEY]
        )

    @patch("ai_backend.prompt_loader.ChatPromptTemplate")
    def test_load_chat_prompt_template_with_additional_prompts(
        self, mock_chat_prompt_template: MagicMock
    ):
        """Test loading a prompt with additional messages."""
        prompt_name = "chat_prompt_additional.yaml"
        prompt_content = {
            PROMPT_KEY: [{"role": "system", "content": "You are a helpful assistant."}]
        }
        additional_prompts = [{"role": "user", "content": "Hello there!"}]
        self.loader.save_prompt_yaml(prompt_name, prompt_content)

        self.loader.load_chat_prompt_template(
            prompt_name, additional_prompts=additional_prompts
        )
        expected_messages = prompt_content[PROMPT_KEY] + additional_prompts
        mock_chat_prompt_template.from_messages.assert_called_with(expected_messages)

    def test_load_prompt_yaml_file_not_found(self):
        """Test that loading a nonexistent YAML file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_prompt_yaml("nonexistent_prompt")

    def test_load_chat_prompt_template_missing_prompts_key(self):
        """Test loading a YAML with no 'prompts' key returns empty messages."""
        prompt_name = "no_prompts.yaml"
        self.loader.save_prompt_yaml(prompt_name, {"model_config": {}})
        with patch("ai_backend.prompt_loader.ChatPromptTemplate") as mock_cpt:
            self.loader.load_chat_prompt_template(prompt_name)
            mock_cpt.from_messages.assert_called_with([])

    def test_get_prompt_metadata(self):
        """Test extracting metadata from a prompt YAML file."""
        prompt_name = "meta_prompt.yaml"
        prompt_data = {
            "metadata": {"name": "Test", "version": "1.0"},
            PROMPT_KEY: [{"role": "system", "content": "Hello"}],
        }
        self.loader.save_prompt_yaml(prompt_name, prompt_data)
        metadata = self.loader.get_prompt_metadata(prompt_name)
        self.assertEqual(metadata["name"], "Test")
        self.assertEqual(metadata["version"], "1.0")

    def test_get_prompt_metadata_missing(self):
        """Test that get_prompt_metadata returns empty dict when no metadata."""
        prompt_name = "no_meta.yaml"
        self.loader.save_prompt_yaml(prompt_name, {PROMPT_KEY: []})
        metadata = self.loader.get_prompt_metadata(prompt_name)
        self.assertEqual(metadata, {})


if __name__ == "__main__":
    unittest.main()
