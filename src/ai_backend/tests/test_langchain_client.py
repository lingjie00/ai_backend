"""Unit tests for the LangChainClient."""

import os
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.runnables import Runnable
from pydantic import BaseModel

from ai_backend.langchain_client import LangChainClient
from ai_backend.prompt_loader import PromptLoader


class StoryBoard(BaseModel):
    """A simple Pydantic model for testing structured output."""

    title: str
    setting: str


class TestLangChainClient(unittest.TestCase):
    """Test suite for the LangChainClient."""

    def setUp(self):
        """Set up a mock PromptLoader and model configuration."""
        self.mock_prompt_loader = MagicMock(spec=PromptLoader)
        self.model_name = "test_model"
        self.model_config_data = {
            "model_config": {
                "provider": "gemini",
                "model_name": "gemini-pro",
                "temperature": 0.7,
                "max_tokens": 100,
            }
        }
        self.prompt_template = MagicMock()

        self.mock_prompt_loader.load_prompt_yaml.return_value = self.model_config_data
        self.mock_prompt_loader.load_chat_prompt_template.return_value = (
            self.prompt_template
        )

    @patch("ai_backend.langchain_client.ChatGoogleGenerativeAI")
    def test_init_with_google_client(self, mock_chat_google: MagicMock):
        """Test client initialization with a Google Gemini model."""
        mock_vanila_model = MagicMock()
        mock_chat_google.return_value = mock_vanila_model
        self.prompt_template.__or__.return_value = "prompt | model"

        client = LangChainClient(
            prompt_loader=self.mock_prompt_loader, model_name=self.model_name
        )

        self.mock_prompt_loader.load_prompt_yaml.assert_called_with(self.model_name)
        self.mock_prompt_loader.load_chat_prompt_template.assert_called_with(
            self.model_name, additional_prompts=[]
        )
        mock_chat_google.assert_called_once()
        self.assertEqual(client.model, "prompt | model")
        self.assertEqual(client.chat_model, "prompt | model")

    @patch("ai_backend.langchain_client.ChatGoogleGenerativeAI")
    def test_init_with_structured_output(self, mock_chat_google: MagicMock):
        """Test client initialization with a structured output model."""
        mock_vanila_model = MagicMock()
        mock_structured_model = MagicMock(spec=Runnable)
        mock_chat_google.return_value = mock_vanila_model
        mock_vanila_model.with_structured_output.return_value = mock_structured_model

        # Mock the chaining behavior
        self.prompt_template.__or__ = MagicMock()
        self.prompt_template.__or__.side_effect = [
            "chat_model_chain",
            "structured_model_chain",
        ]

        client = LangChainClient(
            prompt_loader=self.mock_prompt_loader,
            model_name=self.model_name,
            structured_output_model=StoryBoard,
        )

        mock_vanila_model.with_structured_output.assert_called_with(StoryBoard)
        self.assertEqual(client.model, "structured_model_chain")
        self.assertEqual(client.chat_model, "chat_model_chain")
        self.assertEqual(client.structured_model, mock_structured_model)

    def test_init_with_unsupported_provider(self):
        """Test that initialization fails with an unsupported provider."""
        self.model_config_data["model_config"]["provider"] = "unsupported"
        self.mock_prompt_loader.load_prompt_yaml.return_value = self.model_config_data

        with self.assertRaises(ValueError):
            LangChainClient(
                prompt_loader=self.mock_prompt_loader, model_name=self.model_name
            )

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"})
    @patch("ai_backend.langchain_client.ChatGoogleGenerativeAI")
    def test_google_client_creation_with_env_api_key(self, mock_chat_google: MagicMock):
        """Test that the Google client uses the API key from the environment."""
        client = LangChainClient(
            prompt_loader=self.mock_prompt_loader, model_name=self.model_name
        )
        client._create_google_client(api_key="")
        mock_chat_google.assert_called_with(
            model="gemini-pro",
            temperature=0.7,
            max_tokens=100,
            api_key="test_key",
            convert_system_message_to_human=True,
        )

    @patch("ai_backend.langchain_client.ChatGoogleGenerativeAI")
    def test_get_runtime_config(self, mock_chat_google: MagicMock):
        """Test that the runtime config includes a session_id."""
        mock_chat_google.return_value = MagicMock()
        client = LangChainClient(
            prompt_loader=self.mock_prompt_loader, model_name=self.model_name
        )
        config = client.get_runtime_config()
        self.assertIn("configurable", config)
        self.assertIn("session_id", config.get("configurable", {}))
        self.assertIsInstance(config.get("configurable", {}).get("session_id"), str)

    @patch("ai_backend.langchain_client.ChatOpenAI")
    def test_init_with_openai_client(self, mock_chat_openai: MagicMock):
        """Test client initialization with an OpenAI model."""
        self.model_config_data["model_config"]["provider"] = "openai"
        self.model_config_data["model_config"]["model_name"] = "gpt-4"
        self.mock_prompt_loader.load_prompt_yaml.return_value = self.model_config_data
        mock_vanila_model = MagicMock()
        mock_chat_openai.return_value = mock_vanila_model
        self.prompt_template.__or__.return_value = "prompt | model"

        client = LangChainClient(
            prompt_loader=self.mock_prompt_loader, model_name=self.model_name
        )

        mock_chat_openai.assert_called_once()
        self.assertEqual(client.model, "prompt | model")

    @patch("ai_backend.langchain_client.ChatAnthropic")
    def test_init_with_anthropic_client(self, mock_chat_anthropic: MagicMock):
        """Test client initialization with an Anthropic model."""
        self.model_config_data["model_config"]["provider"] = "anthropic"
        self.model_config_data["model_config"]["model_name"] = "claude-3-opus"
        self.mock_prompt_loader.load_prompt_yaml.return_value = self.model_config_data
        mock_vanila_model = MagicMock()
        mock_chat_anthropic.return_value = mock_vanila_model
        self.prompt_template.__or__.return_value = "prompt | model"

        client = LangChainClient(
            prompt_loader=self.mock_prompt_loader, model_name=self.model_name
        )

        mock_chat_anthropic.assert_called_once()
        self.assertEqual(client.model, "prompt | model")


if __name__ == "__main__":
    unittest.main()
