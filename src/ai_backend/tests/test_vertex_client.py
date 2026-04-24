"""Unit tests for the Vertex AI client implementation."""

import unittest
from unittest.mock import MagicMock, patch

from ai_backend.langchain_client import LangChainClient
from ai_backend.prompt_loader import PromptLoader


class TestVertexClient(unittest.TestCase):
    """Test suite for the Vertex AI client."""

    def setUp(self):
        """Set up a mock PromptLoader and model configuration."""
        self.mock_prompt_loader = MagicMock(spec=PromptLoader)
        self.model_name = "test_vertex_model"
        self.model_config_data = {
            "model_config": {
                "provider": "vertex_ai",
                "model_name": "gemini-1.5-pro",
                "temperature": 0.5,
                "max_tokens": 500,
            }
        }
        self.prompt_template = MagicMock()

        self.mock_prompt_loader.load_prompt_yaml.return_value = self.model_config_data
        self.mock_prompt_loader.load_chat_prompt_template.return_value = (
            self.prompt_template
        )

    @patch("ai_backend.langchain_client.ChatGoogleGenerativeAI")
    def test_init_with_vertex_client(self, mock_chat_google: MagicMock):
        """Test client initialization with a Vertex AI model."""
        mock_vanila_model = MagicMock()
        mock_chat_google.return_value = mock_vanila_model
        self.prompt_template.__or__.return_value = "prompt | model"

        client = LangChainClient(
            prompt_loader=self.mock_prompt_loader, model_name=self.model_name
        )

        self.mock_prompt_loader.load_prompt_yaml.assert_called_with(self.model_name)
        mock_chat_google.assert_called_once_with(
            model="gemini-1.5-pro",
            temperature=0.5,
            max_tokens=500,
            vertexai=True,
        )
        self.assertEqual(client.model, "prompt | model")

    @patch("ai_backend.langchain_client.ChatGoogleGenerativeAI", None)
    def test_init_without_google_package(self):
        """Test that initialization fails when langchain-google-genai is missing."""
        with self.assertRaises(ImportError) as cm:
            LangChainClient(
                prompt_loader=self.mock_prompt_loader, model_name=self.model_name
            )
        self.assertIn("ChatGoogleGenerativeAI is not available", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
