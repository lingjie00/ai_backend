"""Integration test: end-to-end prompt → client → invoke → structured output."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from ai_backend.langchain_client import LangChainClient
from ai_backend.prompt_loader import PromptLoader


class StoryOutput(BaseModel):
    title: str
    summary: str


class TestIntegration(unittest.TestCase):
    """End-to-end integration test with mocked LLM."""

    def setUp(self):
        self.test_dir = Path("/tmp/test_integration_prompts")
        self.test_dir.mkdir(exist_ok=True)
        self.loader = PromptLoader(self.test_dir)
        # Create a real prompt file
        import yaml

        prompt_data = {
            "model_config": {
                "provider": "gemini",
                "model_name": "gemini-pro",
                "temperature": 0.5,
                "max_tokens": 200,
            },
            "prompts": [
                {"role": "system", "content": "You are a story writer."},
            ],
        }
        with open(self.test_dir / "story_model.yaml", "w") as f:
            yaml.safe_dump(prompt_data, f)

    def tearDown(self):
        for f in self.test_dir.glob("*.yaml"):
            f.unlink()
        self.test_dir.rmdir()

    @patch("ai_backend.langchain_client.ChatGoogleGenerativeAI")
    def test_end_to_end_structured_output(self, mock_chat_google: MagicMock):
        """Test full flow: load prompt → create client → invoke → get structured output."""
        # Set up mock model
        mock_model = MagicMock()
        mock_chat_google.return_value = mock_model

        mock_structured_model = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured_model

        # Mock the chain result
        expected_output = StoryOutput(title="The Quest", summary="A brave adventure")
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = expected_output
        mock_chain.stream.return_value = iter([expected_output])

        # Mock __or__ to return our mock chain
        from langchain_core.prompts import ChatPromptTemplate

        with patch.object(ChatPromptTemplate, "__or__", return_value=mock_chain):
            client = LangChainClient(
                prompt_loader=self.loader,
                model_name="story_model",
                structured_output_model=StoryOutput,
                additional_prompts=[("user", "{topic}")],
            )

            # Test invoke
            result = client.invoke({"topic": "dragons"})
            self.assertEqual(result, expected_output)

            # Test stream
            chunks = list(client.stream({"topic": "dragons"}))
            self.assertEqual(len(chunks), 1)

    @patch("ai_backend.langchain_client.ChatGoogleGenerativeAI")
    def test_end_to_end_chat_output(self, mock_chat_google: MagicMock):
        """Test full flow without structured output."""
        mock_model = MagicMock()
        mock_chat_google.return_value = mock_model

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "A story about brave knights"

        from langchain_core.prompts import ChatPromptTemplate

        with patch.object(ChatPromptTemplate, "__or__", return_value=mock_chain):
            client = LangChainClient(
                prompt_loader=self.loader,
                model_name="story_model",
            )

            result = client.invoke({"topic": "knights"})
            self.assertEqual(result, "A story about brave knights")


if __name__ == "__main__":
    unittest.main()
