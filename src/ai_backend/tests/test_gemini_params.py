"""Unit tests for Gemini-only model parameters (thinking_level, media_resolution)."""

import unittest
from unittest.mock import MagicMock, patch

from langchain_google_genai import MediaResolution as GoogleMediaResolution
from pydantic import ValidationError

from ai_backend.langchain_client import LangChainClient
from ai_backend.langchain_model import AIModelConfig, MediaResolution, ThinkingLevel
from ai_backend.prompt_loader import PromptLoader


class TestAIModelConfigGeminiParams(unittest.TestCase):
    """Test validation of Gemini-only fields on AIModelConfig."""

    def test_defaults_to_none(self) -> None:
        """Test that both fields are unset by default."""
        config = AIModelConfig.model_validate(
            {"provider": "gemini", "model_name": "gemini-3.8-flash"}
        )
        self.assertIsNone(config.thinking_level)
        self.assertIsNone(config.media_resolution)

    def test_parses_values_for_gemini_providers(self) -> None:
        """Test that string values parse into enums for gemini and vertex_ai."""
        for provider in ("gemini", "vertex_ai"):
            config = AIModelConfig.model_validate(
                {
                    "provider": provider,
                    "model_name": "gemini-3.8-flash",
                    "thinking_level": "low",
                    "media_resolution": "high",
                }
            )
            self.assertEqual(config.thinking_level, ThinkingLevel.LOW)
            self.assertEqual(config.media_resolution, MediaResolution.HIGH)

    def test_rejects_unknown_values(self) -> None:
        """Test that values outside the enums are rejected."""
        for field, value in (("thinking_level", "max"), ("media_resolution", "ultra")):
            with self.assertRaises(ValidationError):
                AIModelConfig.model_validate(
                    {
                        "provider": "gemini",
                        "model_name": "gemini-3.8-flash",
                        field: value,
                    }
                )

    def test_rejects_gemini_params_for_other_providers(self) -> None:
        """Test that Gemini-only fields raise for non-Gemini providers."""
        for provider in ("openai", "anthropic", "azure_openai"):
            for field, value in (
                ("thinking_level", "low"),
                ("media_resolution", "high"),
            ):
                with self.assertRaises(ValidationError) as cm:
                    AIModelConfig.model_validate(
                        {"provider": provider, "model_name": "m", field: value}
                    )
                self.assertIn(field, str(cm.exception))


class TestGeminiParamsPassedToClient(unittest.TestCase):
    """Test that Gemini-only fields reach ChatGoogleGenerativeAI."""

    def setUp(self) -> None:
        """Set up a mock PromptLoader."""
        self.mock_prompt_loader = MagicMock(spec=PromptLoader)
        self.mock_prompt_loader.load_chat_prompt_template.return_value = MagicMock()

    def _set_config(self, **extra: object) -> None:
        """Set the model_config returned by the mock prompt loader.

        Args:
            **extra: Extra model_config fields to merge in.
        """
        self.mock_prompt_loader.load_prompt_yaml.return_value = {
            "model_config": {
                "provider": "vertex_ai",
                "model_name": "gemini-3.8-flash",
                "temperature": 0.2,
                "max_tokens": 32768,
                **extra,
            }
        }

    @patch("ai_backend.langchain_client.ChatGoogleGenerativeAI")
    def test_passes_thinking_level_and_media_resolution(
        self, mock_chat_google: MagicMock
    ) -> None:
        """Test that set fields are forwarded with the SDK's types."""
        self._set_config(thinking_level="medium", media_resolution="high")

        LangChainClient(prompt_loader=self.mock_prompt_loader, model_name="m")

        kwargs = mock_chat_google.call_args.kwargs
        self.assertEqual(kwargs["thinking_level"], "medium")
        self.assertEqual(
            kwargs["media_resolution"], GoogleMediaResolution.MEDIA_RESOLUTION_HIGH
        )

    @patch("ai_backend.langchain_client.ChatGoogleGenerativeAI")
    def test_omits_unset_fields(self, mock_chat_google: MagicMock) -> None:
        """Test that unset fields are not sent, keeping the model's defaults."""
        self._set_config()

        LangChainClient(prompt_loader=self.mock_prompt_loader, model_name="m")

        kwargs = mock_chat_google.call_args.kwargs
        self.assertNotIn("thinking_level", kwargs)
        self.assertNotIn("media_resolution", kwargs)

    @patch("ai_backend.langchain_client.ChatGoogleGenerativeAI")
    def test_gemini_api_provider_passes_fields(
        self, mock_chat_google: MagicMock
    ) -> None:
        """Test that the gemini provider forwards the fields too."""
        self._set_config(provider="gemini", thinking_level="low")

        LangChainClient(
            prompt_loader=self.mock_prompt_loader, model_name="m", api_key="k"
        )

        self.assertEqual(mock_chat_google.call_args.kwargs["thinking_level"], "low")


if __name__ == "__main__":
    unittest.main()
