"""Implement a client based on LangChain for AI interactions."""

from __future__ import annotations

import os
import uuid
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable

try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
except ImportError:
    ChatGoogleGenerativeAI = Any  # type: ignore

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from ai_backend.langchain_model import AIModelConfig, LLMProvider
from ai_backend.prompt_loader import PromptLoader


class LangChainClient:
    """Client for interacting with AI models using LangChain."""

    # we store different models in case user wants to debug model responses
    _model: Runnable
    _chat_model: Runnable
    _structured_model: Runnable

    @property
    def model(self) -> Runnable:
        """Get the current model being used for interactions."""
        return self._model

    @model.setter
    def model(self, new_model: Runnable) -> None:
        """Set a new model for interactions."""
        self._model = new_model

    @property
    def chat_model(self) -> Runnable:
        """Get the chat model being used for interactions."""
        return self._chat_model

    @chat_model.setter
    def chat_model(self, new_chat_model: Runnable) -> None:
        """Set a new chat model for interactions."""
        self._chat_model = new_chat_model

    @property
    def structured_model(self) -> Runnable:
        """Get the structured output model, if available."""
        return self._structured_model

    @structured_model.setter
    def structured_model(self, new_structured_model: Runnable) -> None:
        """Set a new structured output model for interactions."""
        self._structured_model = new_structured_model

    def __init__(
        self,
        prompt_loader: PromptLoader,
        model_name: str,
        additional_prompts: list[BaseMessage | tuple[str, str]] = list(),
        api_key: str = "",
        structured_output_model: type[BaseModel] | None = None,
    ) -> None:
        """Initialize the LangChain client with a given model configuration.

        Args:
            prompt_loader: An instance of PromptLoader for loading prompts.
            model_name: The name of the model to use. This should correspond to
                a YAML file in the prompt directory.
        """
        self.session_id = str(uuid.uuid4())

        self.prompt_loader = prompt_loader
        model_config = self.prompt_loader.load_prompt_yaml(model_name)
        self.model_config = AIModelConfig.model_validate(model_config["model_config"])
        self.prompt_template = self.prompt_loader.load_chat_prompt_template(
            model_name, additional_prompts=additional_prompts
        )
        vanila_chat_model = self._create_client(api_key)
        self.chat_model = self.prompt_template | vanila_chat_model
        if structured_output_model is not None:
            self.structured_model = vanila_chat_model.with_structured_output(
                structured_output_model
            )
            self.model = self.prompt_template | self.structured_model
        else:
            self.model = self.chat_model

    def get_runtime_config(self) -> RunnableConfig:
        """Get the runtime configuration for the client."""
        return {
            "configurable": {
                "session_id": self.session_id,
            }
        }

    def _get_client_kwargs(self) -> dict[str, Any]:
        """Get the keyword arguments for initializing the AI client."""
        client_kwargs = {
            "model": self.model_config.model_name,
            "temperature": self.model_config.temperature,
            "max_tokens": self.model_config.max_tokens,
        }
        return client_kwargs

    def _create_google_client(
        self, api_key: str
    ) -> ChatGoogleGenerativeAI:  # # type: ignore[reportInvalidTypeForm]
        """Create a Google Gemini client based on the model configuration."""
        if ChatGoogleGenerativeAI is None:
            raise ImportError(
                "ChatGoogleGenerativeAI is not available."
                "Please install langchain_google_genai."
            )
        # If API key is not provided, try to get it from environment variable
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY", "")
        if self.model_config.provider == LLMProvider.GEMINI:
            client_kwargs = self._get_client_kwargs()
            client_kwargs["api_key"] = api_key
            client_kwargs["convert_system_message_to_human"] = True

            return ChatGoogleGenerativeAI(
                **client_kwargs
            )  # # type: ignore[reportInvalidTypeForm]
        else:
            raise ValueError(
                f"Unsupported provider: {self.model_config.provider}. "
                "Only 'gemini' is supported for Google client."
            )

    def _create_client(self, api_key: str = "") -> Any:
        """Create an AI client based on the model configuration."""
        if self.model_config.provider == LLMProvider.GEMINI:
            return self._create_google_client(api_key)
        else:
            raise ValueError(
                f"Unsupported provider: {self.model_config.provider}. "
                "Only 'gemini' is supported in this implementation."
            )
