"""Defines the data model for LangChain-based AI interactions."""

from enum import Enum

from pydantic import BaseModel, Field


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    GEMINI = "gemini"


class AIModelConfig(BaseModel):
    """Configuration for AI models used in the application."""

    provider: LLMProvider = Field(
        ..., description="The AI provider to use, e.g., 'openai' or 'gemini'"
    )
    model_name: str = Field(
        ..., description="The name of the AI model to use, e.g., 'gpt-4'"
    )
    temperature: float = Field(
        0.2, description="Sampling temperature for response generation"
    )
    max_tokens: int = Field(
        2048, description="Maximum number of tokens in the generated response"
    )
