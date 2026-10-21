"""Defines the data model for LangChain-based AI interactions."""

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    VERTEX_AI = "vertex_ai"


GEMINI_PROVIDERS = frozenset({LLMProvider.GEMINI, LLMProvider.VERTEX_AI})


class ThinkingLevel(Enum):
    """Reasoning effort for Gemini 3+ models.

    Supported levels vary by model; e.g. gemini-3.8-flash rejects ``minimal``.
    """

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MediaResolution(Enum):
    """Token budget per image/PDF page/video frame for Gemini models."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


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
    thinking_level: ThinkingLevel | None = Field(
        None,
        description="Gemini-only reasoning effort; unset uses the model default",
    )
    media_resolution: MediaResolution | None = Field(
        None,
        description="Gemini-only media input resolution; unset uses the model default",
    )

    @model_validator(mode="after")
    def _check_gemini_only_fields(self) -> "AIModelConfig":
        """Reject Gemini-only fields when the provider is not Gemini.

        Returns:
            The validated config.

        Raises:
            ValueError: If a Gemini-only field is set for another provider.
        """
        if self.provider in GEMINI_PROVIDERS:
            return self
        for field in ("thinking_level", "media_resolution"):
            if getattr(self, field) is not None:
                raise ValueError(
                    f"{field} is only supported for providers "
                    f"{sorted(p.value for p in GEMINI_PROVIDERS)}, "
                    f"got '{self.provider.value}'"
                )
        return self
