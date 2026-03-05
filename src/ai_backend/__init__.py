"""AI backend package that provides generic AI-related functionalities and services."""

from ai_backend.langchain_client import LangChainClient
from ai_backend.prompt_loader import PromptLoader

__all__ = ["PromptLoader", "LangChainClient"]
