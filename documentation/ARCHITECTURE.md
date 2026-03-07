# Architecture

## Overview

The ai_backend package provides small, composable utilities for loading prompts, images, and PDFs into data structures that can be consumed by LangChain clients. The public API is intentionally small and is centered around the prompt loader, the LangChain client wrapper, and the message loading helpers.

## Core Modules

- src/ai_backend/prompt_loader.py
  - Loads YAML prompts, applies default structure, and builds LangChain prompt templates.
- src/ai_backend/langchain_client.py
  - Wraps provider-specific models, applies runtime configuration, and exposes a consistent client API.
- src/ai_backend/message/
  - image_loader.py handles encoding, decoding, optimization, and image annotations.
  - image_model.py defines Pydantic models for image metadata and base64 content.
  - pdf_loader.py converts PDF inputs into image bytes for downstream processing.
  - MessageLoader also provides crop and annotation helpers over ImageData.

## Data Flow

1. Prompt YAML is loaded and normalized by PromptLoader.
2. LangChainClient reads model configuration and constructs provider-specific models.
3. MessageLoader converts files (images or PDFs) into ImageData for model input.
4. Image utilities normalize bounding boxes and produce optimized JPEG content when needed.

## Testing

Unit tests live under src/ai_backend/tests/ and src/ai_backend/message/tests/. They validate prompt loading behavior, LangChain client initialization, and message conversion utilities including image annotations and bounding box normalization.
