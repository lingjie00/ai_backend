```markdown
## [2026-03-05] Decision: Add Unit Tests for `PromptLoader`

- **Context**: The user requested to check the implementation and add unit tests for the `prompt_loader.py` file.
- **Assumption**: The existing implementation of `PromptLoader` is functionally correct, and the primary goal is to add test coverage. I will create a standard unit test suite using Python's `unittest` module.
- **Statistical/Technical Rationale**: Using the `unittest` framework is a standard practice in Python for ensuring code quality and correctness. It allows for isolated testing of each component of the `PromptLoader` class. Mocking filesystem operations and dependencies like `ChatPromptTemplate` ensures that the tests are fast, reliable, and independent of external factors. This aligns with the "Worth-it" philosophy of using robust, standard tools for foundational tasks.
- **Anti-Regression**: The tests depend on the structure of `DEFAULT_PROMPT_STRUCTURE` and the method signatures in `PromptLoader`. Any changes to these must be reflected in the tests to avoid breakage.
```

## [2026-03-05] Decision: Project Structure and Testing Strategy

- **Context**: The user requested to add human documentation and unit tests for the `ai_backend` project.
- **Assumption**: I assumed that the user wants comprehensive unit tests covering the main functionalities of the `MessageLoader` class, including image and PDF conversion, image optimization, and LangChain content conversion. I also assumed that a new documentation file for multimodal inputs is required.
- **Statistical/Technical Rationale**: Based on the provided file `example/example_multimodal_inputs.py`, the core functionality revolves around the `MessageLoader` class. Therefore, creating dedicated documentation and unit tests for this class is the most logical approach to fulfill the user's request. The "Worth-it" philosophy suggests that creating clear documentation and robust tests is a high-value activity for the long-term health of the project.
- **Anti-Regression**: The unit tests should be maintained and updated whenever the `MessageLoader` class is modified. The documentation should also be kept in sync with any changes to the public API of the `MessageLoader` class.

## [2026-03-07] Decision: Image Utilities Test Coverage

- **Context**: New image utility helpers and ImageData.size were added, and the user requested unit tests for staged changes.
- **Assumption**: No additional task specification files exist under documentation/task_specs, so the update focuses on tests and architecture documentation.
- **Statistical/Technical Rationale**: Unit tests that validate bounding box normalization, crop conversion, annotation safety, and size extraction reduce regressions in image preprocessing flows. Creating ARCHITECTURE.md provides a stable reference point for module responsibilities.
- **Anti-Regression**: The tests assert explicit normalized values and pixel changes; changes to normalization precision or annotation behavior must update these expectations.

## [2026-03-07] Decision: Crop Helper ImageSize Fix

- **Context**: MessageLoader crop helpers were updated and needed new tests.
- **Assumption**: The ImageSize wrapper is the intended input type for normalized cropping utilities; using raw image.size tuples is a bug.
- **Statistical/Technical Rationale**: Ensuring consistent data models avoids silent type mismatches that would only surface at runtime during crop operations.
- **Anti-Regression**: Always build ImageSize from PIL dimensions before passing to normalized crop utilities.
