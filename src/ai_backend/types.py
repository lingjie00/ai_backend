"""Custom data types for the AI backend."""

import os
from typing import TypeAlias, Union

# Wide type for maximum compatibility
PathLike: TypeAlias = Union[str, os.PathLike[str]]
