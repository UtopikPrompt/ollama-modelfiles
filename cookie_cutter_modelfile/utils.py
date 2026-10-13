"""
Utility functions for the cookie cutter modelfile tool.
"""

from __future__ import annotations

import re
import sys
from typing import Any, Optional

from rich.console import Console
from rich.text import Text


def parse_huggingface_url(url: str) -> Optional[str]:
    """Parse a Hugging Face URL and extract the model ID.

    Args:
        url: A Hugging Face URL (e.g., https://huggingface.co/model-id)

    Returns:
        The extracted model ID, or None if parsing fails.
    """
    if not url or not isinstance(url, str):
        return None

    url = url.strip()

    # First try to match API endpoint (has /api/models/)
    if "/api/models/" in url:
        # Extract the path after /api/models/
        path_after_api = url.split("/api/models/")[1]
        # Strip trailing slash and get the last segment (the model ID)
        path_after_api = path_after_api.rstrip("/")
        model_id = path_after_api.split("/")[-1]
        return model_id

    # Then match direct URLs (everything after domain)
    direct_pattern = re.compile(r"^(https?://(?:huggingface\.co|hf\.co)/)(.+?)$")
    direct_match = direct_pattern.match(url)
    if direct_match:
        # Extract everything between the domain and any query parameters
        full_path = direct_match.group(2)
        # Remove query parameters if present
        if "?" in full_path:
            full_path = full_path.split("?")[0]
        # Strip known sub-path segments (repo tree/blob/resolve views and
        # any trailing path after the repo ID, e.g. /tree/main).
        for segment in ("/resolve", "/tree", "/blob"):
            if segment in full_path:
                full_path = full_path.split(segment)[0]
        # The repo ID is the first two path segments (namespace/repo_name).
        parts = full_path.split("/")
        return "/".join(parts[:2])

    return None


def normalize_model_id(model_id: str) -> Optional[str]:
    """Normalize a model ID by removing common prefixes and variations.

    Args:
        model_id: The model identifier to normalize.

    Returns:
        A normalized model ID string, or None if the input is invalid.
    """
    if not model_id or not isinstance(model_id, str):
        return None

    # Remove common prefixes
    model_id = model_id.strip()

    # Handle common HF API prefixes
    prefixes = [
        "https://huggingface.co/",
        "https://huggingface.co/",
        "https://hf.co/",
        "https://hf.co/",
        "https://huggingface.co/",
        "huggingface.co/",
        "hf.co/",
    ]

    for prefix in prefixes:
        if model_id.startswith(prefix):
            model_id = model_id[len(prefix) :]
            break

    # Handle short aliases
    aliases = {
        "hf.co/": "huggingface.co/",
        "huggingface.co/": "huggingface.co/",
    }

    for alias, full in aliases.items():
        if model_id.startswith(alias):
            model_id = model_id[len(alias) :]
            break

    return model_id


def format_context_length(length: int) -> str:
    """Format context length for display.

    Args:
        length: The context length in tokens.

    Returns:
        A formatted string representation.
    """
    if length >= 1_000_000:
        return f"{length / 1_000_000:.2f}M"
    elif length >= 1000:
        return f"{length / 1000:.2f}K"
    else:
        return str(length)
    return str(length)


def _model_info_field(model_info: Any, field: str, default: Any = None) -> Any:
    """Read a field from a model info source.

    Supports both a plain ``dict`` (accessed via ``.get``) and a
    ``huggingface_hub.ModelInfo`` object (accessed via attributes).

    Args:
        model_info: Either a dict of model metadata or a ModelInfo object.
        field: The name of the field to read.
        default: Value to return if the field is missing or empty.

    Returns:
        The field value or ``default``.
    """
    if model_info is None:
        return default

    # dict-like access (e.g. get_model_info)
    if hasattr(model_info, "get"):
        value = model_info.get(field, default)
        return value if value is not None else default

    # attribute access (e.g. ModelInfo object)
    value = getattr(model_info, field, default)
    return value if value is not None else default


def get_model_info_display(model_info: Any) -> Text:
    """Create a formatted text display of model information.

    Args:
        model_info: Either a dict or a ``huggingface_hub.ModelInfo`` object
            containing model metadata.

    Returns:
        A rich Text object with formatted model information.
    """
    text = Text()
    text.append("\n", style="bold")
    text.append("Model Information\n", style="bold cyan")

    # ID
    model_id = _model_info_field(model_info, "id", "Unknown")
    text.append(f"\nID: ", style="dim")
    text.append(model_id, style="cyan")

    # Name
    name = _model_info_field(model_info, "name", "")
    text.append(f"\nName: ", style="dim")
    text.append(name, style="green")

    # Description
    description = _model_info_field(model_info, "description", "")
    if description:
        text.append(f"\nDescription:\n", style="dim")
        text.append(str(description)[:200], style="dim gray")
        if len(str(description)) > 200:
            text.append("...", style="dim")

    # Library
    library = _model_info_field(model_info, "library", "")
    if library:
        text.append(f"\nLibrary: ", style="dim")
        text.append(library, style="magenta")

    # Tags
    tags = _model_info_field(model_info, "tags", [])
    if tags:
        tag_str = ", ".join(tags[:5])
        if len(tags) > 5:
            tag_str += f" (+{len(tags) - 5} more)"
        text.append(f"\nTags: ", style="dim")
        text.append(tag_str, style="blue")

    return text


def print_error(message: str, error: Optional[Exception] = None) -> None:
    """Print a formatted error message.

    Args:
        message: The error message to display.
        error: The exception object (optional).
    """
    console = Console(file=sys.stdout)

    if error:
        console.print(f"\n❌ Error: {message}", style="bold red")
        console.print(f"   {type(error).__name__}: {error}", style="red")
    else:
        console.print(f"\n❌ Error: {message}", style="bold red")

    console.print()
