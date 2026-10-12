"""
Utility functions for the cookie cutter modelfile tool.
"""

from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.text import Text


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
            model_id = model_id[len(prefix):]
            break
    
    # Handle short aliases
    aliases = {
        "hf.co/": "huggingface.co/",
        "huggingface.co/": "huggingface.co/",
    }
    
    for alias, full in aliases.items():
        if model_id.startswith(alias):
            model_id = model_id[len(alias):]
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


def get_model_info_display(model_info: dict) -> Text:
    """Create a formatted text display of model information.
    
    Args:
        model_info: Dictionary containing model metadata.
        
    Returns:
        A rich Text object with formatted model information.
    """
    text = Text()
    text.append("\n", style="bold")
    text.append("Model Information\n", style="bold cyan")
    
    # ID
    model_id = model_info.get("id", "Unknown")
    text.append(f"\nID: ", style="dim")
    text.append(model_id, style="cyan")
    
    # Name
    name = model_info.get("name", "")
    text.append(f"\nName: ", style="dim")
    text.append(name, style="green")
    
    # Description
    description = model_info.get("description", "")
    if description:
        text.append(f"\nDescription:\n", style="dim")
        text.append(description[:200], style="dim gray")
        if len(description) > 200:
            text.append("...", style="dim")
    
    # Library
    library = model_info.get("library", "")
    if library:
        text.append(f"\nLibrary: ", style="dim")
        text.append(library, style="magenta")
    
    # Tags
    tags = model_info.get("tags", [])
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
    console = Console()
    
    if error:
        console.print(f"\n❌ Error: {message}", style="bold red")
        console.print(f"   {type(error).__name__}: {error}", style="red")
    else:
        console.print(f"\n❌ Error: {message}", style="bold red")
    
    console.print()
