"""
Ollama model converter.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

from rich.console import Console

from .modelfile import ModelfileGenerator


class OllamaConverter:
    """Converts Hugging Face models to Ollama format."""

    def __init__(self):
        """Initialize the Ollama converter."""
        self.console = Console(file=sys.stdout)

    def convert_to_ollama(
        self, model_id: str, context_length: int, params: Dict[str, Any]
    ) -> str:
        """Convert a Hugging Face model to Ollama format.

        Args:
            model_id: The Hugging Face model identifier.
            context_length: The context window size in tokens.
            params: Dictionary of parameters.

        Returns:
            Ollama model definition JSON string.
        """
        from .huggingface import HFClient

        hf_client = HFClient()
        model_info = hf_client.fetch_model(model_id)

        if model_info is None:
            self.console.print(
                "[red]Error: Model not found. Fetching model card...[/red]"
            )
            model_info = self._fetch_from_model_card(model_id)

        # Convert ModelInfo object to a plain dict for JSON serialization
        # and to keep the converter methods working with dict-style access.
        if hasattr(model_info, "base_models"):
            # ModelInfo proto object (from fetch_model)
            info_dict = {
                "id": model_info.id,
                "name": getattr(model_info, "name", model_info.id),
                "author": getattr(model_info, "author", ""),
                "tags": list(model_info.tags),
            }
            if model_info.base_models is not None:
                info_dict["base_models"] = list(model_info.base_models)
        else:
            # Already a plain dict (e.g. from _fetch_from_model_card)
            info_dict = model_info

        # Build the model definition
        model_def = {
            "name": self._get_name(info_dict),
            "modelfile": self._get_modelfile(model_id, context_length, params),
            "parameters": self._get_parameters(info_dict, params),
        }

        return self._format_json(model_def)

    def _fetch_from_model_card(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Fetch model information from model card.

        Args:
            model_id: The Hugging Face model identifier.

        Returns:
            Model information dictionary or None.
        """
        from .huggingface import HFClient

        hf_client = HFClient()
        model_card = hf_client.get_model_card(model_id)

        if model_card is None:
            return None

        # Parse model card to extract information
        import re

        # Extract model name
        name_match = re.search(r"## \w+\n\n(.+?)\n\n## ", model_card)
        name = "unknown"
        if name_match:
            name = name_match.group(1).strip().split("\n")[-1].strip()

        # Extract library name
        library = ""
        library_match = re.search(r"### Library\n\n(.+?)\n\n### ", model_card)
        if library_match:
            library = library_match.group(1).strip()

        # Extract tags
        tags = []
        tags_match = re.search(r"### Tags\n\n(.+?)\n\n### ", model_card)
        if tags_match:
            tags_str = tags_match.group(1).strip()
            tags = re.findall(r"\[([^\]]+)\]", tags_str)

        return {
            "name": name,
            "library": library,
            "tags": tags,
            "description": "".join(model_card.split("\n\n")[:5]),
        }

    def _get_name(self, model_info: Dict[str, Any]) -> str:
        """Get model name for Ollama.

        Args:
            model_info: Model information dictionary.

        Returns:
            Model name string.
        """
        # Extract name from tags if available
        tags = model_info.get("tags", [])
        for tag in tags:
            if tag.startswith("task:"):
                task = tag.replace("task:", "")
                if task == "text-generation":
                    return f"{model_info.get('name', '')}-text-generation"
                elif task == "text-classification":
                    return f"{model_info.get('name', '')}-text-classification"
                elif task == "text-generation-python":
                    return f"{model_info.get('name', '')}-text-generation-python"
                elif task == "text2text-generation":
                    return f"{model_info.get('name', '')}-text2text-generation"

        # Return default name
        return model_info.get("name", model_info.get("id", "unknown"))

    def _get_modelfile(
        self, model_id: str, context_length: int, params: Dict[str, Any]
    ) -> str:
        """Get Modelfile content for Ollama.

        Args:
            model_id: The model identifier.
            context_length: Context window size.
            params: Additional parameters.

        Returns:
            Modelfile content string.
        """
        parameters = self._get_parameters(params, context_length)
        generator = ModelfileGenerator()
        return generator.generate(model_id, context_length, params)

    def _get_parameters(
        self, params: Optional[Dict[str, Any]], context_length: int
    ) -> Dict[str, Any]:
        """Get model parameters for Ollama.

        Args:
            params: Dictionary of parameters (from model card or custom params).
            context_length: Context window size.

        Returns:
            Parameters dictionary.
        """
        parameters = {
            "num_ctx": context_length,
            "num_batch": 512,
            "num_gpu": -1,  # Use CPU by default
            "num_thread": 4,
        }

        # Add custom parameters from params if provided
        for key, value in (
            context_length.items() if isinstance(context_length, dict) else {}
        ):
            if key not in parameters:
                parameters[key] = value

        return parameters

    def _format_json(self, data: Dict[str, Any]) -> str:
        """Format JSON with proper escaping.

        Args:
            data: Dictionary to format.

        Returns:
            Formatted JSON string.
        """
        import json

        def escape_modelfile(modelfile: str) -> str:
            """Escape special characters in Modelfile content.

            Args:
                modelfile: The Modelfile content.

            Returns:
                Escaped string.
            """
            lines = modelfile.split("\n")
            escaped_lines = []

            for line in lines:
                escaped = line.replace("\\", "\\\\").replace('"', '\\"')
                if line.strip().startswith("FROM "):
                    escaped = f'"{escaped}"'
                escaped_lines.append(escaped)

            return "\n".join(escaped_lines)

        def format_modelfile(obj: Any) -> Any:
            """Recursively format Modelfile content.

            Args:
                obj: Object to format.

            Returns:
                Formatted object.
            """
            if isinstance(obj, dict):
                if "modelfile" in obj:
                    obj["modelfile"] = escape_modelfile(obj["modelfile"])
                return {k: format_modelfile(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [format_modelfile(item) for item in obj]
            return obj

        formatted = format_modelfile(data)
        return json.dumps(formatted, indent=2, ensure_ascii=False)

    def get_model_card(self, model_id: str) -> Optional[str]:
        """Fetch model card content.

        Args:
            model_id: The Hugging Face model identifier.

        Returns:
            Model card content or None.
        """
        from .huggingface import HFClient

        hf_client = HFClient()
        return hf_client.get_model_card(model_id)

    def extract_parameters(self, model_card: str) -> Dict[str, Any]:
        """Extract parameters from model card.

        Args:
            model_card: Model card content.

        Returns:
            Extracted parameters.
        """
        import re

        # Extract library name
        library = ""
        library_match = re.search(r"### Library\n\n(.+?)\n\n### ", model_card)
        if library_match:
            library = library_match.group(1).strip()

        # Extract tags
        tags = []
        tags_match = re.search(r"### Tags\n\n(.+?)\n\n### ", model_card)
        if tags_match:
            tags_str = tags_match.group(1).strip()
            tags = re.findall(r"\[([^\]]+)\]", tags_str)

        # Extract pipeline tag
        pipeline = ""
        pipeline_match = re.search(r"### Pipeline\n\n(.+?)\n\n### ", model_card)
        if pipeline_match:
            pipeline = pipeline_match.group(1).strip()

        return {
            "library": library,
            "tags": tags,
            "pipeline": pipeline,
        }
