"""
Hugging Face Hub client for fetching model information.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from huggingface_hub import (
    HfApi,
    ModelCard,
    ModelInfo,
    hf_hub_download,
    list_models,
    model_info,
)


class HFClient:
    """Client for interacting with Hugging Face Hub API."""

    def __init__(self, token: Optional[str] = None):
        """Initialize the Hugging Face client.

        A Hugging Face token is resolved from the following sources in order
        of precedence (the first non-empty value wins):

        1. An explicit ``token`` passed to this constructor.
        2. The ``HF_TOKEN`` environment variable.
        3. A ``HF_TOKEN`` entry in a ``.env`` file in the current working
           directory (loaded via :mod:`dotenv`).

        Args:
            token: Optional explicit Hugging Face token. When provided it takes
                precedence over environment and file-based sources.

        Returns:
            None. The resolved token is stored on ``self._hf_token`` and passed
            to :class:`huggingface_hub.HfApi`.
        """
        self._hf_token = self._resolve_token(token)
        self.api = HfApi(token=self._hf_token)
        self._hf_api = self.api  # Internal API for testing

    @staticmethod
    def _resolve_token(
        token: Optional[str],
    ) -> Optional[str]:
        """Resolve the Hugging Face token from the configured sources.

        Precedence: explicit token > ``HF_TOKEN`` env var > ``.env`` file.

        Args:
            token: Optional explicit token provided by the caller.

        Returns:
            The first non-empty token found, or ``None`` if no token was found.
        """
        # 1. Explicit token takes precedence over everything else.
        if token:
            return token

        # 2. Environment variable.
        env_token = os.environ.get("HF_TOKEN")
        if env_token:
            return env_token

        # 3. .env file in the current working directory.
        # ``load_dotenv`` does not override variables already set in the
        # environment, so an ``HF_TOKEN`` env var (checked above) always wins
        # over the same key in the ``.env`` file.
        load_dotenv()
        file_token = os.environ.get("HF_TOKEN")
        if file_token:
            return file_token

        return None

    def fetch_model(self, model_id: str) -> Optional[ModelInfo]:
        """Fetch model metadata from Hugging Face.

        Args:
            model_id: The Hugging Face model identifier.

        Returns:
            ModelInfo object or None if model not found.
        """
        try:
            return model_info(model_id)
        except Exception as e:
            print(f"Error fetching model {model_id}: {e}")
            return None

    def list_models(self, category: str = "all") -> List[ModelInfo]:
        """List models by category.

        Args:
            category: The category to filter by.

        Returns:
            List of ModelInfo objects.
        """
        try:
            return list_models(category=category)
        except Exception as e:
            print(f"Error listing models: {e}")
            return []

    def search_models(self, query: str) -> List[ModelInfo]:
        """Search for models by query.

        Args:
            query: The search query string.

        Returns:
            List of ModelInfo objects matching the query.
        """
        try:
            # search_models is not available in all versions of huggingface_hub
            # Fall back to filtering list_models results
            models = self.api.list_models()
            results = []
            for model in models:
                query_lower = query.lower()
                if (
                    query_lower in model.id.lower()
                    or query_lower in model.name.lower()
                    or any(query_lower in tag.lower() for tag in model.tags)
                ):
                    results.append(model)
            return results
        except Exception as e:
            print(f"Error searching models: {e}")
            return []

    def download_gguf(
        self, model_id: str, filename: str, local_dir: Optional[str] = None
    ) -> Optional[str]:
        """Download a GGUF file from Hugging Face to a local cache.

        Args:
            model_id: The Hugging Face model identifier.
            filename: The filename of the GGUF file to download.
            local_dir: Optional local directory to download into. Defaults to
                a gitignored ``./models/`` folder inside the current
                working directory so downloaded models stay out of git.

        Returns:
            Absolute path to the downloaded GGUF file, or None on failure.
        """
        if local_dir is None:
            local_dir = os.path.join(os.getcwd(), "models")
        try:
            return hf_hub_download(
                repo_id=model_id,
                filename=filename,
                repo_type="model",
                local_dir=local_dir,
            )
        except Exception as e:
            print(f"Error downloading {filename} from {model_id}: {e}")
            return None

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed model information.

        Args:
            model_id: The Hugging Face model identifier.

        Returns:
            Dictionary containing model information.
        """
        model_info = self.fetch_model(model_id)
        if model_info is None:
            return {}

        # Extract additional info from model card
        try:
            card = self.api.model_card(model_id)
            model_info_dict = {
                "id": model_info.id,
                "name": model_info.name,
                "description": str(card),
                "tags": model_info.tags,
                "library": getattr(model_info, "library_name", None),
                "pipeline_tag": getattr(model_info, "pipeline_tag", None),
                "files_count": (
                    len(model_info.files_metadata) if model_info.files_metadata else 0
                ),
            }

            # Add card-specific fields
            if hasattr(card, "model_index"):
                model_info_dict["model_index"] = str(card.model_index)
            if hasattr(card, "license"):
                model_info_dict["license"] = str(card.license)
            if hasattr(card, "tags"):
                model_info_dict["tags"] = list(card.tags)

            return model_info_dict
        except Exception as e:
            # Return basic info if card parsing fails
            return {
                "id": model_info.id,
                "name": model_info.name,
                "description": "Could not fetch model card",
                "tags": model_info.tags,
                "library": getattr(model_info, "library_name", None),
                "pipeline_tag": getattr(model_info, "pipeline_tag", None),
            }

    def get_model_card(self, model_id: str) -> Optional[str]:
        """Get the model card content.

        Args:
            model_id: The Hugging Face model identifier.

        Returns:
            Model card content or None if not found.
        """
        try:
            card = self._hf_api.model_card(model_id)
            return card.content
        except Exception as e:
            print(f"Error fetching model card: {e}")
            return None
