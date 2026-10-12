"""
Hugging Face Hub client for fetching model information.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from huggingface_hub import HfApi, model_info, list_models, ModelInfo, ModelCard


class HFClient:
    """Client for interacting with Hugging Face Hub API."""
    
    def __init__(self):
        """Initialize the Hugging Face client."""
        self.api = HfApi()
        self._hf_api = self.api  # Internal API for testing
    
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
            return self.api.search_models(query)
        except Exception as e:
            print(f"Error searching models: {e}")
            return []
    
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
                "files_count": len(model_info.files_metadata) if model_info.files_metadata else 0,
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
