"""Hugging Face Hub client for metadata extraction."""

from dataclasses import dataclass, field
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import json


@dataclass
class ModelMetadata:
    """Container for Hugging Face model metadata."""
    
    id: Optional[str] = None
    """Hugging Face model identifier (e.g., 'meta-llama/Llama-3.2-1B-Instruct')."""
    
    author: Optional[str] = None
    """Author/organization name."""
    
    model_name: Optional[str] = None
    """Model name (last path component)."""
    
    description: Optional[str] = None
    """Model description."""
    
    library_name: Optional[str] = None
    """Library the model is from (e.g., 'llama-cpp-python')."""
    
    pipeline_tag: Optional[str] = None
    """Task pipeline tag (e.g., 'text-generation')."""
    
    tags: Optional[list] = None
    """List of model tags."""
    
    license: Optional[str] = None
    """License identifier (e.g., 'llama-3.1')."""
    
    downloads: Optional[int] = None
    """Monthly download count."""
    
    last_modified: Optional[str] = None
    """Last modification timestamp."""
    
    size_in_gb: Optional[float] = None
    """Model size in gigabytes."""
    
    @property
    def name(self) -> Optional[str]:
        """Full model name (author/model)."""
        if self.id:
            return self.id
        if self.author and self.model_name:
            return f"{self.author}/{self.model_name}"
        return None
    
    @property
    def author_name(self) -> Optional[str]:
        """Author name with organization handling."""
        if self.author:
            return self.author
        if self.id and '/' in self.id:
            return self.id.split('/')[0]
        return None
    
    @property
    def download_url(self) -> Optional[str]:
        """Hugging Face model download URL."""
        if self.id:
            return f"https://huggingface.co/{self.id}/resolve/main/{self.id}.gguf"
        return None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            attr: value for attr, value in self.__dict__.items() 
                   if value is not None
        }


class HFHubClient:
    """Client for fetching Hugging Face model metadata."""
    
    # HF Hub API endpoint
    API_BASE = "https://huggingface.co/api/models"
    
    # Rate limiting headers
    RATE_LIMIT_HEADER = "x-ratelimit-remaining"
    RATE_LIMIT_RESET_HEADER = "x-ratelimit-reset"
    
    def __init__(self, token: Optional[str] = None):
        """Initialize the HF Hub client.
        
        Args:
            token: Optional Hugging Face API token for authenticated requests.
                   Can also be set via HUGGINGFACE_TOKEN environment variable.
        """
        self.token = token or self._get_token_from_env()
    
    def _get_token_from_env(self) -> Optional[str]:
        """Get token from environment variable."""
        import os
        return os.environ.get("HUGGINGFACE_TOKEN")
    
    def _make_request(self, endpoint: str, headers: dict = None) -> dict:
        """Make HTTP request to HF Hub API.
        
        Args:
            endpoint: API endpoint (e.g., 'meta-llama/Llama-3.2-1B-Instruct')
            headers: Optional custom headers
            
        Returns:
            JSON response as dict
            
        Raises:
            HFHubError: If request fails
        """
        url = f"{self.API_BASE}/{endpoint}"
        
        # Build headers
        headers = headers or {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        headers["User-Agent"] = "ollama-modelfiles/0.1.0"
        
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=30) as response:
                data = response.read().decode('utf-8')
                return json.loads(data)
                
        except HTTPError as e:
            error_msg = self._format_error(e)
            raise HFHubError(f"HF API error: {error_msg}") from e
        except URLError as e:
            error_msg = self._format_error(e)
            raise HFHubError(f"Network error: {error_msg}") from e
    
    def _format_error(self, error: Exception) -> str:
        """Format error message for users."""
        if hasattr(error, "reason"):
            return str(error.reason) if error.reason else str(error)
        return str(error)
    
    def fetch_model_metadata(self, model_id: str, force: bool = False) -> ModelMetadata:
        """Fetch metadata for a single model.
        
        Args:
            model_id: Model identifier (e.g., 'meta-llama/Llama-3.2-1B-Instruct')
            force: Always fetch metadata even if recently fetched
            
        Returns:
            ModelMetadata instance
            
        Raises:
            HFHubError: If metadata fetch fails
        """
        # Check if we have cached metadata
        if not force:
            cached = self._get_cached_metadata(model_id)
            if cached:
                return cached
        
        # Fetch from API
        try:
            response = self._make_request(model_id)
            return self._parse_metadata(response, model_id)
            
        except HFHubError as e:
            # Log warning but don't fail the entire process
            print(f"Warning: Failed to fetch metadata for {model_id}: {e}")
            # Return minimal metadata
            return ModelMetadata(id=model_id)
    
    def _get_cached_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get cached metadata if available."""
        import hashlib
        import pickle
        
        cache_file = self._get_cache_file()
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            
            # Parse model_id to hash key
            namespace, model_name = model_id.split('/', 1)
            key = f"{namespace}:{model_name}"
            
            if key in cache:
                return cache[key]
                
        except Exception:
            pass
        
        return None
    
    def _put_cached_metadata(self, model_id: str, metadata: ModelMetadata):
        """Cache metadata for future retrieval."""
        import hashlib
        import pickle
        
        cache_file = self._get_cache_file()
        
        try:
            if not cache_file.exists():
                cache_file.touch()
            
            cache = {}
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cache = pickle.load(f)
            
            namespace, model_name = model_id.split('/', 1)
            key = f"{namespace}:{model_name}"
            cache[key] = metadata
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache, f)
                
        except Exception as e:
            print(f"Warning: Failed to cache metadata: {e}")
    
    def _get_cache_file(self) -> str:
        """Get path to metadata cache file."""
        cache_dir = ".hf_hub_cache"
        import os
        
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, "metadata_cache.p")
    
    def _parse_metadata(self, response: dict, model_id: str) -> ModelMetadata:
        """Parse API response into ModelMetadata."""
        metadata = ModelMetadata(id=model_id)
        
        # Extract all available fields
        for key, value in response.items():
            # Map API fields to our dataclass
            field_map = {
                'id': 'id',
                'sha': 'id',
                'author': 'author',
                'modelIndex': 'model_name',
                'description': 'description',
                'library_name': 'library_name',
                'pipeline_tag': 'pipeline_tag',
                'tags': 'tags',
                'license': 'license',
                'downloads': 'downloads',
                'lastModified': 'last_modified',
                'sizeInGB': 'size_in_gb',
            }
            
            if key in field_map:
                target_key = field_map[key]
                setattr(metadata, target_key, value)
        
        # Normalize last_modified
        if metadata.last_modified:
            metadata.last_modified = metadata.last_modified[:10]  # YYYY-MM-DD
        
        # Normalize size_in_gb (API returns in MB)
        if metadata.size_in_gb and 'MB' in metadata.size_in_gb:
            try:
                size_mb = float(metadata.size_in_gb)
                metadata.size_in_gb = size_mb / 1024
            except (ValueError, TypeError):
                pass
        
        # Sort tags for consistent output
        if metadata.tags and isinstance(metadata.tags, list):
            metadata.tags = sorted(metadata.tags)
        
        return metadata


class HFHubError(Exception):
    """Custom exception for Hugging Face Hub errors."""
    pass


def fetch_model_metadata(model_id: str, token: Optional[str] = None, force: bool = False) -> ModelMetadata:
    """Convenience function to fetch model metadata.
    
    Args:
        model_id: Model identifier (e.g., 'meta-llama/Llama-3.2-1B-Instruct')
        token: Optional Hugging Face token
        force: Always fetch metadata
        
    Returns:
        ModelMetadata instance
    """
    client = HFHubClient(token)
    return client.fetch_model_metadata(model_id, force)


if __name__ == "__main__":
    # Demo usage
    import sys
    
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
    else:
        model_id = "meta-llama/Llama-3.2-1B-Instruct"
    
    print(f"Fetching metadata for: {model_id}")
    
    metadata = fetch_model_metadata(model_id)
    
    print("\n=== Model Metadata ===")
    print(f"ID: {metadata.id}")
    print(f"Author: {metadata.author}")
    print(f"Model Name: {metadata.model_name}")
    print(f"Description: {metadata.description[:100]}...")
    print(f"License: {metadata.license}")
    print(f"Downloads: {metadata.downloads:,}/month")
    print(f"Size: {metadata.size_in_gb} GB")
    print(f"Pipeline: {metadata.pipeline_tag}")
    print(f"Tags: {len(metadata.tags) if metadata.tags else 0}")
    
    print("\n=== Download URL ===")
    print(metadata.download_url)
