"""
Tests for the Hugging Face client.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from cookie_cutter_modelfile.huggingface import HFClient


class TestHFClient:
    """Tests for the Hugging Face client."""

    @pytest.fixture
    def client(self):
        """Create a test client with mocked API."""
        with patch("cookie_cutter_modelfile.huggingface.HFClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            client = HFClient()
            yield client

    def test_fetch_model_success(self, client):
        """Test fetching a model successfully."""
        mock_model = Mock()
        mock_model.id = "test/model"
        mock_model.name = "test/model"
        mock_model.tags = ["test:tag1", "test:tag2"]

        result = client.fetch_model("test/model")

        assert result is not None
        assert result.id == "test/model"
        client._hf_api.model_info.assert_called_once_with("test/model")

    def test_fetch_model_not_found(self, client):
        """Test fetching a non-existent model."""
        client._hf_api.model_info.return_value = None

        result = client.fetch_model("nonexistent/model")

        assert result is None

    def test_list_models(self, client):
        """Test listing models."""
        mock_model1 = Mock()
        mock_model1.id = "model1"
        mock_model1.name = "Model 1"
        mock_model1.tags = ["tag1"]

        mock_model2 = Mock()
        mock_model2.id = "model2"
        mock_model2.name = "Model 2"
        mock_model2.tags = ["tag2"]

        client._hf_api.list_models.return_value = [mock_model1, mock_model2]

        models = client.list_models()

        assert len(models) == 2
        assert models[0].id == "model1"

    def test_search_models(self, client):
        """Test searching models."""
        mock_model = Mock()
        mock_model.id = "search/model"
        mock_model.name = "Search Model"
        mock_model.tags = ["search"]

        client._hf_api.search_models.return_value = [mock_model]

        results = client.search_models("search")

        assert len(results) == 1
        assert results[0].id == "search/model"

    def test_get_model_info(self, client):
        """Test getting model information."""
        mock_model = Mock()
        mock_model.id = "info/model"
        mock_model.name = "Info Model"
        mock_model.tags = ["info:tag"]
        mock_model.library_name = "transformers"

        mock_card = Mock()
        mock_card.model_index = "model_index"
        mock_card.license = "MIT"
        mock_card.tags = ["card:tag1", "card:tag2"]
        client._hf_api.model_card.return_value = mock_card

        info = client.get_model_info("info/model")

        assert "id" in info
        assert "name" in info
        assert info["id"] == "info/model"
        assert info["name"] == "Info Model"

    def test_get_model_card(self, client):
        """Test getting model card."""
        mock_card = Mock()
        mock_card.content = "Model card content"
        client._hf_api.model_card.return_value = mock_card

        card = client.get_model_card("test/model")

        assert card == "Model card content"

    def test_get_model_card_not_found(self, client):
        """Test getting model card for non-existent model."""
        mock_api.model_card.side_effect = Exception("Model not found")

        card = client.get_model_card("nonexistent/model")

        assert card is None
