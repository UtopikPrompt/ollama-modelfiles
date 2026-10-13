"""
Tests for the Hugging Face client.
"""

import tempfile
from pathlib import Path
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

    @pytest.fixture
    def token_client(self):
        """Create a real HFClient with a mocked HfApi for token-resolution tests."""
        import cookie_cutter_modelfile.huggingface as hf_mod

        with patch("cookie_cutter_modelfile.huggingface.HfApi") as mock_hfapi:
            mock_hfapi.return_value = Mock()
            client = HFClient()
            yield client
            assert mock_hfapi.call_args.kwargs["token"]

    def test_no_token_no_env_no_envfile(self, monkeypatch):
        """No token anywhere resolves to None and HfApi gets no token."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        client = HFClient()

        assert client._hf_token is None
        import cookie_cutter_modelfile.huggingface as hf_mod

        assert hf_mod.HfApi.call_args.kwargs["token"] is None

    def test_explicit_token_takes_precedence(self, monkeypatch):
        """Explicit token wins over env var and .env file."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        env_dir = Path(tempfile.mkdtemp())
        (env_dir / ".env").write_text("HF_TOKEN=envfile_token\n")
        monkeypatch.chdir(env_dir)
        client = HFClient(token="explicit_token")

        assert client._hf_token == "explicit_token"
        import cookie_cutter_modelfile.huggingface as hf_mod

        assert hf_mod.HfApi.call_args.kwargs["token"] == "explicit_token"

    def test_env_var_precedence_over_envfile(self, monkeypatch):
        """HF_TOKEN env var wins over the same key in a .env file."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        env_dir = Path(tempfile.mkdtemp())
        (env_dir / ".env").write_text("HF_TOKEN=envfile_token\n")
        monkeypatch.chdir(env_dir)
        monkeypatch.setenv("HF_TOKEN", "env_var_token")
        client = HFClient()

        assert client._hf_token == "env_var_token"
        import cookie_cutter_modelfile.huggingface as hf_mod

        assert hf_mod.HfApi.call_args.kwargs["token"] == "env_var_token"

    def test_envfile_token_used(self, monkeypatch):
        """When no env var is set, the .env file token is used."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        env_dir = Path(tempfile.mkdtemp())
        (env_dir / ".env").write_text("HF_TOKEN=envfile_token\n")
        monkeypatch.chdir(env_dir)
        client = HFClient()

        assert client._hf_token == "envfile_token"
        import cookie_cutter_modelfile.huggingface as hf_mod

        assert hf_mod.HfApi.call_args.kwargs["token"] == "envfile_token"

    def test_envfile_parsing_quoted_comments_whitespace(self, monkeypatch):
        """.env parsing handles quotes, comments, blank lines and whitespace."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        env_dir = Path(tempfile.mkdtemp())
        content = (
            "# This is a comment\n"
            "\n"
            "  \n"
            'HF_TOKEN = "quoted token"  \n'
            "\n"
            "HF_TOKEN2=second token\n"
        )
        (env_dir / ".env").write_text(content)
        monkeypatch.chdir(env_dir)
        client = HFClient()

        assert client._hf_token == "quoted token"
        import cookie_cutter_modelfile.huggingface as hf_mod

        assert hf_mod.HfApi.call_args.kwargs["token"] == "quoted token"

    def test_no_crash_without_token(self, monkeypatch):
        """Constructing a client with no token present does not crash."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        client = HFClient()

        assert client._hf_token is None
        import cookie_cutter_modelfile.huggingface as hf_mod

        assert hf_mod.HfApi.call_args.kwargs["token"] is None

    def test_get_model_card_not_found(self, client):
        """Test getting model card for non-existent model."""
        mock_api.model_card.side_effect = Exception("Model not found")

        card = client.get_model_card("nonexistent/model")

        assert card is None

    def test_download_gguf_local_dir(self, client):
        """Test downloading a GGUF file into a custom local_dir."""

        with patch(
            "cookie_cutter_modelfile.huggingface.hf_hub_download"
        ) as mock_download:
            mock_download.return_value = r"C:\fake\path\Qwen3.5-9B.Q3_K_M.gguf"
            result = client.download_gguf(
                "user/model", "Qwen3.5-9B.Q3_K_M.gguf", local_dir="my-models"
            )
            mock_download.assert_called_once_with(
                repo_id="user/model",
                filename="Qwen3.5-9B.Q3_K_M.gguf",
                repo_type="model",
                local_dir="my-models",
            )
            assert result == r"C:\fake\path\Qwen3.5-9B.Q3_K_M.gguf"

    def test_download_gguf_default_local_dir(self, client):
        """Test downloading a GGUF file uses default local_dir."""

        with patch(
            "cookie_cutter_modelfile.huggingface.hf_hub_download"
        ) as mock_download:
            mock_download.return_value = r"C:\fake\path\model.gguf"
            result = client.download_gguf("user/model", "model.gguf")
            assert mock_download.call_args.kwargs["local_dir"].endswith("models")
            assert result == r"C:\fake\path\model.gguf"

    def test_download_gguf_local_dir(self, client):
        """Test downloading a GGUF file into a custom local_dir."""

        with patch(
            "cookie_cutter_modelfile.huggingface.hf_hub_download"
        ) as mock_download:
            mock_download.return_value = r"C:\fake\path\Qwen3.5-9B.Q3_K_M.gguf"
            result = client.download_gguf(
                "user/model", "Qwen3.5-9B.Q3_K_M.gguf", local_dir="my-models"
            )
            mock_download.assert_called_once_with(
                repo_id="user/model",
                filename="Qwen3.5-9B.Q3_K_M.gguf",
                repo_type="model",
                local_dir="my-models",
            )
            assert result == r"C:\fake\path\Qwen3.5-9B.Q3_K_M.gguf"

    def test_download_gguf_default_local_dir(self, client):
        """Test downloading a GGUF file uses default local_dir."""

        with patch(
            "cookie_cutter_modelfile.huggingface.hf_hub_download"
        ) as mock_download:
            mock_download.return_value = r"C:\fake\path\model.gguf"
            result = client.download_gguf("user/model", "model.gguf")
            assert mock_download.call_args.kwargs["local_dir"].endswith("models")
            assert result == r"C:\fake\path\model.gguf"
