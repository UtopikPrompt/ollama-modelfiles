"""
Tests for the Ollama converter.
"""

import pytest
from unittest.mock import Mock, patch

from cookie_cutter_modelfile.converter import OllamaConverter


class TestOllamaConverter:
    """Tests for the Ollama converter."""
    
    @pytest.fixture
    def converter(self):
        """Create a test converter."""
        return OllamaConverter()
    
    @pytest.fixture(autouse=True)
    def mock_hf_client(self, converter):
        """Mock the HF client."""
        with patch('cookie_cutter_modelfile.converter.HFClient') as mock_client:
            mock_instance = Mock()
            mock_model = Mock()
            mock_model.id = "test/model"
            mock_model.name = "Test Model"
            mock_model.tags = ["task:text-generation", "test:tag"]
            mock_model.files_metadata = [Mock()]
            mock_instance.fetch_model.return_value = mock_model
            mock_client.return_value = mock_instance
            yield mock_client
    
    def test_convert_to_ollama(self, converter):
        """Test converting a model to Ollama format."""
        content = converter.convert_to_ollama("test/model", 4096, {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_threads": 4,
        })
        
        assert "name" in content
        assert "modelfile" in content
        assert "parameters" in content
        assert "num_ctx" in content
        assert "num_batch" in content
    
    def test_convert_with_custom_params(self, converter):
        """Test conversion with custom parameters."""
        content = converter.convert_to_ollama("custom/model", 8192, {
            "temperature": 0.8,
            "top_p": 0.95,
            "num_threads": 8,
        })
        
        assert "num_ctx" in content
        assert "8192" in content
        assert "num_thread" in content
        assert "8" in content
    
    def test_get_modelfile(self, converter):
        """Test getting Modelfile content."""
        modelfile = converter._get_modelfile("test/model", 4096, {})
        
        assert "FROM test/model" in modelfile
        assert "PARAMETER temperature" in modelfile
    
    def test_get_parameters(self, converter):
        """Test getting model parameters."""
        params = converter._get_parameters({"name": "test"}, {})
        
        assert "num_ctx" in params
        assert "num_batch" in params
        assert "num_gpu" in params
        assert "num_thread" in params
    
    def test_format_json(self, converter):
        """Test JSON formatting."""
        data = {
            "name": "test",
            "modelfile": "FROM test/model\nPARAMETER temperature 0.7",
            "parameters": {"num_ctx": 4096},
        }
        
        json_str = converter._format_json(data)
        
        assert json_str.startswith("{")
        assert json_str.endswith("}")
        assert "num_ctx" in json_str
    
    def test_format_json_escaped_modelfile(self, converter):
        """Test JSON formatting with escaped Modelfile content."""
        data = {
            "name": "test",
            "modelfile": 'FROM test/model\nPARAMETER stop "<s>"',
            "parameters": {},
        }
        
        json_str = converter._format_json(data)
        
        # Check that special characters are properly escaped
        # JSON should escape the < and > characters
        assert "<" in json_str or "\\u003c" in json_str
        assert "num_ctx" in json_str
    
    def test_get_model_card(self, converter):
        """Test getting model card."""
        mock_card = Mock()
        mock_card.__str__ = Mock(return_value="Model card content")
        
        with patch('cookie_cutter_modelfile.converter.HFClient') as mock_client:
            mock_instance = Mock()
            mock_instance.get_model_card.return_value = mock_card.__str__()
            mock_client.return_value = mock_instance
            
            card = converter.get_model_card("test/model")
            
            assert card == "Model card content"
    
    def test_extract_parameters(self, converter):
        """Test extracting parameters from model card."""
        model_card = """Model card content
        ### Library
        
        transformers
        
        ### Tags
        
        [task:text-generation]
        [test:tag1]
        [test:tag2]
        
        ### Pipeline
        
        text-generation
        """
        
        params = converter.extract_parameters(model_card)
        
        assert params["library"] == "transformers"
        assert "task:text-generation" in params["tags"]
        assert "text-generation" in params["pipeline"]
