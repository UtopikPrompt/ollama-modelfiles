"""
Tests for utility functions.
"""

import pytest
from cookie_cutter_modelfile.utils import (
    normalize_model_id,
    format_context_length,
    get_model_info_display,
    print_error,
)


class TestNormalizeModelId:
    """Tests for model ID normalization."""
    
    def test_basic_normalization(self):
        """Test basic normalization."""
        assert normalize_model_id("test/model") == "test/model"
    
    def test_strip_whitespace(self):
        """Test stripping whitespace."""
        assert normalize_model_id("  test/model  ") == "test/model"
    
    def test_hf_co_prefix(self):
        """Test removing hf.co/ prefix."""
        assert normalize_model_id("hf.co/test/model") == "test/model"
    
    def test_huggingface_co_prefix(self):
        """Test removing huggingface.co/ prefix."""
        assert normalize_model_id("huggingface.co/test/model") == "test/model"
    
    def test_https_prefix(self):
        """Test removing https:// prefix."""
        assert normalize_model_id("https://huggingface.co/test/model") == "test/model"
    
    def test_empty_input(self):
        """Test empty input."""
        assert normalize_model_id("") is None
    
    def test_none_input(self):
        """Test None input."""
        assert normalize_model_id(None) is None
    
    def test_invalid_input(self):
        """Test invalid input."""
        assert normalize_model_id(123) is None  # Not a string


class TestFormatContextLength:
    """Tests for context length formatting."""
    
    def test_small_value(self):
        """Test small context length."""
        assert format_context_length(1024) == "1024"
    
    def test_thousand(self):
        """Test thousand tokens."""
        assert format_context_length(1000) == "1.00K"
    
    def test_thousand_point_five(self):
        """Test 1500 tokens."""
        assert format_context_length(1500) == "1.50K"
    
    def test_million(self):
        """Test million tokens."""
        assert format_context_length(1000000) == "1.00M"
    
    def test_million_point_five(self):
        """Test 1.5 million tokens."""
        assert format_context_length(1500000) == "1.50M"
    
    def test_large_value(self):
        """Test large context length."""
        assert format_context_length(5000000) == "5.00M"


class TestGetModelInfoDisplay:
    """Tests for model info display."""
    
    def test_basic_display(self):
        """Test basic model info display."""
        model_info = {
            "id": "test/model",
            "name": "Test Model",
            "tags": ["test:tag"],
        }
        
        display = get_model_info_display(model_info)
        
        assert "Model Information" in str(display)
        assert "ID: test/model" in str(display)
        assert "Name: Test Model" in str(display)
    
    def test_display_with_description(self):
        """Test display with description."""
        model_info = {
            "id": "test/model",
            "name": "Test Model",
            "description": "This is a test model description",
        }
        
        display = get_model_info_display(model_info)
        
        assert "Description:" in str(display)
    
    def test_display_with_tags(self):
        """Test display with multiple tags."""
        model_info = {
            "id": "test/model",
            "name": "Test Model",
            "tags": ["tag1", "tag2", "tag3", "tag4"],
        }
        
        display = get_model_info_display(model_info)
        
        assert "tag1" in str(display)
        assert "tag2" in str(display)
        assert "tag3" in str(display)
        assert "tag4" in str(display)
    
    def test_display_with_more_tags(self):
        """Test display with more than 5 tags."""
        model_info = {
            "id": "test/model",
            "name": "Test Model",
            "tags": ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7"],
        }
        
        display = get_model_info_display(model_info)
        
        assert "+2 more" in str(display)


class TestPrintError:
    """Tests for error printing."""
    
    def test_print_error_without_exception(self):
        """Test printing error without exception."""
        from io import StringIO
        from contextlib import redirect_stdout
        
        f = StringIO()
        with redirect_stdout(f):
            print_error("Test error message")
        
        output = f.getvalue()
        assert "Error: Test error message" in output
        assert "❌" in output
    
    def test_print_error_with_exception(self):
        """Test printing error with exception."""
        from io import StringIO
        from contextlib import redirect_stdout
        
        f = StringIO()
        with redirect_stdout(f):
            print_error("Test error", ValueError("Test value error"))
        
        output = f.getvalue()
        assert "Error: Test error" in output
        assert "ValueError" in output
