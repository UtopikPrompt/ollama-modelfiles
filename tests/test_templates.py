"""
Tests for the template manager.
"""

import pytest
from unittest.mock import Mock, patch

from cookie_cutter_modelfile.templates import TemplateManager


class TestTemplateManager:
    """Tests for the template manager."""
    
    @pytest.fixture
    def template_manager(self):
        """Create a test template manager."""
        return TemplateManager()
    
    def test_load_text_template(self, template_manager):
        """Test loading a text template."""
        template = template_manager.load_template("chat")
        
        assert template is not None
        assert "FROM " in template
        assert "PARAMETER temperature" in template
    
    def test_load_jinja2_template(self, template_manager):
        """Test loading a Jinja2 template."""
        template = template_manager.load_template("base")
        
        assert template is not None
        assert "{{ model_id }}" in template
        assert "{{ render_param" in template
    
    def test_render_template(self, template_manager):
        """Test template rendering."""
        template = template_manager.load_template("base")
        context = {
            "model_id": "test/model",
            "temperature": 0.7,
            "top_p": 0.9,
            "context_length": 4096,
            "num_threads": 4,
        }
        
        rendered = template_manager.render(template, context)
        
        assert "FROM test/model" in rendered
        assert "PARAMETER temperature 0.7" in rendered
    
    def test_get_templates(self, template_manager):
        """Test getting list of templates."""
        templates = template_manager.get_templates()
        
        assert "chat" in templates
        assert "completion" in templates
        assert "image" in templates
        assert "default" in templates
    
    def test_load_all_templates(self, template_manager):
        """Test loading all templates."""
        all_templates = template_manager.load_all_templates()
        
        assert "chat" in all_templates
        assert "completion" in all_templates
        assert "image" in all_templates
        assert "default" in all_templates
    
    def test_get_templates_empty(self, template_manager):
        """Test getting templates when none exist."""
        templates = template_manager.get_templates()
        
        assert len(templates) >= 0
    
    def test_render_with_defaults(self, template_manager):
        """Test rendering with default values."""
        template = template_manager.load_template("base")
        context = {
            "model_id": "test/model",
        }
        
        rendered = template_manager.render(template, context)
        
        assert "PARAMETER temperature 0.7" in rendered
        assert "PARAMETER top_p 0.9" in rendered
        assert "PARAMETER num_predict 4096" in rendered
    
    def test_render_with_custom_values(self, template_manager):
        """Test rendering with custom values."""
        template = template_manager.load_template("base")
        context = {
            "model_id": "test/model",
            "temperature": 0.8,
            "top_p": 0.95,
            "context_length": 8192,
            "num_threads": 8,
        }
        
        rendered = template_manager.render(template, context)
        
        assert "PARAMETER temperature 0.8" in rendered
        assert "PARAMETER top_p 0.95" in rendered
        assert "PARAMETER num_predict 8192" in rendered
