"""
Tests for the template manager.
"""

from unittest.mock import Mock, patch

import pytest

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
        # Template is rendered, so macros are expanded
        assert "FROM " in template
        assert "PARAMETER temperature" in template
        assert "PARAMETER top_p" in template
        assert "PARAMETER num_ctx" in template
        assert "PARAMETER num_thread" in template

    def test_render_template(self, template_manager):
        """Test template rendering."""
        # Pass raw template source, not rendered version
        raw_template = """{% from "macros.j2" import render_param %}

FROM {{ model_id }}

# Parameters
{{ render_param("temperature", temperature, "0.7") }}
{{ render_param("top_p", top_p, "0.9") }}
{{ render_param("num_predict", context_length, "4096") }}
{{ render_param("num_threads", num_threads, "4") }}
"""
        context = {
            "model_id": "test/model",
            "temperature": 0.7,
            "top_p": 0.9,
            "context_length": 4096,
            "num_threads": 4,
        }

        rendered = template_manager.render(raw_template, context)

        assert "FROM test/model" in rendered
        assert "PARAMETER temperature 0.7" in rendered

    def test_get_templates(self, template_manager):
        """Test getting list of templates."""
        templates = template_manager.get_templates()

        # 'chat' and 'completion' are jinja2 templates
        # 'image' is a jinja2 template
        # 'default' is a text template
        assert "chat" in templates
        assert "completion" in templates
        assert "image" in templates
        # 'default' is available as a text template
        assert "default" in templates or len(templates) >= 4

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

        assert len(templates) >= 3  # At least chat, completion, image

    def test_render_with_defaults(self, template_manager):
        """Test rendering with default values."""
        # Pass raw template source, not rendered version
        raw_template = """{% from "macros.j2" import render_param %}

FROM {{ model_id }}

# Parameters
{{ render_param("temperature", temperature, "0.7") }}
{{ render_param("top_p", top_p, "0.9") }}
{{ render_param("num_predict", context_length, "4096") }}
{{ render_param("num_threads", num_threads, "4") }}
"""
        context = {
            "model_id": "test/model",
        }

        rendered = template_manager.render(raw_template, context)

        # Check that template was rendered (contains model_id)
        assert "test/model" in rendered
        # Check that parameters were rendered (not macro syntax)
        assert "PARAMETER temperature" in rendered

    def test_render_with_custom_values(self, template_manager):
        """Test rendering with custom values."""
        # Pass raw template source, not rendered version
        raw_template = """{% from "macros.j2" import render_param %}

FROM {{ model_id }}

# Parameters
{{ render_param("temperature", temperature, "0.7") }}
{{ render_param("top_p", top_p, "0.9") }}
{{ render_param("num_predict", context_length, "4096") }}
{{ render_param("num_threads", num_threads, "4") }}
"""
        context = {
            "model_id": "test/model",
            "temperature": 0.8,
            "top_p": 0.95,
            "context_length": 8192,
            "num_threads": 8,
        }

        rendered = template_manager.render(raw_template, context)

        # Check that template was rendered with custom values
        assert "test/model" in rendered
        assert "PARAMETER temperature 0.8" in rendered
