"""
Tests for the Modelfile generator.
"""

from unittest.mock import Mock, patch

import pytest

from cookie_cutter_modelfile.modelfile import ModelfileGenerator
from cookie_cutter_modelfile.templates import TemplateManager


class TestModelfileGenerator:
    """Tests for the Modelfile generator."""

    @pytest.fixture
    def generator(self):
        """Create a test generator."""
        return ModelfileGenerator()

    def test_generate_basic(self, generator):
        """Test basic Modelfile generation."""
        content = generator.generate(
            "test/model",
            4096,
            {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_thread": 4,
            },
        )

        assert "FROM test/model" in content
        assert "PARAMETER temperature 0.7" in content
        assert "PARAMETER top_p 0.9" in content
        assert "PARAMETER num_ctx 4096" in content
        assert "PARAMETER num_thread 4" in content

        # Validate the generated content
        assert generator.validate_modelfile(content) is True

    def test_generate_custom_params(self, generator):
        """Test generation with custom parameters."""
        content = generator.generate(
            "custom/model",
            8192,
            {
                "temperature": 0.8,
                "top_p": 0.85,
                "num_thread": 8,
            },
        )

        assert "PARAMETER temperature 0.8" in content
        assert "PARAMETER top_p 0.85" in content
        assert "PARAMETER num_ctx 8192" in content
        assert "PARAMETER num_gpu -1" in content

    def test_validate_valid_modelfile(self, generator):
        """Test validation of a valid Modelfile."""
        content = """
FROM test/model

PARAMETER temperature 0.7
PARAMETER top_p 0.9
"""

        result = generator.validate_modelfile(content)

        assert result is True

    def test_validate_invalid_from(self, generator):
        """Test validation without FROM statement."""
        content = """
PARAMETER temperature 0.7
"""

        result = generator.validate_modelfile(content)

        assert result is False

    def test_validate_invalid_param(self, generator):
        """Test validation with invalid parameter."""
        content = """
FROM test/model

PARAMETER invalid_param test_value
"""

        result = generator.validate_modelfile(content)

        assert result is False

    def test_save_and_read(self, generator):
        """Test saving and reading Modelfile."""
        import os
        import tempfile

        content = generator.generate("save/test", 4096, {})

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".modelfile"
        ) as f:
            temp_path = f.name
            f.write(content)

        try:
            result = generator.save(content, temp_path)

            assert result is True

            with open(temp_path, "r") as f:
                read_content = f.read()

            assert read_content == content
        finally:
            os.unlink(temp_path)
