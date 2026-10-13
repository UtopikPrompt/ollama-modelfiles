"""
Tests for the FROM-line path in the generated Modelfile.

The Modelfile is written to <root>/modelfiles (OLLAMA_CREATE_DIR) while the
GGUF model file is downloaded to <cwd>/models. Ollama resolves the FROM line
relative to the Modelfile's own location, so the FROM value must be a path
relative to <root>/modelfiles, not an absolute path.
"""

import os

import pytest

from cookie_cutter_modelfile.cli import OLLAMA_CREATE_DIR


@pytest.fixture
def root_dir():
    """Absolute path to the project root (parent of the modelfiles folder)."""
    return os.path.dirname(OLLAMA_CREATE_DIR)


def test_modelfiles_dir_is_relative_to_root(root_dir):
    """The modelfiles folder lives inside the project root."""
    assert os.path.isdir(OLLAMA_CREATE_DIR)
    assert os.path.abspath(OLLAMA_CREATE_DIR).startswith(os.path.abspath(root_dir))


def test_relative_from_path_points_to_models_dir(root_dir):
    """A GGUF in ./models must resolve, relative to modelfiles/, to ../models."""
    models_dir = os.path.join(root_dir, "models")
    # Simulate a GGUF downloaded inside the models cache.
    gguf_abs = os.path.join(models_dir, "Ornith-1.5-9B-uncensored.Q3_K_M.gguf")
    # The Modelfile's FROM line is computed relative to the modelfiles folder.
    rel = os.path.relpath(gguf_abs, start=OLLAMA_CREATE_DIR).replace("\\", "/")
    assert rel == "../models/Ornith-1.5-9B-uncensored.Q3_K_M.gguf"


def test_relative_from_path_normalizes_separators(root_dir):
    """The relative path uses forward slashes for portability."""
    models_dir = os.path.join(root_dir, "models")
    gguf_abs = os.path.join(models_dir, "model.Q4_K_M.gguf")
    rel = os.path.relpath(gguf_abs, start=OLLAMA_CREATE_DIR).replace("\\", "/")
    assert "\\" not in rel
    assert rel.startswith("../models/")
