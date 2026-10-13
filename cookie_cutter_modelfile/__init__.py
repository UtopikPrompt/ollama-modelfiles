"""
Cookie Cutter Modelfile - Generate Ollama Modelfiles from Hugging Face models.

A CLI tool for generating Ollama Modelfiles from Hugging Face models with
customizable templates and parameters.
"""

__version__ = "0.1.0"

from .cli import main
from .config import Config, HardwareProfile, load_config, print_config
from .converter import OllamaConverter
from .huggingface import HFClient
from .modelfile import ModelfileGenerator
from .templates import TemplateManager

__all__ = [
    "main",
    "HFClient",
    "ModelfileGenerator",
    "OllamaConverter",
    "TemplateManager",
    "load_config",
    "print_config",
    "Config",
    "HardwareProfile",
]
