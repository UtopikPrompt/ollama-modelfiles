"""
Configuration management for the cookie cutter modelfile tool.

Loads the Ollama server URL and the list of available hardware profiles from a
YAML configuration file. Values may be overridden by environment variables
(``OLLAMA_URL`` / ``OLLAMA_HOST``) or by a ``.env`` file (loaded via
:mod:`python-dotenv`).

The default configuration file is ``config.yml`` located in the project
root. It can be pointed elsewhere with the ``OLLAMA_CONFIG`` environment
variable.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

# Default location of the YAML configuration file, one level up from this
# package (i.e. the project root).
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config.yml",
)


def _coerce_int(value: Any) -> int:
    """Best-effort conversion of a config value to ``int``.

    Args:
        value: The raw value from the configuration file.

    Returns:
        The integer value, or ``0`` if conversion fails.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


@dataclass
class OllamaConfig:
    """Ollama server connection settings."""

    url: str = "http://localhost:11434"


@dataclass
class HardwareProfile:
    """A selectable hardware profile.

    Attributes:
        name: Identifier of the profile (used with ``--hardware``).
        devices: Ollama device tags, e.g. ``["GPU:0", "GPU:1"]`` or ``["CPU"]``.
        num_threads: Value for ``PARAMETER num_thread``. ``0`` lets Ollama
            auto-detect the optimal thread count.
        gpu_layers: Value for ``PARAMETER num_gpu``. ``-1`` lets Ollama
            auto-detect; ``0`` forces CPU execution.
        vram: Total VRAM of the primary GPU in bytes. Used to estimate the
            context window (``num_predict``) by subtracting the model size.
            ``None`` disables VRAM-based context estimation.
        description: Human-readable description shown by ``config``.
    """

    name: str
    devices: List[str]
    num_threads: int
    gpu_layers: int = -1
    vram: Optional[int] = None
    description: str = ""


@dataclass
class Config:
    """Top-level configuration object."""

    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    hardware: Dict[str, HardwareProfile] = field(default_factory=dict)
    default_hardware: Optional[str] = None

    def get_hardware(self, override: Optional[str] = None) -> HardwareProfile:
        """Return a hardware profile by name.

        Args:
            override: Explicit profile name. Falls back to
                ``default_hardware`` and finally the first available profile.

        Returns:
            The resolved :class:`HardwareProfile`.
        """
        name = override or self.default_hardware
        if not name:
            names = list(self.hardware.keys())
            name = names[0] if names else "cpu-only"
        return self.hardware[name]

    def get_num_threads(self, override: Optional[str] = None) -> int:
        """Return ``num_threads`` for the resolved hardware profile."""
        return self.get_hardware(override).num_threads

    def get_gpu_layers(self, override: Optional[str] = None) -> int:
        """Return ``gpu_layers`` for the resolved hardware profile."""
        return self.get_hardware(override).gpu_layers

    def get_vram(self, override: Optional[str] = None) -> Optional[int]:
        """Return the VRAM (in bytes) of the resolved hardware profile.

        Returns:
            The VRAM in bytes, or ``None`` if the profile does not define one.
        """
        return self.get_hardware(override).vram

    def get_ollama_url(self, override: Optional[str] = None) -> str:
        """Return the Ollama URL, or ``override`` when provided."""
        return override if override is not None else self.ollama.url


def load_config(path: Optional[str] = None) -> Config:
    """Load configuration from a YAML file, merging with environment defaults.

    Environment variables (``OLLAMA_URL`` / ``OLLAMA_HOST``) may come from the
    process environment or from a ``.env`` file in the current working directory
    (loaded via :mod:`python-dotenv`). ``python-dotenv`` never overrides an
    existing environment variable, so an ``OLLAMA_URL``/``OLLAMA_HOST`` env var
    always wins over the same key in the ``.env`` file.

    Args:
        path: Configuration file path. Defaults to ``OLLAMA_CONFIG`` or the
            built-in project ``config.yml``.

    Returns:
        A populated :class:`Config` (with defaults when no file is found).
    """
    load_dotenv()  # Populate OLLAMA_URL / OLLAMA_HOST from a .env file (if present).
    resolved_path = path or os.environ.get("OLLAMA_CONFIG") or DEFAULT_CONFIG_PATH
    config = Config()

    if not os.path.isfile(resolved_path):
        return config

    with open(resolved_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    ollama_section = data.get("ollama", {})
    url = (
        ollama_section.get("url")
        or os.environ.get("OLLAMA_URL")
        or os.environ.get("OLLAMA_HOST")
    )
    config.ollama = OllamaConfig(url=url or "http://localhost:11434")

    hardware_section = data.get("hardware", {})
    config.default_hardware = hardware_section.get("default")
    profiles = hardware_section.get("profiles", {})
    for key, value in profiles.items():
        if not isinstance(value, dict):
            continue
        devices = value.get("devices", [])
        if isinstance(devices, str):
            devices = [devices]
        config.hardware[key] = HardwareProfile(
            name=value.get("name", key),
            devices=[str(d) for d in devices],
            num_threads=_coerce_int(value.get("num_threads", 0)),
            gpu_layers=_coerce_int(value.get("gpu_layers", -1)),
            description=value.get("description", ""),
        )

    return config


def print_config(config: Config) -> None:
    """Print the resolved Ollama URL and available hardware profiles."""
    from rich.console import Console

    console = Console(file=sys.stdout)

    console.print("[bold cyan]Ollama URL:[/bold cyan] " + config.ollama.url)
    if config.default_hardware:
        console.print(
            "[bold cyan]Default hardware:[/bold cyan] " + config.default_hardware
        )

    console.print("[bold cyan]Available hardware profiles:[/bold cyan]")
    if not config.hardware:
        console.print("  (none configured - using defaults)")
        return

    for name, profile in config.hardware.items():
        console.print(
            f"  • {name} ({profile.description})"
            f"  — devices: {', '.join(profile.devices)}"
            f"  — threads: {profile.num_threads}"
            f"  — gpu_layers: {profile.gpu_layers}"
        )
