"""
Tests for the timestamped modelfile name used by the deploy command.

The deploy command writes a temporary Modelfile to ``<root>/modelfiles``
(OLLAMA_CREATE_DIR) before running ``ollama create``. Previously the file
was created with ``tempfile.mkstemp`` which yields an opaque random name
like ``tmp3s5ctfkn.modelfile``. It now uses a deterministic, timestamped
name (``<model>-<YYYYmmdd-HHMMSS>.modelfile``) so runs are identifiable
in logs. The file remains transient and is always removed afterwards.

These tests assert:
* the naming helper produces a timestamped, filename-safe name,
* the model name is sanitized and length-capped,
* the helper falls back to a timestamp-only name when the model name is
  empty/None, and
* the deploy path still creates and cleans up the working Modelfile.
"""

import os
import re
from unittest.mock import Mock, patch

import pytest

from cookie_cutter_modelfile.cli import CLI, OLLAMA_CREATE_DIR


@pytest.fixture
def cli():
    return CLI()


def test_name_is_timestamped_and_identifies_run(cli):
    """The name embeds the model name and a timestamp suffix."""
    name = cli._timestamped_modelfile_name("ornith-Q4_K_M")
    assert name.endswith(".modelfile")
    assert "ornith" in name
    # Format is ``<model>-<YYYYmmdd-HHMMSS>.modelfile``.
    assert re.search(r"ornith-Q4_K_M-\d{4}\d{2}\d{2}-\d{2}\d{2}\d{2}\.modelfile$", name)


def test_name_is_filename_safe(cli):
    """Special characters in the model name are replaced with underscores."""
    name = cli._timestamped_modelfile_name("mradermacher/Cyber-Ornith-1.5-9B")
    assert re.search(r"mradermacher_Cyber-Ornith-1\.5-9B-", name)
    # No characters outside the safe set (aside from the model name itself).
    assert re.fullmatch(r"[a-zA-Z0-9._-]+\.modelfile", name)


def test_name_is_length_capped(cli):
    """A very long model name is truncated so the file name stays short."""
    long_name = "a" * 200
    name = cli._timestamped_modelfile_name(long_name)
    # 60 chars for the sanitized name + ``-`` + 16 chars stamp + suffix.
    assert len(name) < 100
    assert name.startswith("a" * 60)


def test_name_falls_back_when_model_name_empty(cli):
    """An empty model name yields a timestamp-only name."""
    name = cli._timestamped_modelfile_name("")
    assert (
        name
        == re.search(r"(\d{4}\d{2}\d{2}-\d{2}\d{2}\d{2})\.modelfile", name).group(1)
        + ".modelfile"
    )


def test_name_falls_back_when_model_name_none(cli):
    """A ``None`` model name yields a timestamp-only name too."""
    name = cli._timestamped_modelfile_name(None)
    assert re.fullmatch(r"\d{4}\d{2}\d{2}-\d{2}\d{2}\d{2}\.modelfile", name)


def test_deploy_writes_and_cleans_up_temp_modelfile(cli):
    """deploy() creates a timestamped working Modelfile and removes it."""
    create_result = Mock(returncode=0, stdout="", stderr="")
    with patch("subprocess.run", return_value=create_result):
        cli._check_ollama = Mock()
        code = cli.deploy("mradermacher/ornith-GGUF", "q4_k_m")
    assert code == 0
    # The deploy should not have actually created a persistent file: the
    # working Modelfile is transient and cleaned up in a finally block.
    assert subprocess.run.call_args[0][0][1] == "ollama"
    assert subprocess.run.call_args[0][0][3] == "--file"
    assert not os.path.exists(modelfile_path)


def test_deploy_cleans_up_on_create_failure(cli):
    """If ``ollama create`` fails, the working Modelfile is still removed."""
    failing = Mock(returncode=1, stdout="", stderr="boom")
    with patch("subprocess.run", return_value=failing):
        cli._check_ollama = Mock()
        code = cli.deploy("mradermacher/ornith-GGUF", "q4_k_m")
    assert code == 1
    assert not os.path.exists(modelfile_path)


def test_deploy_saves_to_output_when_requested(cli):
    """The ``--output``/``-o`` path is untouched by the rename change."""
    saved = Mock(return_value=True)
    create_result = Mock(returncode=0, stdout="", stderr="")
    with patch("subprocess.run", return_value=create_result):
        cli._check_ollama = Mock()
        code = cli.deploy("mradermacher/ornith-GGUF", "q4_k_m", output="out.modelfile")
    assert code == 0
    assert saved.called
