"""
Tests for GGUF filename derivation used by the deploy command.

These tests assert that an explicitly requested quant (e.g. from the URL) is
authoritative and never overridden by a repo-file listing that returns a
different/worse quant first.
"""

from unittest.mock import Mock, patch

import pytest

from cookie_cutter_modelfile.cli import CLI


@pytest.fixture
def cli():
    return CLI()


def test_requested_quant_is_authoritative(cli):
    """A URL-pinned quant must produce ``{base}.{quant}.gguf`` directly,
    without listing the repo files (which would return another quant).
    """
    # Should honor q3_k_m without consulting the repo contents (no file listing).
    filename = cli._requested_gguf_filename("mradermacher/ornith-GGUF", "q3_k_m")
    assert filename == "ornith.q3_k_m.gguf"


def test_requested_quant_no_base_stripping(cli):
    """The ``-GGUF`` marker must be stripped before appending the quant."""
    filename = cli._requested_gguf_filename("mradermacher/ornith-GGUF", "iq4_xs")
    assert filename == "ornith.iq4_xs.gguf"


def test_no_requested_quant_lists_files(cli):
    """Without an explicit quant, the repo files are listed and the first
    ``.gguf`` is returned.
    """
    mock_list = Mock(side_effect=[["a.Q4.gguf", "b.IQ4_XS.gguf", "README.md"], []])
    with patch("huggingface_hub.list_repo_files", mock_list):
        filename = cli._derive_gguf_filename("mradermacher/ornith-GGUF")
    assert filename == "a.Q4.gguf"


def test_no_requested_quant_no_files_fallback(cli):
    """When no ``.gguf`` files are found, fall back to ``{base}.gguf``."""
    mock_list = Mock(return_value=[])
    with patch("huggingface_hub.list_repo_files", mock_list):
        filename = cli._derive_gguf_filename("mradermacher/ornith-GGUF")
    assert filename == "ornith.gguf"


def test_repo_base_name_strips_gguf(cli):
    assert cli._repo_base_name("mradermacher/ornith-GGUF") == "ornith"


def test_repo_base_name_with_best(cli):
    assert (
        cli._repo_base_name("mradermacher/ornith-GGUF", "q3_k_m")
        == "ornith.q3_k_m.gguf"
    )


def test_extract_quant_from_url_preserves_case(cli):
    """HuggingFace is case-sensitive, so the quant extracted from a URL must
    keep its original case (e.g. ``Q3_K_L``), not be lowercased.
    """
    url = (
        "https://huggingface.co/mradermacher/Ornith-1.5-9B-uncensored-GGUF"
        "/resolve/main/Ornith-1.5-9B-uncensored.Q3_K_L.gguf?download=true"
    )
    assert cli._extract_quant_from_url(url) == "Q3_K_L"


def test_extract_quant_from_url_mixed_case(cli):
    """Mixed-case quants (e.g. ``Q4_0``) must also be preserved verbatim."""
    url = (
        "https://huggingface.co/mradermacher/ornith-GGUF"
        "/resolve/main/ornith.Q4_0.gguf?download=true"
    )
    assert cli._extract_quant_from_url(url) == "Q4_0"


def test_extract_quant_from_url_tree_url(cli):
    """Case-preservation applies to ``/tree/`` URLs too, not just ``/resolve/``."""
    url = (
        "https://huggingface.co/mradermacher/ornith-GGUF"
        "/tree/main/Ornith-1.5-9B-uncensored.Q5_K_M.gguf"
    )
    assert cli._extract_quant_from_url(url) == "Q5_K_M"


def test_extract_quant_from_url_ignores_model_suffix(cli):
    """A model-name suffix such as ``i1`` in ``...OBLITERATED-i1`` must NOT be
    treated as an explicit quant: the deployer should still prompt for a
    quant (returns None) rather than trying to download a non-existent file.
    """
    url = "https://huggingface.co/mradermacher/Cyber-Ornith-1.5-9B-OBLITERATED-i1-GGUF/resolve/main/Cyber-Ornith-1.5-9B-OBLITERATED-i1.gguf"
    assert cli._extract_quant_from_url(url) is None


def test_extract_quant_from_url_real_quant_still_works(cli):
    """A genuine quant tag (e.g. ``Q3_K_M``) is preserved."""
    url = (
        "https://huggingface.co/mradermacher/ornith-GGUF"
        "/resolve/main/ornith.Q3_K_M.gguf"
    )
    assert cli._extract_quant_from_url(url) == "Q3_K_M"


def test_extract_quant_from_url_lowercased(cli):
    """All-lowercase quants (e.g. ``q4_k_m``) are returned as-is."""
    url = (
        "https://huggingface.co/mradermacher/ornith-GGUF"
        "/resolve/main/ornith.q4_k_m.gguf?download=true"
    )
    assert cli._extract_quant_from_url(url) == "q4_k_m"


def test_extract_quant_from_url_no_quant(cli):
    """When no quant tag is present in the URL, return None."""
    url = "https://huggingface.co/mradermacher/ornith-GGUF/resolve/main/ornith.gguf"
    assert cli._extract_quant_from_url(url) is None


def test_requested_gguf_filename_preserves_dash_separator(cli):
    """When the URL provides the exact stem, the original separator between the
    model name and the quant tag is preserved verbatim (a dash in
    ``i1-Q4_K_M``), not rebuilt with a dot. This is the fix for the 404 where
    ``...i1.Q4_K_M.gguf`` was requested instead of ``...i1-Q4_K_M.gguf``.
    """
    filename = cli._requested_gguf_filename(
        "Cyber-Ornith-1.5-9B-OBLITERATED-i1-GGUF",
        "Q4_K_M",
        None,
        "Cyber-Ornith-1.5-9B-OBLITERATED-i1-Q4_K_M",
    )
    assert filename == "Cyber-Ornith-1.5-9B-OBLITERATED-i1-Q4_K_M.gguf"


def test_requested_gguf_filename_preserves_dot_separator(cli):
    """A stem whose model name and quant are dot-separated (e.g. ``base.quant``)
    is also preserved verbatim when provided by the URL.
    """
    filename = cli._requested_gguf_filename(
        "mradermacher/ornith-GGUF",
        "Q3_K_L",
        None,
        "ornith.Q3_K_L",
    )
    assert filename == "ornith.Q3_K_L.gguf"


def test_requested_gguf_filename_no_original_falls_back_to_dot(cli):
    """Without an original stem, the filename is reconstructed as
    ``{base}.{quant}.gguf`` (dot separator) for backward compatibility.
    """
    filename = cli._requested_gguf_filename(
        "mradermacher/ornith-GGUF",
        "Q3_K_L",
        None,
        None,
    )
    assert filename == "ornith.Q3_K_L.gguf"


def test_requested_gguf_filename_no_quant_ignores_original(cli):
    """When no quant is requested, the original stem is ignored and the
    derived filename (which lists repo files) is used instead.
    """
    filename = cli._requested_gguf_filename(
        "mradermacher/ornith-GGUF",
        None,
        None,
        "ornith.Q3_K_L",
    )
    assert filename == "ornith.gguf"


def test_extract_gguf_filename_from_url_full_stem(cli):
    """The full GGUF filename stem (without .gguf) is extracted from the URL,
    preserving case and the dash separator between model name and quant.
    """
    url = (
        "https://huggingface.co/mradermacher/Cyber-Ornith-1.5-9B-OBLITERATED-i1-GGUF/"
        "resolve/main/Cyber-Ornith-1.5-9B-OBLITERATED-i1-Q4_K_M.gguf?download=true"
    )
    assert (
        cli._extract_gguf_filename_from_url(url)
        == "Cyber-Ornith-1.5-9B-OBLITERATED-i1-Q4_K_M"
    )


def test_extract_gguf_filename_from_url_simple(cli):
    """A simple stem (no quant) is extracted verbatim."""
    url = "https://huggingface.co/mradermacher/ornith/resolve/main/ornith.gguf"
    assert cli._extract_gguf_filename_from_url(url) == "ornith"


def test_extract_gguf_filename_from_url_none(cli):
    """A URL without a GGUF filename returns None."""
    assert cli._extract_gguf_filename_from_url("mradermacher/ornith-GGUF") is None


def test_requested_gguf_filename_case_is_not_lowercased(cli):
    """The constructed filename must keep the case; a previous bug lowercased
    the quant (``q3_k_l``) which 404s on HuggingFace.
    """
    filename = cli._requested_gguf_filename(
        "mradermacher/Ornith-1.5-9B-uncensored-GGUF", "Q3_K_L"
    )
    assert filename == "Ornith-1.5-9B-uncensored.Q3_K_L.gguf"


def test_extract_quant_from_url_q4_k_m_in_version_dotted_name(cli):
    # Regression: the version dot (1.5) must not be mistaken for the quant.
    # Q4_K_M appears in the last stem component -> recognized quant.
    url = (
        "https://huggingface.co/Cyber-Ornith-1.5-9B-OBLITERATED/"
        "resolve/main/Cyber-Ornith-1.5-9B-OBLITERATED.i1-Q4_K_M.gguf"
    )
    assert cli._extract_quant_from_url(url) == "Q4_K_M"


def test_extract_quant_from_url_model_suffix_only_is_not_quant(cli):
    # A non-quant component (e.g. "i1") is not a quant tag -> None.
    url = (
        "https://huggingface.co/Cyber-Ornith-1.5-9B-OBLITERATED/"
        "resolve/main/Cyber-Ornith-1.5-9B-OBLITERATED.i1.gguf"
    )
    assert cli._extract_quant_from_url(url) is None
