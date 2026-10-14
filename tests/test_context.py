"""
Tests for VRAM-based context estimation.
"""

import platform

import pytest

from cookie_cutter_modelfile.context import (
    DEFAULT_BYTES_PER_TOKEN,
    estimate_context,
    vram_windows_overhead,
)


class TestEstimateContext:
    """Tests for estimate_context."""

    def test_returns_none_when_vram_missing(self):
        """estimate_context returns None when VRAM is not provided."""
        assert estimate_context(None, 1000) is None

    def test_returns_none_when_vram_zero(self):
        """estimate_context returns None when VRAM is zero."""
        assert estimate_context(0, 1000) is None

    def test_returns_none_when_vram_negative(self):
        """estimate_context returns None when VRAM is negative."""
        assert estimate_context(-100, 1000) is None

    def test_returns_none_when_model_missing(self):
        """estimate_context returns None when model size is not provided."""
        assert estimate_context(10_000_000_000, None) is None

    def test_returns_none_when_model_zero(self):
        """estimate_context returns None when model size is zero."""
        assert estimate_context(10_000_000_000, 0) is None

    def test_basic_calculation(self):
        """Basic context calculation: usable_bytes / bytes_per_token."""
        vram = 10_000_000_000
        model = 2_000_000_000
        # Default 4.0 bytes per token; no overhead on non-Windows.
        usable = vram - model
        assert estimate_context(vram, model, vram_windows_overhead=0) == int(
            usable / DEFAULT_BYTES_PER_TOKEN
        )

    def test_usable_zero_returns_zero(self):
        """Returns 0 when the model alone fills all the VRAM."""
        vram = 1_000_000
        model = 1_000_000
        result = estimate_context(vram, model, vram_windows_overhead=0)
        assert result == 0

    def test_negative_usable_returns_zero(self):
        """Returns 0 when VRAM cannot even fit the model."""
        vram = 1_000_000
        model = 2_000_000
        result = estimate_context(vram, model, vram_windows_overhead=0)
        assert result == 0

    def test_min_context_floor(self):
        """min_context raises the result to the configured floor."""
        vram = 10_000_000_000
        model = 8_000_000_000
        # Natural estimate is small; the floor forces it up.
        result = estimate_context(vram, model, min_context=65536)
        assert result >= 65536

    def test_min_context_ignored_when_natural_is_larger(self):
        """min_context does not lower the natural estimate."""
        vram = 10_000_000_000
        model = 1_000_000_000
        natural = estimate_context(vram, model)
        result = estimate_context(vram, model, min_context=natural)
        assert result == natural

    def test_windows_overhead_applied_by_default_on_windows(self):
        """On Windows, the platform overhead is subtracted automatically."""
        if platform.system() != "Windows":
            pytest.skip("Test only valid on Windows.")
        vram = 10_000_000_000
        model = 1_000_000_000
        expected = vram - model - vram_windows_overhead
        assert estimate_context(vram, model) == int(expected / DEFAULT_BYTES_PER_TOKEN)

    def test_windows_overhead_overridable(self):
        """The vram_windows_overhead parameter overrides the platform default."""
        vram = 10_000_000_000
        model = 1_000_000_000
        overhead = 500_000_000
        expected = vram - model - overhead
        result = estimate_context(vram, model, vram_windows_overhead=overhead)
        assert result == int(expected / DEFAULT_BYTES_PER_TOKEN)

    def test_windows_overhead_not_applied_on_macos(self):
        """On macOS the Windows overhead is not applied."""
        if platform.system() == "Windows":
            pytest.skip("Test only valid on macOS.")
        vram = 10_000_000_000
        model = 1_000_000_000
        expected = vram - model
        result = estimate_context(vram, model)
        assert result == int(expected / DEFAULT_BYTES_PER_TOKEN)


class TestConstants:
    """Tests for the context constants."""

    def test_windows_overhead_is_around_1_5_gb(self):
        """vram_windows_overhead should be ~1.5 GB (1,570,000,000 bytes)."""
        assert abs(vram_windows_overhead - 1_570_000_000) < 10_000_000

    def test_default_bytes_per_token_is_4(self):
        """DEFAULT_BYTES_PER_TOKEN should be 4.0."""
        assert DEFAULT_BYTES_PER_TOKEN == 4.0
