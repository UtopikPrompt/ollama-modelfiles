"""Human-friendly size parsing for config values.

Decimal (SI) units for ``K``/``M``/``G``/``T`` (10**3 .. 10**12), matching NVIDIA's
convention so ``"10GB"`` == 10 * 10**9 == 10564246528. Binary units
(``Ki``/``Mi``/``Gi``/``Ti``) use 1024-based multipliers. Bare integers work too.

Examples:
    parse_size_or_int("10GB") == 10564246528
    parse_size_or_int("64K")  == 64000
    parse_size_or_int("8Gi")  == 8 * 2**30
    parse_size_or_int("65536") == 65536
"""

from __future__ import annotations

import re

# Decimal multipliers for SI unit letters (uppercase/lowercase).
_DECIMAL = {
    "K": 10**3,
    "M": 10**6,
    "G": 10**9,
    "T": 10**12,
    "k": 10**3,
    "m": 10**6,
    "g": 10**9,
    "t": 10**12,
}

# Binary multipliers for binary unit labels.
_BINARY = {
    "Ki": 2**10,
    "Mi": 2**20,
    "Gi": 2**30,
    "Ti": 2**40,
    "KiB": 2**10,
    "MiB": 2**20,
    "GiB": 2**30,
    "TiB": 2**40,
}

# Bare binary single-letter labels (letter + optional trailing 'i').
_BINARY_LETTER = {
    "K": 2**10,
    "M": 2**20,
    "G": 2**30,
    "T": 2**40,
    "k": 2**10,
    "m": 2**20,
    "g": 2**30,
    "t": 2**40,
}

# Numeric-token regex: digits (optional decimal point) then an optional unit suffix.
_TOKEN_RE = re.compile(r"(\d+(?:\.\d+)?)([KMGTkmg]?(?:i)?|GB|GiB)?")


def parse_size_or_int(value) -> int:
    """Convert a human-friendly size string to an integer byte count.

    Args:
        value: A numeric value (int/float) or a size string such as
            ``"10GB"``, ``"64K"``, ``"8Gi"``, or a bare integer like ``"65536"``.

    Returns:
        The number of bytes, or ``0`` if parsing fails.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str):
        return 0

    text = value.strip()
    if not text:
        return 0

    match = _TOKEN_RE.match(text)
    if not match:
        return 0

    number = float(match.group(1))
    suffix = match.group(2) or ""
    multiplier = _multiplier(suffix)
    if multiplier is None:
        return 0
    return int(number * multiplier)


def _multiplier(suffix: str):
    """Return the byte multiplier for a unit suffix, or ``None`` if unknown."""
    if not suffix:
        return 1  # Bare number, no unit.
    # SI decimal letters (uppercase/lowercase).
    if suffix in _DECIMAL:
        return _DECIMAL[suffix]
    # Binary labels (e.g. Ki/Mi/Gi/Ti and KiB/GiB).
    if suffix in _BINARY:
        return _BINARY[suffix]
    # Bare SI-letters G (and GB == G == 10**9).
    if suffix in {"G", "GB"}:
        return 10**9
    # Bare binary single-letter labels.
    if suffix in _BINARY_LETTER:
        return _BINARY_LETTER[suffix]
    return None
