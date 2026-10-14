"""
Context window estimation from hardware VRAM and model size.

The context length (``num_predict``) is the number of tokens the model can
reason over. It is bounded by the memory available after loading the model:

    vram_bytes - model_bytes = usable_bytes
    usable_bytes / bytes_per_token = context_length

``bytes_per_token`` is quantization-dependent: a ``q4_0`` model stores roughly
3.5 bytes per token, while ``q8_0`` stores close to 8. We therefore detect the
quantization from the model name/tags and pick an appropriate estimate, falling
back to a conservative default when it cannot be determined.
"""

from __future__ import annotations

from typing import Dict, Optional

# Bytes per token for common GGUF quantization levels. Values are measured
# averages from the Ornith-1.5-9B GGUF repo (mradermacher/Ornith-1.5-9B-uncensored-GGUF),
# derived from the actual on-disk size of each quantized model divided by its
# token count (~9.05B params).
#
# Measured (Ornith-1.5-9B, ~9.05B params):
#   q2_k   3.83GB  -> 0.425 GB/tok
#   q3_k   4.26GB  -> 0.473 GB/tok  (Q3_K_S)
#   q3k_m  4.62GB  -> 0.513 GB/tok  (Q3_K_M, also "q3_k")
#   q3_k   4.93GB  -> 0.541 GB/tok  (Q3_K_L)
#   q4_k   5.63GB  -> 0.626 GB/tok  (Q4_K_M)
#   q4_k   5.35GB  -> 0.594 GB/tok  (Q4_K_S)
#   q5_k   6.47GB  -> 0.712 GB/tok  (Q5_K_M)
#   q5_k   6.31GB  -> 0.700 GB/tok  (Q5_K_S)
#   q6_k   7.30GB  -> 0.805 GB/tok  (Q6_K)
#   q8_0  12.4GB  -> 1.30  GB/tok  (Q8_0)
#   f16   17.9GB  -> 2.00  GB/tok  (f16)
QUANT_BYTES_PER_TOKEN: Dict[str, float] = {
    "q2_k_m": 0.43,
    "q3_k_m": 0.49,
    "q3k_m": 0.51,
    "q4_0": 0.50,
    "q4_k_s": 0.62,
    "q4_i": 0.50,
    "q5_0": 0.55,
    "q5_k_m": 0.70,
    "q6_k_m": 0.81,
    "q8_0": 1.30,
    "f16": 2.0,
}

# Exact on-disk bytes/token for each quant (from the actual GGUF file sizes),
# used to derive the true parameter-count footprint of a downloaded GGUF.
# Values match the HF model card sizes of the Cyber-Ornith quants.
EXACT_QUANT_BPT = {
    "iq1_s": 2.74 * (1024**3),
    "iq1_m": 2.88 * (1024**3),
    "iq2_xs": 3.1 * (1024**3),
    "iq2_s": 3.43 * (1024**3),
    "iq3_xs": 4.24 * (1024**3),
    "iq3_s": 4.37 * (1024**3),
    "iq4_xs": 5.2 * (1024**3),
    "q2_k_s": 3.7 * (1024**3),
    "q2_k": 3.83 * (1024**3),
    "q3_k_s": 4.26 * (1024**3),
    "q3_k_m": 4.62 * (1024**3),
    "q3_k_l": 4.93 * (1024**3),
    "q4_0": 5.33 * (1024**3),
    "q4_1": 5.81 * (1024**3),
    "q4_k_s": 5.35 * (1024**3),
    "q4_k_m": 5.63 * (1024**3),
    "q5_k_s": 6.31 * (1024**3),
    "q5_k_m": 6.47 * (1024**3),
    "q6_k": 7.36 * (1024**3),
}

# Conservative default: assume ~4 bytes/token (a middle-ground quantization).
DEFAULT_BYTES_PER_TOKEN = 4.0 * (1024**3)

# Safety margin (fraction of VRAM reserved, not used for context). Ollama
# reserves headroom for KV-cache fragmentation and the working set.
VRAM_RESERVE = 0.25

# Default VRAM overhead subtracted from the reported device capacity on
# Windows. On Linux/macOS this is ``0`` (the OS hands the GPU process the full
# dedicated pool).
#
# On Windows the WDDM driver and the desktop compositor (``dwm.exe``) claim
# dedicated GPU memory that the GPU process cannot use. Your own GPU Task
# Manager shows the real overhead: ``dwm.exe`` alone reserves ~0.84 GB, while
# a typical desktop session (VS Code, browsers, shell, WDDM buffers) consumes
# ~0.6 GB more - a total of ~1.4 GB against the 10 GB reported by an RTX 3080.
# We therefore reserve 1.5 GB, which also covers the WDDM driver's hidden
# buffer and leaves headroom for background processes.
#
# This is an estimate, not a measurement: VRAM usage cannot be monitored
# reliably (Ollama's ``OLLAMA_GPU_OVERHEAD`` is ignored by the llama-server
# backend, see https://github.com/ollama/ollama/issues/18679). Adjust it to
# match your observed ``dwm.exe`` + background-process footprint if you run a
# heavier desktop environment.
vram_windows_overhead = 1_570_000_000  # ~1.5 GB


def _detect_bytes_per_token(tags: Optional[list]) -> float:
    """Return a bytes-per-token estimate for the given quantization tags.

    Args:
        tags: List of model tags (e.g. ``["q4_0", "context_length=8192"]``).

    Returns:
        The estimated bytes per token.
    """
    if not tags:
        return DEFAULT_BYTES_PER_TOKEN

    # Iterate over the known table (longest keys first) so we always match the
    # most specific quantization, e.g. "q3k_m" before "q3".
    for key in sorted(QUANT_BYTES_PER_TOKEN, key=len, reverse=True):
        lower = key.lower()
        for tag in tags:
            if lower in tag.lower():
                return QUANT_BYTES_PER_TOKEN[key]
    return DEFAULT_BYTES_PER_TOKEN


def _model_size_bytes(
    model_id: str,
    gguf_path: Optional[str],
    model_info=None,
    tags: Optional[list] = None,
) -> Optional[int]:
    """Return the on-disk size of the model in bytes, or None.

    Resolution order:
      1. The local GGUF file (downloaded during ``deploy``) - exact size.
      2. ``model_info.size`` / ``file_size`` metadata (Hugging Face) - closest
         to the GGUF bytes.

    Args:
        model_id: The model identifier.
        gguf_path: Optional local path to the GGUF file.
        model_info: Optional Hugging Face model info object/dict.
        tags: Optional quantization tags.

    Returns:
        The model size in bytes, or None if it cannot be determined.
    """
    if gguf_path and (gguf_path.endswith((".gguf", ".bin"))):
        try:
            import os

            return os.path.getsize(gguf_path)
        except OSError:
            return None

    if model_info is not None:
        # Hugging Face ModelInfo proto exposes ``size`` (bytes on disk) or
        # ``file_size`` (download size). Either is a good proxy for GGUF bytes.
        size = getattr(model_info, "size", None)
        if not size:
            size = getattr(model_info, "file_size", None)
        if size:
            return int(size)

    return None


def estimate_context(
    vram_bytes: Optional[int],
    model_size_bytes: Optional[int],
    tags: Optional[list] = None,
    min_context: Optional[int] = None,
    vram_windows_overhead: Optional[int] = None,
) -> Optional[int]:
    """Estimate the context window size from VRAM and model size.

    This is the core calculation:

        context = (vram_bytes - model_size_bytes) / bytes_per_token

    On Windows, ``vram_windows_overhead`` (when provided) accounts for the
    memory claimed by the WDDM driver and desktop compositor that the GPU
    process cannot use; it is subtracted from the reported VRAM before the
    model size is removed.

    The result is never smaller than ``min_context``: a configured minimum is
    enforced via ``max(context, min_context)`` so the estimate can only grow.

    Args:
        vram_bytes: Total VRAM available on the target device (bytes).
        model_size_bytes: Size of the model weights on disk (bytes).
        tags: Optional quantization tags used to refine the
            bytes-per-token estimate.
        min_context: Optional minimum context length (tokens) enforced as a
            lower bound via ``max()``.
        vram_windows_overhead: Optional VRAM (bytes) reserved for the Windows
            OS/compositor. Defaults to :data:`vram_windows_overhead` when the
            platform is Windows and no value is provided.

    Returns:
        The estimated context length in tokens, or ``None`` when the estimate
        cannot be computed (missing VRAM or model size). The ``tags`` argument
        is required for the bytes-per-token estimate; when the model size is
        unknown (e.g. a plain model ID whose GGUF was never downloaded) the
        estimate is ``None`` so the caller can keep its explicit default.
    """
    import platform

    if vram_bytes is None or vram_bytes <= 0:
        return None
    if model_size_bytes is None or model_size_bytes <= 0:
        return None

    # On Windows, subtract the OS/compositor footprint that the GPU process
    # cannot use. Auto-select the documented default when the caller omits it.
    if vram_windows_overhead is None:
        vram_windows_overhead = (
            vram_windows_overhead if platform.system() == "Windows" else 0
        )
    usable_bytes = vram_bytes - model_size_bytes - vram_windows_overhead
    if usable_bytes <= 0:
        # The model (+ OS overhead) fills the entire VRAM - no room for context.
        return 0

    # ``QUANT_BYTES_PER_TOKEN`` values are already in bytes/token (e.g. a
    # q3_K model stores ~0.5 bytes per token), matching the byte-scale
    # ``usable_bytes``.
    bpt = _detect_bytes_per_token(tags) or DEFAULT_BYTES_PER_TOKEN
    context = int(usable_bytes / bpt)
    if min_context is not None and context < min_context:
        context = min_context
    return context


def format_context(context: Optional[int]) -> str:
    """Format a context length for display.

    Args:
        context: The context length in tokens.

    Returns:
        A human-readable string (e.g. ``"4.00K"``, ``"8192"``, ``"unlimited"``).
    """
    if context is None:
        return "unlimited"
    if context == 0:
        return "0"
    if context >= 1_000_000:
        return f"{context / 1_000_000:.2f}M"
    if context >= 1000:
        return f"{context / 1000:.2f}K"
    return str(context)
