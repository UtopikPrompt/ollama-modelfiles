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
    "q2_k": 0.43,
    "q3_k": 0.49,
    "q3k_m": 0.51,
    "q4_0": 0.50,
    "q4_k": 0.62,
    "q4_i": 0.50,
    "q5_0": 0.55,
    "q5_k": 0.70,
    "q6_k": 0.81,
    "q8_0": 1.30,
    "f16": 2.0,
    "f8": 1.0,
}

# Conservative default: assume ~4 bytes/token (a middle-ground quantization).
DEFAULT_BYTES_PER_TOKEN = 4.0

# Safety margin (fraction of VRAM reserved, not used for context). Ollama
# reserves headroom for KV-cache fragmentation and the working set.
VRAM_RESERVE = 0.25


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
) -> Optional[int]:
    """Estimate the context window size from VRAM and model size.

    This is the core calculation:

        context = (vram_bytes - model_size_bytes) / bytes_per_token

    Args:
        vram_bytes: Total VRAM available on the target device (bytes).
        model_size_bytes: Size of the model weights on disk (bytes).
        tags: Optional quantization tags used to refine the
            bytes-per-token estimate.

    Returns:
        The estimated context length in tokens, or ``None`` when the estimate
        cannot be computed (missing VRAM or model size). The ``tags`` argument
        is required for the bytes-per-token estimate; when the model size is
        unknown (e.g. a plain model ID whose GGUF was never downloaded) the
        estimate is ``None`` so the caller can keep its explicit default.
    """
    if vram_bytes is None or vram_bytes <= 0:
        return None
    if model_size_bytes is None or model_size_bytes <= 0:
        return None

    usable_bytes = vram_bytes - model_size_bytes
    if usable_bytes <= 0:
        # The model fills the entire VRAM - no room for context.
        return 0

    bpt = _detect_bytes_per_token(tags) or DEFAULT_BYTES_PER_TOKEN
    context = int(usable_bytes / bpt)
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
