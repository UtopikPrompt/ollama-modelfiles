"""Reference KV-cache sizing for the deployed q4_K_M model.

The tool (cookie_cutter_modelfile) estimates context as:
    usable = vram - model - overhead
    context = usable / bytes_per_token

This module instead models Ollama's *actual* VRAM cost of a context:
    VRAM(context) = model + kvv_scale * context
where kvv_scale is the real KV-cache overhead (Ollama: ~1.5-2.5 B/token).

We compute the largest context that fits in device VRAM, which is an honest
ground-truth answer to compare against the tool's estimate.

Functions
---------
max_context(model_size_bytes, vram_bytes, kvv_scale) -> int
tool_estimate(usable_bytes, bpt) -> int
QUANT_BPT: quant name -> bytes/token table.

Run standalone for a report:
    python _kv_reference.py --device 10 --model 4.87 --quant q4_k_m
"""

import argparse

# Ollama's real KV-cache overhead (B/token). Ollama reserves ~1.5-2.5x the context.
KVV_SCALE = 2.0

# Default fallback bytes/token when the requested quant is not in the table.
DEFAULT_BPT = 0.62

# Windows VRAM overhead reserved for non-context use (device memory, etc.).
vram_windows_overhead = 1_570_000_000

# q4_k_m bytes/token (from the package QUANT table).
MODEL_BYTES_PER_TOKEN = 0.62

# quant name -> bytes/token table (monotonic: q2 < q3 < q4 < q5 < q6, saturating at q8_0/f16).
QUANT_BPT = {
    "q2_k": 0.43,
    "q3_k": 0.49,
    "q3k_m": 0.51,
    "q4_0": 0.50,
    "q4_k": 0.62,
    "q5_k": 0.70,
    "q6_k": 0.81,
    "q8_0": 1.30,
    "f16": 2.0,
}


def max_context(model_size_bytes, vram_bytes, kvv_scale=KVV_SCALE):
    """Largest context that fits in device VRAM.

    The whole model plus its KV-cache plus the Windows VRAM overhead must fit in
    device VRAM:

        model + kvv_scale * context + overhead <= vram

    Solving for the largest integer context:

        context = floor((vram - model - overhead) / kvv_scale)

    The KV cost is linear in context, so this is exact (no bisection needed).
    """
    usable = max(vram_bytes - model_size_bytes - vram_windows_overhead, 0)
    if usable <= 0:
        return 0
    return usable // kvv_scale


def tool_estimate(usable_bytes, bpt):
    """The tool's own (constant-per-token) estimate: int(usable / bpt)."""
    return max(usable_bytes // bpt, 0)


def main():
    ap = argparse.ArgumentParser(description="Reference KV-cache sizing report.")
    ap.add_argument("--device", type=float, help="VRAM in GiB (default 10)")
    ap.add_argument("--model", type=float, help="model size in GiB (default 4.87)")
    ap.add_argument("--quant", default="q4_k_m", help="quant, selects bytes/token")
    ap.add_argument(
        "--kvv-scale",
        type=float,
        default=KVV_SCALE,
        help="KV-cache overhead (default 2.0)",
    )
    args = ap.parse_args()

    device = args.device * (1024**3)
    model = args.model * (1024**3)
    bpt = QUANT_BPT.get(args.quant, DEFAULT_BPT)

    tool = tool_estimate(device - model - vram_windows_overhead, bpt)
    truth = max_context(model, device, args.kvv_scale)

    print(
        f"device={args.device}GB  model={args.model}GB  quant={args.quant}  kvv_scale={args.kvv_scale}"
    )
    print(f"  tool estimate : {tool}")
    print(f"  reference    : {truth}")
    truth_gb = model / (1024**3) + args.kvv_scale * truth / (1024**3)
    usable_gb = args.device - args.model - 1.57
    print(
        f"  reference VRAM used by context: {truth_gb:.3f} GB (usable={usable_gb:.2f} GB)"
    )


if __name__ == "__main__":
    main()
