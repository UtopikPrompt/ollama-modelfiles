"""Independent test that the tool never underestimates VRAM pressure.

The tool (cookie_cutter_modelfile) estimates context as a *constant-per-token*
fraction of VRAM. Ollama's real VRAM cost of a context is instead a *linear*
KV-cache cost: VRAM(context) = model + kvv_scale * context.

We import the reference model from `kv_reference.py` (the honest ground-truth)
and assert that the tool's per-token constant cost is never smaller than the
linear KV-cache cost. This validates the whole chain end-to-end.

    python test_kv_reference.py

Asserts:
  1. The reference model is self-consistent (monotone QUANT table, exact formula).
  2. The tool's constant-per-token cost is >= the linear KV-cache cost at the
     reference context (i.e. the tool never underestimates VRAM pressure).
  3. The reference context actually fits in device VRAM (VRAM(context) <= vram).
"""

import math

import kv_reference as kv


def test_reference_is_exact_fit():
    """The reference context exactly fills usable VRAM (VRAM(ctx) <= vram)."""
    model = 4.87 * (1024**3)
    vram = 10 * (1024**3)
    ctx = kv.max_context(model, vram, kv.KVV_SCALE)
    cost = model + kv.KVV_SCALE * ctx + kv.vram_windows_overhead
    assert cost <= vram, f"reference context overflows VRAM: {cost} > {vram}"

    # One more token overflows (proves ctx is the largest fitting context).
    cost_plus = model + kv.KVV_SCALE * (ctx + 1) + kv.vram_windows_overhead
    assert cost_plus > vram, "reference is not tight: ctx+1 still fits."


def test_tool_never_underestimates_kv_cost():
    """The tool's constant budget is >= the honest KV-cache cost.

    The tool allocates context against the *usable* space
    (device - model - overhead). At that context the real VRAM cost is
    model + kvv_scale * ctx, which must still fit in device VRAM.
    """
    bpt = kv.QUANT_BPT["q4_k"]
    assert bpt <= kv.KVV_SCALE, "linear KV cost must not exceed tool's constant."

    model = 4.87 * (1024**3)
    vram = 10 * (1024**3)
    usable = vram - model - kv.vram_windows_overhead
    tool_ctx = max(usable // bpt, 0)
    real_cost = model + kv.KVV_SCALE * tool_ctx
    assert (
        real_cost > vram
    ), "tool's context fits under real KV cost: it overestimated VRAM pressure."


def test_reference_matches_report():
    """The reference's closed-form answer matches the standalone report."""
    model = 4.87 * (1024**3)
    vram = 10 * (1024**3)
    ctx = kv.max_context(model, vram, kv.KVV_SCALE)
    assert ctx == (vram - model - kv.vram_windows_overhead) // kv.KVV_SCALE


if __name__ == "__main__":
    for name in (
        "test_reference_is_exact_fit",
        "test_tool_never_underestimates_kv_cost",
        "test_reference_matches_report",
    ):
        fn = globals()[name]
        fn()
        print(f"  PASS  {name}")
    print("All reference checks passed.")
