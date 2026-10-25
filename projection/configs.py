"""Per-shape launch settings, with conservative defaults for nearby sizes.

Edit one (layer, M) entry to isolate a scored configuration. Option formats:
  ("torch",)                         BF16 cuBLAS fallback
  ("gemv", rows, warps)               one-token FP32 reduction
  (family, BM, BN, BK, splits, warps, stages)
Families: gemm (streaming), dot (BF16 / router hi+lo), orientation, tma.
BM always means tokens; BN means output features, even for weight-major math.

PDL preserves the measured 1.2310594x candidate. Combined and TMA have only
CPU/compile validation. These are inherited settings, not a completed sweep.
"""

from .shapes import SHAPES, SCORED_TOKENS, token_bucket

PRESET = "pdl"
# Exact-shape overrides emitted by the offline tuner; unswept cases retain defaults.
TUNED_CONFIGS = {}

PDL_CONFIGS = {
    ('qkv_gate', 1): ('gemv', 1, 4),
    ('qkv_gate', 8): ('gemm', 16, 64, 128, 8, 4, 3),
    ('qkv_gate', 16): ('gemm', 16, 64, 128, 8, 4, 3),
    ('qkv_gate', 32): ('torch',),
    ('qkv_gate', 128): ('torch',),
    ('qkv_gate', 256): ('torch',),
    ('o', 1): ('gemv', 4, 4),
    ('o', 8): ('torch',),
    ('o', 16): ('torch',),
    ('o', 32): ('torch',),
    ('o', 128): ('torch',),
    ('o', 256): ('torch',),
    ('pooler', 1): ('gemv', 1, 4),
    ('pooler', 8): ('gemm', 16, 64, 128, 4, 4, 3),
    ('pooler', 16): ('gemm', 16, 64, 128, 4, 4, 3),
    ('pooler', 32): ('gemm', 32, 64, 128, 4, 4, 3),
    ('pooler', 128): ('torch',),
    ('pooler', 256): ('torch',),
    ('unpooler', 1): ('gemv', 1, 4),
    ('unpooler', 8): ('dot', 16, 64, 64, 16, 4, 3),
    ('unpooler', 16): ('torch',),
    ('unpooler', 32): ('torch',),
    ('unpooler', 128): ('torch',),
    ('unpooler', 256): ('torch',),
    ('router', 1): ('gemv', 1, 4),
    ('router', 8): ('dot', 16, 64, 64, 32, 4, 3),
    ('router', 16): ('dot', 16, 64, 64, 32, 4, 3),
    ('router', 32): ('dot', 32, 64, 64, 32, 4, 3),
    ('router', 128): ('dot', 64, 64, 64, 32, 4, 3),
    ('router', 256): ('dot', 64, 64, 64, 16, 4, 3),
}

COMBINED_CONFIGS = {
    **PDL_CONFIGS,
    ('o', 8): ('orientation', 16, 64, 128, 1, 4, 4),
    ('o', 16): ('orientation', 16, 64, 128, 1, 4, 4),
    ('o', 32): ('orientation', 32, 64, 128, 1, 4, 4),
}

TMA_CONFIGS = {
    **COMBINED_CONFIGS,
    ('qkv_gate', 8): ('tma', 16, 64, 128, 4, 4, 3),
    ('qkv_gate', 16): ('tma', 16, 64, 128, 4, 4, 3),
    ('qkv_gate', 32): ('tma', 32, 64, 128, 4, 4, 3),
    ('qkv_gate', 128): ('tma', 64, 64, 128, 2, 4, 3),
    ('qkv_gate', 256): ('tma', 64, 64, 128, 1, 4, 3),
    ('pooler', 8): ('tma', 16, 64, 128, 2, 4, 3),
    ('pooler', 16): ('tma', 16, 64, 128, 2, 4, 3),
    ('pooler', 32): ('tma', 32, 64, 128, 2, 4, 3),
    ('pooler', 128): ('tma', 64, 64, 128, 1, 4, 3),
    ('pooler', 256): ('tma', 64, 64, 128, 1, 4, 3),
    ('unpooler', 16): ('tma', 16, 64, 128, 1, 4, 3),
    ('unpooler', 32): ('tma', 32, 64, 128, 1, 4, 3),
    ('unpooler', 128): ('tma', 64, 64, 128, 1, 4, 3),
    ('unpooler', 256): ('tma', 64, 64, 128, 1, 4, 3),
}

CONFIGS = {"pdl": PDL_CONFIGS, "combined": COMBINED_CONFIGS, "tma": TMA_CONFIGS}


def _splits(tiles, k, target=256, limit=32, alignment=256):
    splits = 1
    while tiles * splits < target and splits < limit and k % (splits * 2 * alignment) == 0:
        splits *= 2
    return splits


def fallback_config(config, preset):
    """Preserve previous range policies for non-scored and large correctness cases."""
    m, n, k = config["m"], config["n"], config["k"]
    layer = config["layer"]
    bm = 16 if m <= 16 else 32 if m <= 32 else 64
    tiles = ((m + bm - 1) // bm) * ((n + 63) // 64)
    if preset == "tma" and 1 < m <= 256 and (
            layer in ("qkv_gate", "pooler") or (layer == "unpooler" and m > 8)):
        return ("tma", bm, 64, 128, _splits(tiles, k, 128, 8, 128), 4, 3)
    if preset in ("combined", "tma") and layer == "o" and 1 < m <= 32:
        return ("orientation", bm, 64, 128, 1, 4, 4)
    if m == 1:
        return ("gemv", 4 if k <= 1024 else 1, 4)
    if (layer == "qkv_gate" and m <= 16) or (layer == "pooler" and m <= 32):
        return ("gemm", bm, 64, 128, _splits(tiles, k), 4, 3)
    if layer == "router" or (layer == "unpooler" and m <= 8):
        return ("dot", bm, 64, 64, _splits(tiles, k), 4, 3)
    return ("torch",)


def select_config(config):
    key = (config["layer"], config["m"])
    if key in TUNED_CONFIGS:
        return TUNED_CONFIGS[key]
    if key in CONFIGS[PRESET]:
        return CONFIGS[PRESET][key]
    return fallback_config(config, PRESET)
