"""Per-shape launch settings, with conservative defaults for nearby sizes.

Edit one (layer, M) entry to isolate a scored configuration. Option formats:
  ("torch",)                         BF16 cuBLAS fallback
  ("torch_transposed",)              BF16 cuBLAS with prepared [K,N] weights
  ("torch_reversed",)                (W @ X.T).T, native [N,M] backing tensor
  ("torch_transposed_reversed",)     same using prepared [K,N] weight storage
  ("gemv", rows, warps)               one-token FP32 reduction
  ("gemv_chunked", rows, warps, chunk_k) one-token chunked FP32 reduction
  ("router_direct", tokens, warps)    full-K FP32 router, no split or reducer
  (family, BM, BN, BK, splits, warps, stages)
Families: gemm (streaming), weight_major (streaming with reversed operands),
dot (BF16 / router hi+lo), orientation (unsplit weight-major), tma,
weight_major_tiled (prepared tiles), orientation_native ([N,M] backing),
router_interleaved (one wider dot with independent hi/lo accumulations).
dot_reduce32 keeps the dot producer and uses 32 outputs per reduction CTA.
router_cluster uses the splits field for CTAs per cluster; it has no global
partial buffer or second kernel. Its final field is the compiler stages setting.
weight_major_unroll2 uses the original weight-major math with a loop unroll hint.
hopper_ws uses explicit buffer count in the final stages field.
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

# Isolated first experiment: large-batch pooler only, against measured PDL.
# 64 token slots x 64 output features, K=128, no split-K. Both launches were
# compiled in the archived Hopper audit. Check dc36c8ef363340e3 subsequently
# passed correctness but lost to SGLang at both scored shapes; keep as a control.
POOLER_TMA_CONFIGS = {
    **PDL_CONFIGS,
    ('pooler', 128): ('tma', 64, 64, 128, 1, 4, 3),
    ('pooler', 256): ('tma', 64, 64, 128, 1, 4, 3),
}

# Scheduling ablation: same tiles/pipeline, more independent K partitions.
POOLER_SPLITK_CONFIGS = {
    **PDL_CONFIGS,
    ('pooler', 128): ('tma', 64, 64, 128, 4, 4, 3),
    ('pooler', 256): ('tma', 64, 64, 128, 2, 4, 3),
}

# Round 3 is an experiment, not a promoted default. Its unchanged small GEMMs
# use the passing weight-major nearby configuration (67a1b145d83b4f89).
ROUND3_CONFIGS = {
    **COMBINED_CONFIGS,
    **{(layer, m): ("weight_major", 16 if m <= 16 else 32,
                    64, 128, 8 if layer == "qkv_gate" else 4, 4, 3)
       for layer in ("qkv_gate", "pooler") for m in (8, 16, 32)},
    **{(layer, m): ("torch_transposed",)
       for layer in ("qkv_gate", "pooler") for m in (128, 256)},
    **{("unpooler", m): ("torch_transposed",) for m in (16, 32, 128, 256)},
    **{("router", m): ("dot", 16 if m <= 16 else 32, 32, 64, 16, 4, 3)
       for m in (8, 16, 32)},
    ("router", 128): ("dot", 64, 64, 64, 16, 4, 3),
    **{(layer, 1): ("gemv_chunked", 1, 4, 1024)
       for layer in ("qkv_gate", "pooler", "unpooler", "router")},
}

# Consolidation choices selected from passing per-case measurements. The
# aggregate remains pending until the user supplies the keepers check result.
KEEPER_CONFIGS = {
    **COMBINED_CONFIGS,
    **{(layer, m): ("weight_major", 16 if m <= 16 else 32,
                    64, 128, 8 if layer == "qkv_gate" else 4, 4, 3)
       for layer in ("qkv_gate", "pooler") for m in (8, 16, 32)},
    **{("pooler", m): ("torch_transposed",) for m in (128, 256)},
    **{("unpooler", m): ("torch_transposed",) for m in (16, 32, 128, 256)},
    ("router", 128): ("dot", 64, 64, 64, 16, 4, 3),
}

ROUND4_CONFIGS = {
    **KEEPER_CONFIGS,
    **{(layer, m): ("weight_major_tiled", 16 if m <= 16 else 32,
                    64, 128, 8 if layer == "qkv_gate" else 4, 4, 3)
       for layer in ("qkv_gate", "pooler") for m in (8, 16, 32)},
    **{("qkv_gate", m): ("torch_reversed",) for m in (128, 256)},
    **{("unpooler", m): ("torch_transposed_reversed",) for m in (16, 32, 128, 256)},
    **{("o", m): ("orientation_native", 16 if m <= 16 else 32, 64, 128, 1, 4, 4)
       for m in (8, 16, 32)},
    **{("router", m): ("router_interleaved", *KEEPER_CONFIGS[("router", m)][1:])
       for m in (8, 16, 32, 128, 256)},
}

ROUND5_CONFIGS = {
    **KEEPER_CONFIGS,
    **{(layer, m): ("weight_major_unroll2", *KEEPER_CONFIGS[(layer, m)][1:])
       for layer in ("qkv_gate", "pooler") for m in (8, 16, 32)},
    **{(layer, 1): ("gemv", 1, 8)
       for layer in ("qkv_gate", "pooler", "unpooler", "router")},
    **{("router", m): ("router_direct", 2, 8) for m in (8, 16, 32)},
}

# Selected earlier wins plus round-five GEMV8; no measured aggregate yet.
BANKED_CONFIGS = {
    **KEEPER_CONFIGS,
    **{(layer, 1): ("gemv", 1, 8)
       for layer in ("qkv_gate", "pooler", "unpooler", "router")},
}

# Diagnostic: keep the producer exactly fixed while exposing more reducer CTAs.
# B=32 instead of B=256; four reducer warps and PDL remain unchanged.
ROUTER_REDUCE_CONFIGS = {
    **BANKED_CONFIGS,
    **{("router", m): ("dot_reduce32", *BANKED_CONFIGS[("router", m)][1:])
       for m in (8, 16, 32)},
}

ROUTER_CLUSTER_CONFIGS = {
    **ROUTER_REDUCE_CONFIGS,
    **{("router", m): ("router_cluster", 16 if m <= 16 else 32, 16, 256, 8, 4, 1)
       for m in (8, 16, 32)},
}

# Same small-GEMM tiles and splits, with descriptor loads and WGMMA.
SMALL_TMA_CONFIGS = {
    **ROUTER_REDUCE_CONFIGS,
    **{(layer, m): ("tma", *ROUTER_REDUCE_CONFIGS[(layer, m)][1:])
       for layer in ("qkv_gate", "pooler") for m in (8, 16, 32)},
}

# Separate scheduling probes: more K partitions on the six small GEMMs and
# sixteen warps on the four existing one-row GEMVs. Arithmetic is unchanged.
OCCUPANCY_CONFIGS = {
    **ROUTER_REDUCE_CONFIGS,
    **{(layer, m): (*BANKED_CONFIGS[(layer, m)][:4],
                    BANKED_CONFIGS[(layer, m)][4] * 2,
                    *BANKED_CONFIGS[(layer, m)][5:])
       for layer in ("qkv_gate", "pooler") for m in (8, 16, 32)},
    **{(layer, 1): ("gemv", 1, 16)
       for layer in ("qkv_gate", "pooler", "unpooler", "router")},
}

# Round-seven confirmation: retain only the strongest new per-case signals.
ROUND7_CONFIGS = {
    **ROUTER_REDUCE_CONFIGS,
    ("router", 1): ("gemv", 1, 16),
    ("qkv_gate", 32): ("tma", *BANKED_CONFIGS[("qkv_gate", 32)][1:]),
}

# Compare TMA pipeline depth at fixed tiles/splits against the round-seven
# selection. These remain experiments until the remote checks complete.
ROUND8_STAGES2_CONFIGS = {
    **ROUND7_CONFIGS,
    **{(layer, m): ("tma", *BANKED_CONFIGS[(layer, m)][1:6], 2)
       for layer in ("qkv_gate", "pooler") for m in (8, 16, 32)},
}
ROUND8_STAGES4_CONFIGS = {
    **ROUND8_STAGES2_CONFIGS,
    **{(layer, m): (*ROUND8_STAGES2_CONFIGS[(layer, m)][:6], 4)
       for layer in ("qkv_gate", "pooler") for m in (8, 16, 32)},
}

# Double the expert tile, halve BK and double splits: the producer grid,
# per-CTA weight bytes and K-loop iteration count match the selected geometry.
ROUND8_WIDE_CONFIGS = {
    **ROUND7_CONFIGS,
    **{(layer, m): ("tma", 16 if m <= 16 else 32, 128, 64,
                    BANKED_CONFIGS[(layer, m)][4] * 2, 4, 3)
       for layer in ("qkv_gate", "pooler") for m in (8, 16, 32)},
}

ROUND8_EARLY_CONFIGS = {
    key: ((option[0] + "_early", *option[1:])
          if option[0] in ("dot", "dot_reduce32", "weight_major", "tma")
          and option[4] > 1 else option)
    for key, option in ROUND7_CONFIGS.items()
}

ROUND8_ROUTER_CONFIGS = {
    **ROUND7_CONFIGS,
    ("router", 1): ("gemv", 1, 32),
    **{("router", m): ("dot_reduce32_warp1", *ROUND7_CONFIGS[("router", m)][1:])
       for m in (8, 16, 32)},
}

# Prepared after quota exhaustion; GPU correctness and performance unmeasured.
QKV_SPLIT4_CONFIGS = {
    **ROUND7_CONFIGS,
    ("qkv_gate", 32): ("tma", 32, 64, 128, 4, 4, 3),
}

# Four scored cases exercise the new engine; all others keep selected choices.
HOPPER_WS_CONFIGS = {
    **BANKED_CONFIGS,
    **{(layer, m): ("hopper_ws", 64, 64, 128,
                    2 if layer == "qkv_gate" and m == 128 else 1, 4, 2)
       for layer in ("qkv_gate", "pooler") for m in (128, 256)},
}
HOPPER_OVERLAP_CONFIGS = {
    **ROUTER_REDUCE_CONFIGS,
    **{(layer, m): ("hopper_overlap", *HOPPER_WS_CONFIGS[(layer, m)][1:])
       for layer in ("qkv_gate", "pooler") for m in (128, 256)},
}
TRACE_RESOURCES = PRESET in ("hopper_ws", "router_reduce", "router_cluster", "small_tma", "hopper_overlap", "occupancy", "round7", "round8_stages2", "round8_stages4", "round8_wide", "round8_early", "round8_router")


CONFIGS = {"pdl": PDL_CONFIGS, "combined": COMBINED_CONFIGS, "tma": TMA_CONFIGS,
           "pooler_tma": POOLER_TMA_CONFIGS, "pooler_splitk": POOLER_SPLITK_CONFIGS,
           "round3": ROUND3_CONFIGS, "keepers": KEEPER_CONFIGS, "round4": ROUND4_CONFIGS,
           "round5": ROUND5_CONFIGS, "banked": BANKED_CONFIGS,
           "hopper_ws": HOPPER_WS_CONFIGS, "router_reduce": ROUTER_REDUCE_CONFIGS,
           "router_cluster": ROUTER_CLUSTER_CONFIGS, "small_tma": SMALL_TMA_CONFIGS,
           "hopper_overlap": HOPPER_OVERLAP_CONFIGS, "occupancy": OCCUPANCY_CONFIGS,
           "round7": ROUND7_CONFIGS,
           "round8_stages2": ROUND8_STAGES2_CONFIGS,
           "round8_stages4": ROUND8_STAGES4_CONFIGS,
           "round8_wide": ROUND8_WIDE_CONFIGS, "round8_early": ROUND8_EARLY_CONFIGS,
           "round8_router": ROUND8_ROUTER_CONFIGS, "qkv_split4": QKV_SPLIT4_CONFIGS}


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
    if preset == "qkv_split4":
        option = fallback_config(config, "round7")
        if layer == "qkv_gate" and 16 < m <= 32:
            return ("tma", 32, 64, 128, 4, 4, 3)
        return option
    if preset == "round8_router":
        if layer == "router" and m == 1:
            return ("gemv", 1, 32)
        option = fallback_config(config, "round7")
        if layer == "router" and 1 < m <= 32:
            return ("dot_reduce32_warp1", *option[1:])
        return option
    if preset == "round8_early":
        option = fallback_config(config, "round7")
        if (option[0] in ("dot", "dot_reduce32", "weight_major", "tma")
                and option[4] > 1):
            return (option[0] + "_early", *option[1:])
        return option
    if preset == "round8_wide":
        option = fallback_config(config, "round7")
        if layer in ("qkv_gate", "pooler") and 1 < m <= 32:
            return ("tma", bm, 128, 64, option[4] * 2, 4, 3)
        return option
    if preset in ("round8_stages2", "round8_stages4"):
        option = fallback_config(config, "round7")
        if layer in ("qkv_gate", "pooler") and 1 < m <= 32:
            return ("tma", *option[1:6], 2 if preset == "round8_stages2" else 4)
        return option
    if preset == "round7":
        if layer == "router" and m == 1:
            return ("gemv", 1, 16)
        option = fallback_config(config, "router_reduce")
        if layer == "qkv_gate" and 16 < m <= 32:
            return ("tma", *option[1:])
        return option
    if preset == "occupancy":
        option = fallback_config(config, "router_reduce")
        if m == 1 and layer != "o":
            return ("gemv", 1, 16)
        if layer in ("qkv_gate", "pooler") and 1 < m <= 32:
            return (*option[:4], option[4] * 2, *option[5:])
        return option
    if preset == "hopper_overlap":
        if layer in ("qkv_gate", "pooler") and m > 32:
            option = fallback_config(config, "hopper_ws")
            return ("hopper_overlap", *option[1:])
        return fallback_config(config, "router_reduce")
    if preset == "small_tma":
        option = fallback_config(config, "router_reduce")
        if layer in ("qkv_gate", "pooler") and 1 < m <= 32:
            return ("tma", *option[1:])
        return option
    if preset == "router_cluster":
        if layer == "router" and 1 < m <= 32:
            return ("router_cluster", bm, 16, 256, 8, 4, 1)
        return fallback_config(config, "router_reduce")
    if preset == "router_reduce":
        option = fallback_config(config, "banked")
        if layer == "router" and 1 < m <= 32:
            return ("dot_reduce32", *option[1:])
        return option
    if preset == "hopper_ws":
        if layer in ("qkv_gate", "pooler") and m > 32:
            return ("hopper_ws", 64, 64, 128, _splits(tiles, k, 128, 4, 128), 4, 2)
        return fallback_config(config, "banked")
    if preset == "banked":
        if m == 1 and layer != "o":
            return ("gemv", 1, 8)
        return fallback_config(config, "keepers")
    if preset == "round5":
        if m == 1 and layer != "o":
            return ("gemv", 1, 8)
        if layer in ("qkv_gate", "pooler") and 1 < m <= 32:
            return ("weight_major_unroll2", bm, 64, 128,
                    8 if layer == "qkv_gate" else 4, 4, 3)
        if layer == "router" and 1 < m <= 32:
            return ("router_direct", 2, 8)
        return fallback_config(config, "keepers")
    if preset in ("keepers", "round4"):
        if layer in ("qkv_gate", "pooler") and 1 < m <= 32:
            return ("weight_major_tiled" if preset == "round4" else "weight_major",
                    bm, 64, 128, 8 if layer == "qkv_gate" else 4, 4, 3)
        if layer == "qkv_gate" and m > 32:
            return ("torch_reversed",) if preset == "round4" else ("torch",)
        if (layer == "pooler" and m > 32) or (layer == "unpooler" and m > 8):
            family = ("torch_transposed_reversed" if preset == "round4" and layer == "unpooler"
                      else "torch_transposed")
            return (family,)
        if layer == "o" and 1 < m <= 32:
            return ("orientation_native" if preset == "round4" else "orientation",
                    bm, 64, 128, 1, 4, 4)
        if layer == "router" and m > 1:
            splits = 16 if 32 < m <= 128 else _splits(tiles, k)
            return ("router_interleaved" if preset == "round4" else "dot",
                    bm, 64, 64, splits, 4, 3)
    if preset == "round3":
        if m == 1 and layer != "o":
            return ("gemv_chunked", 1, 4, 1024)
        if layer in ("qkv_gate", "pooler") and 1 < m <= 32:
            return ("weight_major", bm, 64, 128,
                    8 if layer == "qkv_gate" else 4, 4, 3)
        if ((layer in ("qkv_gate", "pooler") and m > 32)
                or (layer == "unpooler" and m > 8)):
            return ("torch_transposed",)
        if layer == "o" and 1 < m <= 32:
            return ("orientation", bm, 64, 128, 1, 4, 4)
        if layer == "router" and 1 < m <= 32:
            return ("dot", bm, 32, 64, 16, 4, 3)
        if layer == "router" and 32 < m <= 128:
            return ("dot", 64, 64, 64, 16, 4, 3)
    if preset == "pooler_tma" and layer == "pooler" and 32 < m <= 256:
        return ("tma", 64, 64, 128, 1, 4, 3)
    if preset == "pooler_splitk" and layer == "pooler" and 32 < m <= 256:
        return ("tma", 64, 64, 128, _splits(tiles, k, 512, 8, 128), 4, 3)
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
