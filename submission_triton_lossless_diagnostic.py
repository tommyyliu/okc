# okc: problem=muse_bf16_proj
"""Diagnostic copy of submission 5; GPU work and launch settings are unchanged.

Blocks of 256 weights use an 8-bit sign/mantissa plane and 4-bit exponent
deltas when possible, or a full 8-bit exponent plane otherwise. Every BF16
bit is reconstructed before FP32 GEMV or BF16 tensor-core multiplication.
Only weights are prepared; every activation is consumed inside kernel_fn.
Prints compiler metadata after compressed projection launches. No tensor
values, input-dependent work, timer changes, or reference imports are added.
"""

import json
import re
import torch
import triton
import triton.language as tl


@triton.jit
def _gemv(X, W, Y, P, N: tl.constexpr, K: tl.constexpr, S: tl.constexpr,
          BN: tl.constexpr, BK: tl.constexpr):
    """One token: coalesced weight reads and FP32 reduction over K."""
    ns = tl.program_id(0) * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    ks = split * (K // S) + tl.arange(0, BK)
    end = (split + 1) * (K // S)
    acc = tl.full((BN,), 0, tl.float32)
    for _ in range(tl.cdiv(K // S, BK)):
        x = tl.load(X + ks, ks < end, other=0).to(tl.float32)
        w = tl.load(W + ns[:, None] * K + ks[None, :],
                    (ns[:, None] < N) & (ks[None, :] < end),
                    other=0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
        ks += BK
    if S == 1:
        tl.store(Y + ns, acc, ns < N)
    else:
        tl.store(P + split * N + ns, acc, ns < N)


@triton.jit
def _gemm(X, WH, WL, Y, P, M: tl.constexpr, N: tl.constexpr,
          K: tl.constexpr, S: tl.constexpr, ROUTER: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """BF16 tensor cores; independent FP32 high/low router accumulators."""
    ms = tl.program_id(0) * BM + tl.arange(0, BM)
    ns = tl.program_id(1) * BN + tl.arange(0, BN)
    split = tl.program_id(2)
    ks = split * (K // S) + tl.arange(0, BK)
    high = tl.full((BM, BN), 0, tl.float32)
    low = tl.full((BM, BN), 0, tl.float32)
    for _ in range(K // (S * BK)):
        x = tl.load(X + ms[:, None] * K + ks[None, :],
                    ms[:, None] < M, other=0)
        offsets = ns[None, :] * K + ks[:, None]
        wh = tl.load(WH + offsets, ns[None, :] < N, other=0)
        high = tl.dot(x, wh, high)
        if ROUTER:
            wl = tl.load(WL + offsets, ns[None, :] < N, other=0)
            low = tl.dot(x, wl, low)
        ks += BK
    if ROUTER:
        high += low
    offsets = ms[:, None] * N + ns[None, :]
    mask = (ms[:, None] < M) & (ns[None, :] < N)
    if S == 1:
        tl.store(Y + offsets, high, mask)
    else:
        tl.store(P + split * M * N + offsets, high, mask)


@triton.jit
def _reduce(P, Y, TOTAL: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
    offsets = tl.program_id(0) * B + tl.arange(0, B)
    splits = tl.arange(0, S)
    values = tl.load(P + splits[:, None] * TOTAL + offsets[None, :],
                     offsets[None, :] < TOTAL, other=0)
    tl.store(Y + offsets, tl.sum(values, axis=0), offsets < TOTAL)


def _splits(tiles, k, limit=32):
    """A fixed occupancy heuristic, to be tuned using H200 graph timings."""
    splits = 1
    while tiles * splits < 256 and splits < limit and k % (splits * 2 * 256) == 0:
        splits *= 2
    return splits


def _prepare_selective(weights):
    weights = {name: w.contiguous() for name, w in weights.items()}
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    return {"weights": weights, "router_hi": hi, "router_lo": lo}


def _hybrid_kernel_fn(data, state):
    x, config = data["x"], data["config"]
    m, n, k = config["m"], config["n"], config["k"]
    layer = config["layer"]
    router = layer == "router"
    w = state["weights"][layer]

    # These paths beat SGLang in the first H200 check. Dispatch depends only
    # on layer and token count, and also covers nearby unscored batch sizes.
    use_custom = (router
                  or (m == 1 and layer in ("o", "pooler"))
                  or (layer == "unpooler" and 1 < m <= 8))
    if not use_custom:
        return torch.mm(x, w[:n].T)

    out = torch.empty((m, n), device=x.device,
                      dtype=torch.float32 if router else torch.bfloat16)

    if m == 1:
        splits = _splits(triton.cdiv(n, 4), k, limit=8)
        partial = (torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
                   if splits > 1 else out)
        _gemv[(triton.cdiv(n, 4), splits)](
            x, w, out, partial, n, k, splits, BN=4, BK=512, num_warps=4)
    else:
        bm = 16 if m <= 16 else (32 if m <= 32 else 64)
        bn, bk = 64, 64
        tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
        splits = _splits(tiles, k)
        partial = (torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
                   if splits > 1 else out)
        high = state["router_hi"] if router else w
        low = state["router_lo"] if router else w
        _gemm[(triton.cdiv(m, bm), triton.cdiv(n, bn), splits)](
            x, high, low, out, partial, m, n, k, splits, router,
            BM=bm, BN=bn, BK=bk, num_warps=4, num_stages=3)

    if splits > 1:
        _reduce[(triton.cdiv(m * n, 256),)](
            partial, out, m * n, splits, B=256, num_warps=4)
    return out


@triton.jit
def _gemv_full(X, W, Y, N: tl.constexpr, K: tl.constexpr,
               ROWS: tl.constexpr, BLOCK_K: tl.constexpr):
    ns = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    ks = tl.arange(0, BLOCK_K)
    x = tl.load(X + ks, ks < K, other=0).to(tl.float32)
    w = tl.load(W + ns[:, None] * K + ks[None, :],
                (ns[:, None] < N) & (ks[None, :] < K), other=0).to(tl.float32)
    # Neither inputs nor weights are rounded before multiplication.
    sums = tl.sum(w * x[None, :], axis=1)
    tl.store(Y + ns, sums, ns < N)


@triton.jit
def _stream_gemm(X, W, Y, P, M: tl.constexpr, N: tl.constexpr,
                 K: tl.constexpr, S: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # Adjacent programs consume the same weight tile for different token
    # tiles. This also works for irregular M and the sliced unpooler.
    mt = tl.cdiv(M, BM)
    tile = tl.program_id(0)
    ms = (tile % mt) * BM + tl.arange(0, BM)
    ns = (tile // mt) * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    chunk: tl.constexpr = tl.cdiv(K, S)
    ks = split * chunk + tl.arange(0, BK)
    end = tl.minimum((split + 1) * chunk, K)
    acc = tl.full((BM, BN), 0, tl.float32)
    for _ in range(tl.cdiv(chunk, BK)):
        x = tl.load(X + ms[:, None] * K + ks[None, :],
                    (ms[:, None] < M) & (ks[None, :] < end), other=0)
        w = tl.load(W + ns[None, :] * K + ks[:, None],
                    (ns[None, :] < N) & (ks[:, None] < end), other=0)
        acc = tl.dot(x, w, acc)
        ks += BK
    offsets = ms[:, None] * N + ns[None, :]
    mask = (ms[:, None] < M) & (ns[None, :] < N)
    if S == 1:
        tl.store(Y + offsets, acc, mask)
    else:
        tl.store(P + split * M * N + offsets, acc, mask)


# An offline tuner can replace this literal with measured choices. Keys are
# (layer, upper end of the token bucket). There is no input-value dispatch.
# Options: ("hybrid",), ("gemv", rows, warps),
#          ("gemm", BM, BN, BK, splits, warps, stages).
TUNED_CONFIGS = {}


def token_bucket(m):
    for upper in (1, 8, 16, 32, 64, 128, 256):
        if m <= upper:
            return upper
    return None


def default_config(config):
    m, n, k = config["m"], config["n"], config["k"]
    layer = config["layer"]
    if m == 1:
        return ("gemv", 4 if k <= 1024 else 1, 4)
    # Check ba0e8c407d054666 found wins for QKV through 16 tokens and pooler
    # through 32. Its full-unpooler and 32-token QKV paths lost to cuBLAS.
    # Use token ranges so neighboring batch sizes follow the same policy.
    if (layer == "qkv_gate" and m <= 16) or (layer == "pooler" and m <= 32):
        bm = 16 if m <= 16 else 32
        splits = _splits(triton.cdiv(m, bm) * triton.cdiv(n, 64), k)
        return ("gemm", bm, 64, 128, splits, 4, 3)
    return ("hybrid",)


def run_config(data, state, option):
    if option[0] == "hybrid":
        return _hybrid_kernel_fn(data, state)
    x, config = data["x"], data["config"]
    m, n, k = config["m"], config["n"], config["k"]
    layer = config["layer"]
    w = state["weights"][layer]
    out = torch.empty((m, n), device=x.device,
                      dtype=torch.float32 if layer == "router" else torch.bfloat16)
    if option[0] == "gemv":
        if m != 1:
            raise ValueError("gemv requires exactly one token")
        _, rows, warps = option
        _gemv_full[(triton.cdiv(n, rows),)](
            x, w, out, n, k, ROWS=rows, BLOCK_K=triton.next_power_of_2(k),
            num_warps=warps)
    elif option[0] == "gemm":
        if layer == "router":
            raise ValueError("BF16 streaming GEMM does not support FP32 router weights")
        _, bm, bn, bk, splits, warps, stages = option
        # Keep partition starts aligned to tensor-core K tiles.
        if k % (splits * bk):
            raise ValueError("K must be divisible by splits * BK")
        partial = (torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
                   if splits > 1 else out)
        _stream_gemm[(triton.cdiv(m, bm) * triton.cdiv(n, bn), splits)](
            x, w, out, partial, m, n, k, splits, BM=bm, BN=bn, BK=bk,
            num_warps=warps, num_stages=stages)
        if splits > 1:
            _reduce[(triton.cdiv(m * n, 256),)](
                partial, out, m * n, splits, B=256, num_warps=4)
    else:
        raise ValueError("unknown kernel option")
    return out


def _selective_kernel_fn(data, state):
    config = data["config"]
    key = (config["layer"], token_bucket(config["m"]))
    option = TUNED_CONFIGS.get(key, default_config(config))
    return run_config(data, state, option)


@triton.jit
def _exponent_ranges(W, Bases, Sizes):
    group = tl.program_id(0)
    lanes = tl.arange(0, 256)
    bits = tl.load(W + group * 256 + lanes).to(tl.uint16, bitcast=True).to(tl.uint32)
    exponent = (bits >> 7) & 255
    lo = tl.min(exponent, axis=0)
    hi = tl.max(exponent, axis=0)
    compact = hi - lo <= 15
    # 256 is an explicit raw-block marker, outside the exponent range.
    tl.store(Bases + group, tl.where(compact, lo, 256))
    tl.store(Sizes + group, tl.where(compact, 384, 512))


@triton.jit
def _pack_lossless(W, Payload, Offsets, Bases):
    group = tl.program_id(0)
    lanes = tl.arange(0, 256)
    bits = tl.load(W + group * 256 + lanes).to(tl.uint16, bitcast=True).to(tl.uint32)
    base = tl.load(Bases + group).to(tl.uint32)
    start = tl.load(Offsets + group)
    sign_mantissa = (bits & 127) | ((bits >> 8) & 128)
    tl.store(Payload + start + lanes, sign_mantissa.to(tl.uint8))
    if base < 256:
        # One writer per byte: its low and high nibbles encode adjacent
        # weights. The last 128 lanes do not write a packed exponent byte.
        pairs = lanes * 2
        even_bits = tl.load(W + group * 256 + pairs, lanes < 128, other=0)
        odd_bits = tl.load(W + group * 256 + pairs + 1, lanes < 128, other=0)
        even = (even_bits.to(tl.uint16, bitcast=True).to(tl.uint32) >> 7) & 255
        odd = (odd_bits.to(tl.uint16, bitcast=True).to(tl.uint32) >> 7) & 255
        code = (even - base) | ((odd - base) << 4)
        tl.store(Payload + start + 256 + lanes, code.to(tl.uint8), lanes < 128)
    else:
        exponent = (bits >> 7) & 255
        tl.store(Payload + start + 256 + lanes, exponent.to(tl.uint8))


@triton.jit
def _decode_lossless(Payload, Offsets, Bases, groups, lanes, valid):
    start = tl.load(Offsets + groups, valid, other=0)
    base = tl.load(Bases + groups, valid, other=256).to(tl.uint32)
    compact = base < 256
    sm = tl.load(Payload + start + lanes, valid, other=0).to(tl.uint32)
    index = tl.where(compact, lanes // 2, lanes)
    code = tl.load(Payload + start + 256 + index, valid, other=0).to(tl.uint32)
    delta = (code >> ((lanes & 1) * 4)) & 15
    exponent = tl.where(compact, base + delta, code)
    bits = (sm & 127) | ((sm & 128) << 8) | (exponent << 7)
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


def compress_weight(w):
    """Untimed, weight-only preparation. Return (payload, offsets, bases, ratio)."""
    n, k = w.shape
    if w.dtype != torch.bfloat16 or not w.is_contiguous() or k % 256:
        return None
    groups = n * k // 256
    # These challenge matrices are below 2 GiB; 32-bit byte offsets suffice.
    if 2 * w.numel() >= 2 ** 31:
        return None
    bases = torch.empty(groups, device=w.device, dtype=torch.int16)
    sizes = torch.empty(groups, device=w.device, dtype=torch.int32)
    _exponent_ranges[(groups,)](w, bases, sizes, num_warps=4)
    ends = torch.cumsum(sizes, dim=0, dtype=torch.int32)
    # Weight-dependent allocation is allowed in prepare, never in kernel_fn.
    total_bytes = int(ends[-1].item())
    ratio = (total_bytes + groups * 6) / (w.numel() * 2)
    offsets = torch.empty_like(ends)
    offsets[0] = 0
    offsets[1:] = ends[:-1]
    payload = torch.empty(total_bytes, device=w.device, dtype=torch.uint8)
    _pack_lossless[(groups,)](w, payload, offsets, bases, num_warps=4)
    return payload, offsets, bases, ratio


def prepare(weights):
    state = _prepare_selective(weights)
    packed = {}
    for layer, w in state["weights"].items():
        if layer != "router":
            representation = compress_weight(w)
            # Wide exponent distributions remain supported by the raw-block
            # format. Avoid using it for projections without a byte saving.
            if representation is not None and representation[3] <= 0.90:
                packed[layer] = representation
    state["compressed"] = packed
    return state


@triton.jit
def _lossless_gemv(X, Payload, Offsets, Bases, Y,
                   N: tl.constexpr, K: tl.constexpr, ROWS: tl.constexpr,
                   GROUPS_K: tl.constexpr):
    ns = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    kg = tl.arange(0, GROUPS_K)
    lanes = tl.arange(0, 256)
    groups = ns[:, None, None] * (K // 256) + kg[None, :, None]
    valid = (ns[:, None, None] < N) & (kg[None, :, None] < K // 256)
    w = _decode_lossless(Payload, Offsets, Bases, groups, lanes[None, None, :], valid)
    w = tl.reshape(w, (ROWS, GROUPS_K * 256)).to(tl.float32)
    ks = tl.arange(0, GROUPS_K * 256)
    x = tl.load(X + ks, ks < K, other=0).to(tl.float32)
    sums = tl.sum(w * x[None, :], axis=1)
    tl.store(Y + ns, sums, ns < N)


@triton.jit
def _lossless_gemm(X, Payload, Offsets, Bases, Y, P,
                   M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                   S: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                   BK: tl.constexpr):
    mt = tl.cdiv(M, BM)
    tile = tl.program_id(0)
    ms = (tile % mt) * BM + tl.arange(0, BM)
    ns = (tile // mt) * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    k0 = split * (K // S)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for _ in range(K // (S * BK)):
        ks = k0 + kk
        x = tl.load(X + ms[:, None] * K + ks[None, :], ms[:, None] < M, other=0)
        # BK=128 and split boundaries align to 128: each tile lies within
        # one 256-element compressed group for each weight row.
        groups = ns[:, None] * (K // 256) + k0 // 256
        lanes = (k0 % 256) + kk[None, :]
        w = _decode_lossless(Payload, Offsets, Bases, groups, lanes, ns[:, None] < N)
        acc = tl.dot(x, tl.trans(w), acc)
        k0 += BK
    offsets = ms[:, None] * N + ns[None, :]
    mask = (ms[:, None] < M) & (ns[None, :] < N)
    if S == 1:
        tl.store(Y + offsets, acc, mask)
    else:
        tl.store(P + split * M * N + offsets, acc, mask)


def _report_codegen(compiled, layer, m, n, k):
    ptx = compiled.asm.get("ptx", "")
    ptx = re.sub(r"/\*.*?\*/", "", ptx, flags=re.S)
    ptx = re.sub(r"//[^\n]*", "", ptx)
    counts = {}
    for opcode in re.findall(r"^\s*(?:@!?%[\w$]+\s+)?(ld\.(?:global|local)[\w.]*|st\.local[\w.]*|cp\.async[\w.]*)\s", ptx, re.M):
        counts[opcode] = counts.get(opcode, 0) + 1
    print("LOSSLESS_CODEGEN " + json.dumps({
        "layer": layer, "m": m, "n": n, "k": k,
        "kernel": compiled.name,
        "triton_version": triton.__version__, "torch_version": torch.__version__,
        "registers_per_thread": getattr(compiled, "n_regs", None),
        "n_spills_raw": getattr(compiled, "n_spills", None),
        "shared_bytes": getattr(compiled.metadata, "shared", None),
        "static_memory_opcodes": counts,
        "num_warps": getattr(compiled.metadata, "num_warps", None),
        "num_stages": getattr(compiled.metadata, "num_stages", None)}, sort_keys=True))


def kernel_fn(data, state):
    x, config = data["x"], data["config"]
    m, n, k = config["m"], config["n"], config["k"]
    layer = config["layer"]
    representation = state["compressed"].get(layer)
    # Preserve the validated fused router. Large prefill batches retain
    # cuBLAS; this experiment targets decode and short verify blocks.
    if representation is None or m > 256:
        return _selective_kernel_fn(data, state)
    payload, offsets, bases, _ = representation
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    if m == 1:
        rows = 4 if k <= 1024 else 1
        compiled = _lossless_gemv[(triton.cdiv(n, rows),)](
            x, payload, offsets, bases, out, n, k, ROWS=rows,
            GROUPS_K=triton.next_power_of_2(k // 256), num_warps=4)
    else:
        bm = 16 if m <= 16 else (32 if m <= 32 else (64 if m <= 64 else 128))
        bn, bk = 64, 128
        tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
        splits = _splits(tiles, k, limit=16)
        partial = (torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
                   if splits > 1 else out)
        compiled = _lossless_gemm[(tiles, splits)](
            x, payload, offsets, bases, out, partial, m, n, k, splits,
            BM=bm, BN=bn, BK=bk, num_warps=8 if bm >= 128 else 4, num_stages=3)
        if splits > 1:
            _reduce[(triton.cdiv(m * n, 256),)](
                partial, out, m * n, splits, B=256, num_warps=4)
    _report_codegen(compiled, layer, m, n, k)
    return out
