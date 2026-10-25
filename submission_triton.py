# okc: problem=muse_bf16_proj
"""Triton candidate: GEMV, split-K GEMM, and fused hi/lo router GEMM.

Standalone submission. All activations and intermediate sums are computed in
kernel_fn; prepare only transforms weights. Accumulation is FP32 throughout.
"""

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


def prepare(weights):
    weights = {name: w.contiguous() for name, w in weights.items()}
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    return {"weights": weights, "router_hi": hi, "router_lo": lo}


def kernel_fn(data, state):
    x, config = data["x"], data["config"]
    m, n, k = config["m"], config["n"], config["k"]
    router = config["layer"] == "router"
    w = state["weights"][config["layer"]]
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
