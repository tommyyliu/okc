"""Dense BF16 projections and their current shape dispatch."""

import torch
import triton
import triton.language as tl

@triton.jit
def _gemv(X, W, Y, N: tl.constexpr, K: tl.constexpr,
          ROWS: tl.constexpr, BK: tl.constexpr):
    n = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    k = tl.arange(0, BK)
    x = tl.load(X + k, k < K, other=0).to(tl.float32)
    w = tl.load(W + n[:, None] * K + k[None, :],
                (n[:, None] < N) & (k[None, :] < K), other=0).to(tl.float32)
    tl.store(Y + n, tl.sum(w * x[None, :], 1), n < N)

@triton.jit
def _small_gemm(X, W, P, Y, M: tl.constexpr, N: tl.constexpr,
                K: tl.constexpr, S: tl.constexpr, BM: tl.constexpr,
                BN: tl.constexpr, BK: tl.constexpr):
    mt: tl.constexpr = tl.cdiv(M, BM)
    tile = tl.program_id(0)
    m = (tile % mt) * BM + tl.arange(0, BM)
    n = (tile // mt) * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    k = split * (K // S) + tl.arange(0, BK)
    acc = tl.full((BN, BM), 0, tl.float32)
    for _ in range(K // (S * BK)):
        w = tl.load(W + n[:, None] * K + k[None, :], n[:, None] < N, other=0)
        x = tl.load(X + m[None, :] * K + k[:, None], m[None, :] < M, other=0)
        acc = tl.dot(w, x, acc)
        k += BK
    offsets = m[None, :] * N + n[:, None]
    mask = (m[None, :] < M) & (n[:, None] < N)
    if S == 1:
        tl.store(Y + offsets, acc, mask)
    else:
        tl.extra.cuda.gdc_launch_dependents()
        tl.store(P + split * M * N + offsets, acc, mask)

@triton.jit
def _reduce_projection(P, Y, TOTAL: tl.constexpr, S: tl.constexpr,
                       BLOCK: tl.constexpr):
    tl.extra.cuda.gdc_wait()
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    s = tl.arange(0, S)
    x = tl.load(P + s[:, None] * TOTAL + i[None, :], i[None, :] < TOTAL, other=0)
    tl.store(Y + i, tl.sum(x, 0), i < TOTAL)

def _project(x, state, name):
    w = state["dense"][name]
    m, k = x.shape
    n = 1024 if name == "unpooler" and m <= 8 else w.shape[0]
    if m == 1:
        y = torch.empty((m, n), dtype=x.dtype, device=x.device)
        _gemv[(triton.cdiv(n, 4 if k <= 1024 else 1),)](
            x, w, y, n, k, ROWS=4 if k <= 1024 else 1,
            BK=triton.next_power_of_2(k), num_warps=4)
        return y
    small = (name in ("qkv_gate", "pooler") and m <= 32)
    orient = (name == "o" and m <= 32)
    sliced = name == "unpooler" and m <= 8
    if small or orient or sliced:
        bm, bn, bk = (16 if m <= 16 else 32), 64, (64 if sliced else 128)
        splits = 1 if orient else 16 if sliced else 8 if name == "qkv_gate" else 4
        y = torch.empty((m, n), dtype=x.dtype, device=x.device)
        partial = torch.empty((splits, m, n), dtype=torch.float32, device=x.device) if splits > 1 else y
        _small_gemm[(triton.cdiv(m, bm) * triton.cdiv(n, bn), splits)](
            x, w, partial, y, m, n, k, splits, bm, bn, bk,
            num_warps=4, num_stages=3 if splits > 1 else 4)
        if splits > 1:
            _reduce_projection[(triton.cdiv(m * n, 256),)](
                partial, y, m * n, splits, 256, num_warps=4, launch_pdl=True)
        return y
    if name in state["dense_t"]:
        return torch.mm(x, state["dense_t"][name][:, :n])
    return torch.mm(x, w[:n].T)

