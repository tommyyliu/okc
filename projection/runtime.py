"""Preparation and dispatch. Keep activation-dependent work inside GPU kernels."""

import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor

from .configs import select_config
from .kernels.gemv import _gemv_full
from .kernels.gemm import _gemm, _stream_gemm, _weight_major_gemm
from .kernels.reduce import _reduce
from .kernels.tma import _tma_weight_major


def prepare(weights):
    weights = {name: w.contiguous() for name, w in weights.items()}
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    return {"weights": weights, "router_hi": hi, "router_lo": lo}


def run_config(data, state, option):
    x, config = data["x"], data["config"]
    m, n, k = config["m"], config["n"], config["k"]
    layer = config["layer"]
    router = layer == "router"
    w = state["weights"][layer]
    family = option[0]
    if family == "torch":
        if router:
            raise ValueError("router requires original FP32 GEMV or BF16 hi/lo GEMM")
        return torch.mm(x, w[:n].T)

    out = torch.empty((m, n), device=x.device,
                      dtype=torch.float32 if router else torch.bfloat16)
    if family == "gemv":
        if m != 1:
            raise ValueError("gemv requires exactly one token")
        _, rows, warps = option
        _gemv_full[(triton.cdiv(n, rows),)](
            x, w, out, n, k, ROWS=rows, BLOCK_K=triton.next_power_of_2(k),
            num_warps=warps)
        return out

    _, bm, bn, bk, splits, warps, stages = option
    if k % (splits * bk):
        raise ValueError("K must be divisible by splits * BK")
    if router and family != "dot":
        raise ValueError("router GEMM requires its BF16 hi/lo pair")
    if family == "orientation":
        if splits != 1:
            raise ValueError("orientation kernel does not support split-K")
        _weight_major_gemm[(triton.cdiv(n, bn), triton.cdiv(m, bm))](
            w, x, out, m, n, k, BM=bm, BN=bn, BK=bk,
            num_warps=warps, num_stages=stages)
        return out

    partial = (torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
               if splits > 1 else out)
    if family == "dot":
        high = state["router_hi"] if router else w
        low = state["router_lo"] if router else w
        _gemm[(triton.cdiv(m, bm), triton.cdiv(n, bn), splits)](
            x, high, low, out, partial, m, n, k, splits, router,
            BM=bm, BN=bn, BK=bk, num_warps=warps, num_stages=stages)
    elif family == "gemm":
        _stream_gemm[(triton.cdiv(m, bm) * triton.cdiv(n, bn), splits)](
            x, w, out, partial, m, n, k, splits, BM=bm, BN=bn, BK=bk,
            num_warps=warps, num_stages=stages)
    elif family == "tma":
        # Bind descriptors to current tensors, including cloned prepared state.
        wd = TensorDescriptor(w, [n, k], [k, 1], [bn, bk])
        xd = TensorDescriptor(x, [m, k], [k, 1], [bm, bk])
        _tma_weight_major[(triton.cdiv(m, bm) * triton.cdiv(n, bn), splits)](
            wd, xd, out, partial, m, n, k, splits, BM=bm, BN=bn, BK=bk,
            num_warps=warps, num_stages=stages)
    else:
        raise ValueError(f"unknown kernel family: {family}")
    if splits > 1:
        _reduce[(triton.cdiv(m * n, 256),)](
            partial, out, m * n, splits, B=256, num_warps=4, launch_pdl=True)
    return out


def kernel_fn(data, state):
    return run_config(data, state, select_config(data["config"]))
