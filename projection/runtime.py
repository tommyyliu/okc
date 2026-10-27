"""Preparation and dispatch. Keep activation-dependent work inside GPU kernels."""

import json
import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GluonTensorDescriptor

from .configs import CONFIGS, PRESET, TUNED_CONFIGS, TRACE_RESOURCES, select_config
from .kernels.gemv import _gemv_full, _gemv_chunked, _router_direct
from .kernels.gemm import _gemm, _stream_gemm, _weight_major_gemm, _weight_major_stream_gemm, _weight_major_unroll2_gemm
from .kernels.layout import _weight_major_tiled_gemm, _weight_major_native_gemm, _router_interleaved_gemm
from .kernels.reduce import _reduce
from .kernels.tma import _tma_weight_major
from .kernels.hopper_ws import _hopper_ws_gemm
from .kernels.hopper_overlap import _hopper_overlap_gemm
from .kernels.early import _gemm_early, _weight_major_stream_gemm_early, _tma_weight_major_early
from .kernels.router_cluster import _router_cluster


_RESOURCE_SEEN = set()


def _report_resources(compiled, layer, m, family):
    """Compiler metadata only: no tensor inspection, timing, or input caching."""
    if not TRACE_RESOURCES or compiled is None:
        return
    metadata = compiled.metadata
    key = (metadata.name, metadata.shared, getattr(compiled, "n_regs", None),
           getattr(compiled, "n_spills", None))
    if key in _RESOURCE_SEEN:
        return
    _RESOURCE_SEEN.add(key)
    print("OKC_RESOURCE " + json.dumps(dict(
        layer=layer, m=m, family=family, kernel=metadata.name,
        registers=getattr(compiled, "n_regs", None),
        spills=getattr(compiled, "n_spills", None), shared_bytes=metadata.shared,
        num_warps=metadata.num_warps)))


def _prepare_weights(weights):
    weights = {name: w.contiguous() for name, w in weights.items()}
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    return {"weights": weights, "router_hi": hi, "router_lo": lo}


def prepare(weights):
    state = _prepare_weights(weights)
    # Preset families can still be selected by nearby-size fallback even when
    # every scored entry is overridden by a tuning or diagnostic export.
    choices = [*CONFIGS[PRESET].items(), *TUNED_CONFIGS.items()]
    transposed_layers = {layer for (layer, _), option in choices
                         if option[0] in ("torch_transposed", "torch_transposed_reversed")}
    if transposed_layers:
        state["weights_transposed"] = {
            layer: state["weights"][layer].T.contiguous()
            for layer in transposed_layers}
    tiled_specs = {(layer, option[2], option[3])
                   for (layer, _), option in choices
                   if option[0] == "weight_major_tiled"}
    if tiled_specs:
        state["weights_tiled"] = {}
        for layer, bn, bk in tiled_specs:
            w = state["weights"][layer]
            n, k = w.shape
            if n % bn or k % bk:
                raise ValueError("tiled weights require N/BN and K/BK to be integral")
            state["weights_tiled"][(layer, bn, bk)] = (
                w.reshape(n // bn, bn, k // bk, bk).permute(0, 2, 1, 3).contiguous())
    if any(option[0] == "router_interleaved" for _, option in choices):
        state["router_pair"] = torch.stack(
            (state["router_hi"], state["router_lo"]), dim=1).reshape(
                -1, state["weights"]["router"].shape[1])
    return state


def run_config(data, state, option):
    x, config = data["x"], data["config"]
    m, n, k = config["m"], config["n"], config["k"]
    layer = config["layer"]
    router = layer == "router"
    w = state["weights"][layer]
    family = option[0]
    early = family.endswith("_early")
    family = family.removesuffix("_early")
    reduce_warps = 1 if family == "dot_reduce32_warp1" else 4
    if family == "dot_reduce32_warp1":
        family = "dot_reduce32"
    if family in ("torch", "torch_transposed", "torch_reversed", "torch_transposed_reversed"):
        if router:
            raise ValueError("router requires original FP32 GEMV or BF16 hi/lo GEMM")
        if family == "torch_transposed":
            return torch.mm(x, state["weights_transposed"][layer][:, :n])
        if family == "torch_reversed":
            return torch.mm(w[:n], x.T).T
        if family == "torch_transposed_reversed":
            return torch.mm(state["weights_transposed"][layer][:, :n].T, x.T).T
        return torch.mm(x, w[:n].T)

    out = torch.empty((n, m) if family == "orientation_native" else (m, n), device=x.device,
                      dtype=torch.float32 if router else torch.bfloat16)
    if family == "gemv":
        if m != 1:
            raise ValueError("gemv requires exactly one token")
        _, rows, warps = option
        compiled = _gemv_full[(triton.cdiv(n, rows),)](
            x, w, out, n, k, ROWS=rows, BLOCK_K=triton.next_power_of_2(k),
            num_warps=warps)
        _report_resources(compiled, layer, m, family)
        return out

    if family == "gemv_chunked":
        if m != 1:
            raise ValueError("gemv_chunked requires exactly one token")
        _, rows, warps, chunk_k = option
        compiled = _gemv_chunked[(triton.cdiv(n, rows),)](
            x, w, out, n, k, ROWS=rows, BLOCK_K=chunk_k,
            num_warps=warps)
        _report_resources(compiled, layer, m, family)
        return out

    if family == "router_direct":
        if not router:
            raise ValueError("router_direct requires original FP32 router weights")
        _, tokens, warps = option
        compiled = _router_direct[(n, triton.cdiv(m, tokens))](
            x, w, out, m, n, k, BM=tokens, BLOCK_K=triton.next_power_of_2(k),
            num_warps=warps)
        _report_resources(compiled, layer, m, family)
        return out

    _, bm, bn, bk, splits, warps, stages = option
    if k % (splits * bk):
        raise ValueError("K must be divisible by splits * BK")
    if family == "router_cluster":
        if not router or splits not in (2, 4, 8) or warps != 4:
            raise ValueError("router_cluster requires router, 2/4/8 CTAs and four warps")
        compiled = _router_cluster[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
            x, state["router_hi"], state["router_lo"], out, m, n, k,
            BM=bm, BN=bn, BK=bk, C=splits, num_ctas=splits,
            num_warps=warps, num_stages=stages)
        _report_resources(compiled, layer, m, family)
        return out
    if family == "dot_reduce32" and not router:
        raise ValueError("dot_reduce32 is a router reduction experiment")
    if router and family not in ("dot", "dot_reduce32", "router_interleaved"):
        raise ValueError("router GEMM requires its BF16 hi/lo pair")
    if family in ("orientation", "orientation_native"):
        if splits != 1:
            raise ValueError("orientation kernel does not support split-K")
        kernel = _weight_major_native_gemm if family == "orientation_native" else _weight_major_gemm
        compiled = kernel[(triton.cdiv(n, bn), triton.cdiv(m, bm))](
            w, x, out, m, n, k, BM=bm, BN=bn, BK=bk,
            num_warps=warps, num_stages=stages)
        _report_resources(compiled, layer, m, family)
        return out.T if family == "orientation_native" else out

    partial = (torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
               if splits > 1 else out)
    if family == "router_interleaved":
        if not router:
            raise ValueError("router_interleaved requires router FP32 weights split into BF16 hi/lo")
        compiled = _router_interleaved_gemm[(triton.cdiv(m, bm), triton.cdiv(n, bn), splits)](
            x, state["router_pair"], out, partial, m, n, k, splits,
            BM=bm, BN=bn, BK=bk, num_warps=warps, num_stages=stages)
    elif family in ("dot", "dot_reduce32"):
        high = state["router_hi"] if router else w
        low = state["router_lo"] if router else w
        kernel = _gemm_early if early else _gemm
        compiled = kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn), splits)](
            x, high, low, out, partial, m, n, k, splits, router,
            BM=bm, BN=bn, BK=bk, num_warps=warps, num_stages=stages)
    elif family == "gemm":
        compiled = _stream_gemm[(triton.cdiv(m, bm) * triton.cdiv(n, bn), splits)](
            x, w, out, partial, m, n, k, splits, BM=bm, BN=bn, BK=bk,
            num_warps=warps, num_stages=stages)
    elif family == "weight_major":
        kernel = _weight_major_stream_gemm_early if early else _weight_major_stream_gemm
        compiled = kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn), splits)](
            x, w, out, partial, m, n, k, splits, BM=bm, BN=bn, BK=bk,
            num_warps=warps, num_stages=stages)
    elif family == "weight_major_unroll2":
        compiled = _weight_major_unroll2_gemm[(triton.cdiv(m, bm) * triton.cdiv(n, bn), splits)](
            x, w, out, partial, m, n, k, splits, BM=bm, BN=bn, BK=bk,
            num_warps=warps, num_stages=stages)
    elif family == "weight_major_tiled":
        compiled = _weight_major_tiled_gemm[(triton.cdiv(m, bm) * triton.cdiv(n, bn), splits)](
            x, state["weights_tiled"][(layer, bn, bk)], out, partial, m, n, k, splits,
            BM=bm, BN=bn, BK=bk, num_warps=warps, num_stages=stages)
    elif family in ("hopper_ws", "hopper_overlap"):
        if warps != 4 or stages < 2:
            raise ValueError("hopper_ws requires four compute warps and at least two buffers")
        wd = GluonTensorDescriptor.from_tensor(
            w, [bn, bk], gl.NVMMASharedLayout.get_default_for([bn, bk], gl.bfloat16))
        xd = GluonTensorDescriptor.from_tensor(
            x, [bm, bk], gl.NVMMASharedLayout.get_default_for([bm, bk], gl.bfloat16))
        kernel = _hopper_overlap_gemm if family == "hopper_overlap" else _hopper_ws_gemm
        compiled = kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn), splits)](
            wd, xd, out, partial, m, n, k, splits, BM=bm, BN=bn, BK=bk,
            BUFFERS=stages, num_warps=warps, num_stages=1, maxnreg=96)
    elif family == "tma":
        # Bind descriptors to current tensors, including cloned prepared state.
        wd = TensorDescriptor(w, [n, k], [k, 1], [bn, bk])
        xd = TensorDescriptor(x, [m, k], [k, 1], [bm, bk])
        kernel = _tma_weight_major_early if early else _tma_weight_major
        compiled = kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn), splits)](
            wd, xd, out, partial, m, n, k, splits, BM=bm, BN=bn, BK=bk,
            num_warps=warps, num_stages=stages)
    else:
        raise ValueError(f"unknown kernel family: {family}")
    _report_resources(compiled, layer, m, family)
    if splits > 1:
        reduce_block = 32 if family == "dot_reduce32" else 256
        compiled = _reduce[(triton.cdiv(m * n, reduce_block),)](
            partial, out, m * n, splits, B=reduce_block, num_warps=reduce_warps, launch_pdl=True)
        _report_resources(compiled, layer, m, "reduce")
    return out


def kernel_fn(data, state):
    return run_config(data, state, select_config(data["config"]))
