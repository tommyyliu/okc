"""Lossless NVFP4 repacking and the SGLang Marlin expert path."""

import types
import torch
import triton
import triton.language as tl
from sgl_kernel.scalar_type import scalar_types
from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import fused_post_expert_norm_weighted_sum
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
from .spec import N_EXPERTS, TOP_K, LOCAL_EXPERTS, POST_NORM_EPS, BLOCK_SIZE_M_CANDIDATES
from .config import SMALL_EXPERT_FINALIZE, DIRECT_SINGLE_EXPERTS
from .nvfp4_gemv import _single_token_experts


@triton.jit
def _small_expert_finalize(Y, ROUTE_W, IDS, NORM_W, OUT):
    t = tl.program_id(0)
    s = tl.arange(0, 8)
    d = tl.arange(0, 4096)
    expert = tl.load(IDS + t * 8 + s)
    local = (expert >= 0) & (expert < 32)
    # Marlin does not initialize remote slots. Mask their loads before any
    # arithmetic: a later zero routing weight would not remove a NaN.
    y = tl.load(Y + (t * 8 + s[:, None]) * 4096 + d[None, :],
                local[:, None], other=0.).to(tl.float32)
    w = tl.load(NORM_W + d).to(tl.float32)
    scaled = (y * w[None, :]).to(tl.bfloat16).to(tl.float32)
    inv = tl.rsqrt(tl.sum(scaled * scaled, 1) / 4096 + 1e-8)
    route = tl.load(ROUTE_W + t * 8 + s)
    route = tl.where(local, route, 0.)
    weighted = scaled * inv[:, None] * route[:, None]
    tl.store(OUT + t * 4096 + d, tl.sum(weighted, 0))

def _marlin_bank(weights):
    """The local experts in Marlin's layout, via SGLang's own loader step."""
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }

def _experts(x, topk_weights, topk_ids, state):
    """MuseSpark1xFp4Experts._marlin_expert_outputs and its post-expert-norm finalize."""
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = bank["expert_map"][topk_ids].contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    if DIRECT_SINGLE_EXPERTS and num_tokens == 1:
        y = _single_token_experts(x, local_ids, topk_weights, state)
        out = torch.empty_like(x)
        _small_expert_finalize[(num_tokens,)](
            y, topk_weights, local_ids, state["post_expert_w"], out, num_warps=8)
        return out
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    if SMALL_EXPERT_FINALIZE and num_tokens <= 16:
        out = torch.empty_like(x)
        _small_expert_finalize[(num_tokens,)](
            y, topk_weights, local_ids, state["post_expert_w"], out, num_warps=8)
        return out
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)

