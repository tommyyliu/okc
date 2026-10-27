"""Constant-only preparation and whole-layer stream scheduling."""

import torch
import torch.nn.functional as F
from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from .spec import N_EXPERTS, TOP_K
from .projections import _project
from .attention import _attention
from .boundaries import _norm, _residual_boundary
from .experts import _marlin_bank, _experts
from .config import DIRECT_SINGLE_EXPERTS

def prepare(weights):
    device = weights["qkv_gate"].device
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    if DIRECT_SINGLE_EXPERTS:
        # Preserve original packing independently of the Marlin loader.
        raw_bank = {name: weights[name].clone().contiguous() for name in
                    ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs")}
    state = {
        "weights": {k: weights[k].contiguous() for k in
                    ("pre_ffn_norm", "pre_expert_norm", "post_ffn_norm", "next_pre_attn_norm")},
        "dense": {k: weights[k].contiguous() for k in ("qkv_gate", "o", "pooler", "unpooler")},
        "dense_t": {k: weights[k].T.contiguous() for k in ("pooler", "unpooler")},
        "router": router.contiguous(),
        "router_hi_t": hi.T.contiguous(),
        "router_lo_t": lo.T.contiguous(),
        "routing": FusedMuseSparkRouting(
            expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
            router_output_multiplier=0.1767766952966369, router_score_division_eps=1e-15),
        "rope_freq": 1. / (500000. ** (torch.arange(0, 64, 2, device=device).float() / 64)),
        "post_expert_w": (1. + weights["post_expert_norm"].float()).to(torch.bfloat16),
        "bank": _marlin_bank(weights),
        "side_stream": torch.cuda.Stream(device),
    }
    for name, key in (("attn", "attn_residual_gate"), ("ffn", "ffn_residual_gate")):
        gate = weights[key].float() / 0.3
        beta = torch.sigmoid(gate)
        state[name + "_alpha"] = torch.sqrt(torch.clamp(torch.sigmoid(-gate) * (1. + beta), min=1e-3))
        state[name + "_beta"] = beta
    if DIRECT_SINGLE_EXPERTS:
        state["raw_bank"] = raw_bank
    return state

def kernel_fn(data, state):
    tokens = data["config"]["tokens"]
    attn_output = _project(_attention(data, state), state, "o")
    h, stream, router_input = _residual_boundary(data["stream"], attn_output, state)
    main, side = torch.cuda.current_stream(), state["side_stream"]
    side.wait_stream(main)
    # Independent router and pooler branches overlap, including under CUDA graphs.
    pooled = _project(h, state, "pooler")
    x = torch.empty_like(pooled)
    _norm[(tokens,)](pooled, state["weights"]["pre_expert_norm"], x, 4096, 1e-5, 4096, num_warps=4)
    with torch.cuda.stream(side):
        if tokens <= 8:
            logits = F.linear(router_input, state["router"])
        else:
            logits = torch.mm(h, state["router_hi_t"], out_dtype=torch.float32)
            logits += torch.mm(h, state["router_lo_t"], out_dtype=torch.float32)
        topk_weights, ids = state["routing"](None, logits, TOP_K, True)
    main.wait_stream(side)
    if not torch.cuda.is_current_stream_capturing():
        topk_weights.record_stream(main)
        ids.record_stream(main)
        # The pooled/router input is created on main and consumed on side.
        h.record_stream(side)
        router_input.record_stream(side)
    local_ids = (ids - data["local_start"]) % N_EXPERTS
    expert_output = _experts(x, topk_weights, local_ids, state)
    ffn = _project(expert_output, state, "unpooler")
    hidden, stream, _ = _residual_boundary(stream, ffn, state, final=True)
    return hidden, stream, ids.to(torch.int32)

