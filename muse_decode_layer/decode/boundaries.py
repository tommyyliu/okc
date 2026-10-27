"""Expert-input RMSNorm and the two residual/norm boundaries."""

import torch
import triton
import triton.language as tl

@triton.jit
def _norm(X, W, Y, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    x = tl.load(X + t * N + d, d < N, other=0).to(tl.float32)
    w = tl.load(W + d, d < N, other=0).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, 0) / N + EPS) * (1. + w)
    tl.store(Y + t * N + d, y, d < N)

@triton.jit
def _boundary(STREAM, BRANCH, ALPHA, BETA, POST_W, PRE_W,
               HIDDEN_OUT, STREAM_OUT, ROUTER_OUT,
               WIDTH: tl.constexpr, SCALED: tl.constexpr,
               WRITE_ROUTER: tl.constexpr):
    t = tl.program_id(0)
    d = tl.arange(0, 8192)
    branch = tl.load(BRANCH + t * WIDTH + d, d < WIDTH, other=0).to(tl.float32)
    if SCALED:
        post = tl.load(POST_W + d).to(tl.float32)
        branch = (branch * (1. + post)).to(tl.bfloat16).to(tl.float32)
    branch *= tl.rsqrt(tl.sum(branch * branch, 0) / 8192 + 1e-8)
    old = tl.load(STREAM + t * 8192 + d)
    alpha = tl.load(ALPHA + d)
    beta = tl.load(BETA + d)
    stream = alpha * old + beta * branch
    tl.store(STREAM_OUT + t * 8192 + d, stream)
    pre = tl.load(PRE_W + d).to(tl.float32)
    hidden = stream * tl.rsqrt(tl.sum(stream * stream, 0) / 8192 + 1e-5) * (1. + pre)
    tl.store(HIDDEN_OUT + t * 8192 + d, hidden)
    if WRITE_ROUTER:
        tl.store(ROUTER_OUT + t * 8192 + d, hidden)

def _residual_boundary(stream, branch, state, final=False):
    w = state["weights"]
    prefix = "ffn" if final else "attn"
    hidden = torch.empty(stream.shape, dtype=torch.bfloat16, device=stream.device)
    updated = torch.empty_like(stream)
    write_router = not final and stream.shape[0] <= 8
    router = torch.empty_like(stream) if write_router else hidden
    _boundary[(stream.shape[0],)](
        stream, branch, state[prefix + "_alpha"], state[prefix + "_beta"],
        w["post_ffn_norm"], w["next_pre_attn_norm" if final else "pre_ffn_norm"],
        hidden, updated, router, branch.shape[1], final, write_router,
        num_warps=8, enable_fp_fusion=False)
    return hidden, updated, router

