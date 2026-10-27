"""Single-token expert GEMVs over original NVFP4 weights, decoded in registers."""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_nvfp4(code, scale, global_scale):
    mag = code.to(tl.uint32) & 7
    # For codes 2..7: (1 + low_bit/2) * 2**(code//2 - 1).
    bits = (((mag >> 1) + 126) << 23) | ((mag & 1) << 22)
    normal = bits.to(tl.float32, bitcast=True)
    value = tl.where(mag < 2, mag.to(tl.float32) * .5, normal)
    value = tl.where((code & 8) != 0, -value, value)
    # Weights remain NVFP4 in storage. Only in-register values are expanded.
    return (value * (scale.to(tl.float32) * global_scale)).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _nvfp4_gate_up(X, IDS, W, SF, GS, INTER, ROWS: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    slot = tl.program_id(1)
    expert = tl.load(IDS + slot)
    if (expert >= 0) & (expert < 32):
        row = tile * ROWS + tl.arange(0, ROWS)
        lane = tl.arange(0, BK)
        gs = tl.load(GS + expert)
        gate_acc = tl.full((ROWS, BK), 0., tl.float32)
        up_acc = tl.full((ROWS, BK), 0., tl.float32)
        for start in range(4096 // BK):
            k = start * BK + lane
            x = tl.load(X + k).to(tl.float32)
            byte_offset = expert * (8192 * 2048) + row[:, None] * 2048 + k[None, :] // 2
            scale_offset = expert * (8192 * 256) + row[:, None] * 256 + k[None, :] // 16
            gate_packed = tl.load(W + byte_offset).to(tl.uint32)
            up_packed = tl.load(W + byte_offset + 4096 * 2048).to(tl.uint32)
            shift = (k[None, :] % 2) * 4
            gate_code = (gate_packed >> shift) & 15
            up_code = (up_packed >> shift) & 15
            gate_scale = tl.load(SF + scale_offset)
            up_scale = tl.load(SF + scale_offset + 4096 * 256)
            gate = _decode_nvfp4(gate_code, gate_scale, gs)
            up = _decode_nvfp4(up_code, up_scale, gs)
            gate_acc += gate * x[None, :]
            up_acc += up * x[None, :]
        # Preserve the BF16 gate/up GEMM boundary before SiLU, and the BF16
        # SwiGLU boundary before the second projection.
        gate = tl.sum(gate_acc, 1).to(tl.bfloat16).to(tl.float32)
        up = tl.sum(up_acc, 1).to(tl.bfloat16).to(tl.float32)
        inter = gate * tl.sigmoid(gate) * up
        tl.store(INTER + slot * 4096 + row, inter)


@triton.jit
def _nvfp4_down(INTER, IDS, W, SF, GS, Y, ROWS: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    slot = tl.program_id(1)
    expert = tl.load(IDS + slot)
    if (expert >= 0) & (expert < 32):
        row = tile * ROWS + tl.arange(0, ROWS)
        lane = tl.arange(0, BK)
        gs = tl.load(GS + expert)
        acc = tl.full((ROWS, BK), 0., tl.float32)
        for start in range(4096 // BK):
            k = start * BK + lane
            x = tl.load(INTER + slot * 4096 + k).to(tl.float32)
            byte_offset = expert * (4096 * 2048) + row[:, None] * 2048 + k[None, :] // 2
            scale_offset = expert * (4096 * 256) + row[:, None] * 256 + k[None, :] // 16
            packed = tl.load(W + byte_offset).to(tl.uint32)
            code = (packed >> ((k[None, :] % 2) * 4)) & 15
            scale = tl.load(SF + scale_offset)
            weight = _decode_nvfp4(code, scale, gs)
            acc += weight * x[None, :]
        tl.store(Y + slot * 4096 + row, tl.sum(acc, 1))


def _single_token_experts(x, local_ids, topk_weights, state):
    bank = state["raw_bank"]
    inter = torch.empty((8, 4096), dtype=x.dtype, device=x.device)
    y = torch.empty_like(inter)
    _nvfp4_gate_up[(512, 8)](
        x, local_ids, bank["w13"], bank["w13_sf"], bank["w13_gs"], inter,
        ROWS=8, BK=512, num_warps=8)
    _nvfp4_down[(512, 8)](
        inter, local_ids, bank["w2"], bank["w2_sf"], bank["w2_gs"], y,
        ROWS=8, BK=512, num_warps=8)
    return y
