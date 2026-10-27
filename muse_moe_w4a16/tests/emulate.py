"""CPU emulation of the prepared layout and the GEMM's decode/index math (no GPU).

  <venv with torch+triton>/bin/python tests/emulate.py [submission.py]

Runs prepare's per-matrix repack on random NVFP4 weights, then rebuilds each program's
A operand the way _weights/_activations do (same asm bit ops, same join/reshape/permute
order) and checks A @ B against dequantize(W) @ x.
"""
import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from nvfp4 import dequantize, random_weights  # noqa: E402

spec = importlib.util.spec_from_file_location("sub", sys.argv[1] if len(sys.argv) > 1 else HERE / "submission.py")
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)


def bf16_pairs(u32):
    """int64 tensor of 32-bit patterns -> float32 [..., 2] (low half, high half)."""
    lo = (u32 & 0xFFFF).to(torch.int32) << 16
    hi = ((u32 >> 16) & 0xFFFF).to(torch.int32) << 16
    return torch.stack((lo.view(torch.float32), hi.view(torch.float32)), -1)


def to_bits(pairs):
    """float32 [..., 2] (exact in BF16) -> int64 32-bit patterns."""
    b = pairs.to(torch.bfloat16).view(torch.int16).to(torch.int64) & 0xFFFF
    return b[..., 0] | (b[..., 1] << 16)


def scale_asm(s16):
    t = s16.to(torch.int64) & 0xFFFF
    b0, b1 = t & 0xFF, t >> 8
    sa = ((b0 | (b0 << 16)) * 16 + 0x76007600) & 0xFFFFFFFF
    sb = ((b1 | (b1 << 16)) * 16 + 0x76007600) & 0xFFFFFFFF
    return sa, sb


def decode_asm(w, sa, sb):
    w = w.to(torch.int64) & 0xFFFFFFFF
    m = 0x81C081C0
    r0 = w & m
    r1 = (w << 3) & m
    r2 = (w << 6) & m
    r3 = ((w >> 4) & 0x00C000C0) | ((w >> 5) & 0x01000100) | ((w << 1) & 0x80008000)
    out = []
    for r, s in ((r0, sa), (r1, sb), (r2, sa), (r3, sb)):
        p = bf16_pairs(r) * bf16_pairs(s)  # exact products
        out += [p[..., 0], p[..., 1]]
    return out  # o0..o7


def a_operand(words, scales, BN):
    sa, sb = scale_asm(scales)
    sa = sa[..., None].expand(words.shape)
    sb = sb[..., None].expand(words.shape)
    o = decode_asm(words, sa, sb)
    j = lambda a, b: torch.stack((a, b), -1)
    t = j(j(j(o[0], o[1]), j(o[2], o[3])), j(j(o[4], o[5]), j(o[6], o[7])))
    t = t.reshape(BN // 16, 8, 4, 4, 4, 2, 2, 2).permute(0, 6, 1, 2, 4, 7, 3, 5)
    return t.reshape(BN, 256)


def b_operand(xs, BM):
    t = xs.reshape(BM, 4, 4, 4, 2, 2).permute(1, 3, 4, 2, 5, 0)
    return t.reshape(256, BM)


def main():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    rows, K, BN, BM = 128, 4096, 64, 8
    packed, sf, gs = random_weights(rows, K, generator=gen, device="cpu")
    sf = sf.clone()
    sf[3, 5] = 0  # a zero-scale block
    sf[7, 9] = torch.tensor(2.0 ** -8, dtype=torch.float32).to(torch.float8_e4m3fn)  # an E4M3 subnormal
    words, scales = sub._prepare_matrix(packed, sf)
    words = words.view(torch.int32).to(torch.int64)
    scales = scales.view(torch.int16)
    ref_w = dequantize(packed, sf, gs)  # [rows, K]
    x = torch.randn(BM, K).to(torch.bfloat16).float()
    ref = ref_w @ x.T  # [rows, BM]

    i = torch.arange(BN // 2)[:, None, None, None]
    ch = torch.arange(4)[None, :, None, None]
    c = torch.arange(4)[None, None, :, None]
    v = torch.arange(4)[None, None, None, :]
    si, sch, sc = i[..., 0], ch[..., 0], c[..., 0]
    n = torch.arange(BM)[:, None, None, None]
    xj = torch.arange(16)[None, None, None, :]
    worst = 0.0
    for nt in range(rows // BN):
        rt0 = nt * (BN // 16)
        w_off = (rt0 + i // 8) * 8192 + (i % 8) * 16 + ch * 128 + c * 4 + v
        s_off = (rt0 + si // 8) * 2048 + (si % 8) * 16 + sc * 4 + sch
        x_off = n * 4096 + ch * 64 + c * 16 + xj
        acc = torch.zeros(BN, BM, dtype=torch.float64)
        for kc in range(16):
            A = a_operand(words[w_off + kc * 512], scales[s_off + kc * 128], BN)
            B = b_operand(x.reshape(-1)[x_off + kc * 256], BM)
            acc += A.double() @ B.double()
        got = acc * float(gs) * 256
        want = ref[nt * BN:(nt + 1) * BN].double()
        err = ((got - want).norm() / want.norm()).item()
        worst = max(worst, err)
    print(f"relative error {worst:.3e}")
    assert worst < 1e-5, worst


if __name__ == "__main__":
    main()
