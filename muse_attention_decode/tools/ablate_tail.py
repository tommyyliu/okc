

# ---------------------------------------------------------------- verify ablation (not submitted)
import traceback as _tb
import triton
import triton.language as tl


def _p(*a):
    print(*a, flush=True)


@triton.jit
def _vab(Q, KC, VC, TABLE, OUT, NPER: tl.constexpr, BN: tl.constexpr, MODE: tl.constexpr,
         HALF: tl.constexpr, QK8: tl.constexpr):
    pid = tl.program_id(0)
    h = pid % 2
    r = tl.arange(0, 64)
    d = tl.arange(0, 64)
    kn = tl.arange(0, BN)
    q = tl.load(Q + r[:, None] * 64 + d[None, :])
    if QK8:
        q = q.to(tl.float8e4nv)
    else:
        q = q.to(tl.float16)
    m = tl.full((64,), -float("inf"), tl.float32)
    l = tl.zeros((64,), tl.float32)
    acc = tl.zeros((64, 64), tl.float32)
    base = (pid // 2) * NPER
    for st in range(base, base + NPER, BN):
        pg = tl.load(TABLE + st + kn)
        if HALF:
            off = pg.to(tl.int64)[:, None] * 128 + h * 64 + d[None, :]
        else:
            off = pg.to(tl.int64)[:, None] * 128 + d[None, :]
        k = tl.load(KC + off)
        v = tl.load(VC + off)
        if MODE == 0:
            acc += (k.to(tl.float32) + v.to(tl.float32)).sum(0)[None, :] * 0 + 1
            l += 1
        else:
            if QK8:
                qk = tl.dot(q, tl.trans(k))
            else:
                qk = tl.dot(q, tl.trans(k.to(tl.float16)))
            if MODE == 1:
                l += tl.sum(qk, 1)
                acc += v.to(tl.float32).sum(0)[None, :]
            elif MODE == 2:
                m_new = tl.maximum(m, tl.max(qk, 1))
                p = tl.exp2(qk - m_new[:, None])
                alpha = tl.exp2(m - m_new)
                l = l * alpha + tl.sum(p, 1)
                acc = acc * alpha[:, None] + v.to(tl.float32).sum(0)[None, :]
                m = m_new
            elif MODE == 3:
                m_new = tl.maximum(m, tl.max(qk, 1))
                p = tl.exp2(qk - m_new[:, None])
                alpha = tl.exp2(m - m_new)
                l = l * alpha + tl.sum(p, 1)
                acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v.to(tl.float16))
                m = m_new
            else:  # no max tracking: static offset
                p = tl.exp2(qk - 40.0)
                l = l + tl.sum(p, 1)
                acc = acc + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
    tl.store(OUT + pid * 64 + r, l + tl.sum(acc, 1) + m)


def _abl():
    import torch
    rows = 32 * 86016
    kc = torch.randn((rows, 128), device="cuda").to(torch.float8_e4m3fn)
    vc = torch.randn((rows, 128), device="cuda").to(torch.float8_e4m3fn)
    tab = torch.randperm(rows, device="cuda").to(torch.int32)
    q = torch.randn((64, 64), device="cuda") * 0.3
    outb = torch.empty(1 << 20, device="cuda")
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")

    def tm(f, reps=10):
        f()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            f()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(reps):
            flush.zero_()
            e0.record()
            g.replay()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1000)
        ts.sort()
        return ts[len(ts) // 2]

    for mode in (0, 1, 2, 3, 4):
        for half in (1, 0):
            if half == 0 and mode not in (0, 3):
                continue
            for q8 in ((1,) if mode else (0,)):
                for ns, bn, w, st in ((8, 64, 4, 2), (8, 64, 4, 3), (16, 64, 4, 3), (8, 32, 4, 3), (8, 128, 8, 3), (16, 32, 4, 2)):
                    ctas = 32 * ns * 2  # two heads
                    nper = rows // (32 * ns)
                    if nper % bn:
                        continue
                    f = lambda: _vab[(ctas,)](q, kc, vc, tab, outb, nper, bn, mode, half, q8, num_warps=w, num_stages=st)
                    try:
                        t = tm(f)
                        k = _vab[(ctas,)](q, kc, vc, tab, outb, nper, bn, mode, half, q8, num_warps=w, num_stages=st)
                        _p(f"mode {mode} half {half} q8 {q8} ns {ns} bn {bn} w {w} st {st}: {t:.1f}us regs {k.n_regs}")
                    except Exception:
                        _p("fail", mode, half, ns, bn, w, st, _tb.format_exc()[-600:])


try:
    _abl()
except Exception:
    _p("ABL FAIL", _tb.format_exc())
