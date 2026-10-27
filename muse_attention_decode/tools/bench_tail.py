

# ---------------------------------------------------------------- dev bench (not submitted)
import math as _math
import sys as _sys
import time as _timemod
import traceback as _tb
import triton
import triton.language as tl


def _p(*a):
    print(*a, flush=True)


def _gen(batch, ctx, layer, q_len=1, spread="fixed", page=1, heads="tp8", kv_scale=1.0, seed=0, fill=True):
    dev = "cuda"
    hq, hkv = {"tp8": (16, 2), "dp": (128, 16)}[heads]
    g = torch.Generator(device=dev).manual_seed(seed)
    if spread == "agentic":
        u = torch.rand((batch,), generator=g, device=dev)
        seq = (ctx * (1 + 0.075 * (2 * u - 1))).long().clamp(1)
        longest = _math.ceil(ctx * 1.075)
    elif spread == "mixed":
        z = torch.randn((batch,), generator=g, device=dev)
        seq = (ctx * torch.exp(z - 0.5)).clamp(16, 4 * ctx).long()
        longest = 4 * ctx
    else:
        seq = torch.full((batch,), ctx, dtype=torch.int64, device=dev)
        longest = ctx
    pages_per = (seq + q_len + page - 1) // page
    max_pages = (longest + q_len + page - 1) // page
    total = batch * max_pages
    perm = torch.randperm(total, generator=g, device=dev).to(torch.int32)
    table = torch.full((batch, max_pages), -1, dtype=torch.int32, device=dev)
    offs = torch.cumsum(pages_per, 0) - pages_per
    for b in range(batch):
        n, o = int(pages_per[b]), int(offs[b])
        table[b, :n] = perm[o:o + n]
    shape = (total, page, hkv, 64)
    kc = torch.empty(shape, dtype=torch.float8_e4m3fn, device=dev)
    vc = torch.empty(shape, dtype=torch.float8_e4m3fn, device=dev)
    for c in (kc, vc):
        flat = c.view(-1, hkv * 64)
        for at in range(0, flat.shape[0], 1 << 20):
            n = min(1 << 20, flat.shape[0] - at)
            flat[at:at + n] = (torch.randn((n, hkv * 64), generator=g, device=dev) / kv_scale).to(torch.float8_e4m3fn)
    w = (2 * hq + 2 * hkv) * 64
    qkv = torch.randn((batch, q_len, w), generator=g, device=dev).to(torch.bfloat16)
    qkv[..., :hq * 64] *= 2
    positions = (seq[:, None] + torch.arange(q_len, device=dev)).reshape(-1)
    pos = positions.view(batch, q_len)
    loc = (torch.gather(table.long(), 1, pos // page) * page + pos % page).reshape(-1)
    cache_seqlens = (seq + q_len).to(torch.int32)
    cu = torch.arange(0, batch * q_len + 1, q_len, dtype=torch.int32, device=dev)
    sched = None
    if layer == "full" and q_len == 1:
        try:
            from sgl_kernel.flash_attn import get_scheduler_metadata
            sched = get_scheduler_metadata(
                batch_size=batch, max_seqlen_q=1, max_seqlen_k=int(cache_seqlens.max()),
                num_heads=hq, num_heads_k=hkv, headdim=64, cache_seqlens=cache_seqlens,
                qkv_dtype=torch.float8_e4m3fn, cu_seqlens_q=cu, page_size=page, causal=True, num_splits=0)
        except Exception as e:  # noqa
            _p("sched fail", e)
    return {
        "qkv": qkv, "positions": positions, "cache_loc": loc, "cache_seqlens": cache_seqlens,
        "cu_seqlens_q": cu, "scheduler_metadata": sched, "k_cache": kc, "v_cache": vc,
        "k_scale": torch.tensor(kv_scale, dtype=torch.float32, device=dev),
        "v_scale": torch.tensor(kv_scale, dtype=torch.float32, device=dev),
        "page_table": table, "seq_lens": seq.to(torch.int32),
        "config": {"batch": batch, "q_len": q_len, "num_q_heads": hq, "num_kv_heads": hkv, "head_dim": 64,
                   "page_size": page, "window": 2048 if layer == "sliding" else 0,
                   "rope_theta": 500000.0 if layer == "sliding" else 0.0,
                   "qk_scale_factor": 5.4730064863875, "sm_scale": 0.125, "kv_scale": kv_scale},
    }


def _ref(data):
    """fp32 reference (cache already holds the new entries)."""
    cfg = data["config"]
    B, L, hq, hkv, page = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"], cfg["page_size"]
    q_in, k_in, v_in, gate = data["qkv"].split([hq * 64, hkv * 64, hkv * 64, hq * 64], -1)
    q = q_in.reshape(B, L, hq, 64).float()
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-5) * cfg["qk_scale_factor"]
    pos = data["seq_lens"].long()[:, None] + torch.arange(L, device=q.device)
    if cfg["rope_theta"]:
        inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, 64, 2, device=q.device).float() / 64))
        ang = pos.float()[..., None] * inv
        c, s_ = ang.cos().unsqueeze(-2), ang.sin().unsqueeze(-2)
        e, o = q[..., 0::2], q[..., 1::2]
        q = torch.stack([e * c - o * s_, e * s_ + o * c], -1).flatten(-2)
    kf = data["k_cache"].view(-1, hkv, 64)
    vf = data["v_cache"].view(-1, hkv, 64)
    out = torch.empty((B, L, hq, 64), device=q.device)
    for b in range(B):
        S = int(data["seq_lens"][b]) + L
        t = torch.arange(S, device=q.device)
        rows = data["page_table"][b].long()[t // page] * page + t % page
        K = kf[rows].float() * data["k_scale"]
        V = vf[rows].float() * data["v_scale"]
        for j in range(L):
            p = int(pos[b, j])
            st = max(0, p - cfg["window"] + 1) if cfg["window"] else 0
            att = torch.einsum("kgd,tkd->kgt", q[b, j].view(hkv, hq // hkv, 64), K[st:p + 1]) * cfg["sm_scale"]
            out[b, j] = torch.einsum("kgt,tkd->kgd", att.softmax(-1), V[st:p + 1]).reshape(hq, 64)
    out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + 1e-5) * torch.sigmoid(gate.float()).view(B, L, hq, 64)
    return out.reshape(B, L, hq * 64)


def _sgl():
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    from sglang.kernels.ops.kvcache.kvcache import store_cache
    from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
    from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate
    norms, ropes = {}, {}
    FP8 = torch.float8_e4m3fn

    def fn(data):
        cfg = data["config"]
        batch, length = cfg["batch"], cfg["q_len"]
        hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], 64
        tokens = batch * length
        qkv = data["qkv"].view(tokens, -1)
        key = (hq, hkv)
        if key not in norms:
            norms[key] = FusedQKNorm(head_dim=d, num_q_heads=hq, num_kv_heads=hkv, eps=1e-5)
        if cfg["rope_theta"]:
            th = cfg["rope_theta"]
            if th not in ropes:
                inv_freq = 1.0 / (th ** (torch.arange(0, d, 2, dtype=torch.float, device="cuda") / d))
                fr = torch.einsum("i,j -> ij", torch.arange(1 << 21, dtype=torch.float, device="cuda"), inv_freq)
                ropes[th] = torch.cat((fr.cos(), fr.sin()), dim=-1)
            q, k = norms[key](qkv, data["positions"], ropes[th])
        else:
            q, k = norms[key](qkv)
        v = qkv[:, (hq + hkv) * d:(hq + 2 * hkv) * d]
        k = k.reshape(tokens, hkv * d)
        store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                    data["k_cache"].view(-1, hkv * d).view(torch.uint8),
                    data["v_cache"].view(-1, hkv * d).view(torch.uint8), data["cache_loc"], reserved_skip_index=-1)
        window = cfg["window"]
        out = flash_attn_with_kvcache(
            q=q.view(tokens, hq, d).to(FP8), k_cache=data["k_cache"], v_cache=data["v_cache"],
            page_table=data["page_table"], cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"],
            max_seqlen_q=length, softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
            window_size=(window - 1, 0) if window else (-1, -1), scheduler_metadata=data["scheduler_metadata"])
        if isinstance(out, tuple):
            out = out[0]
        gate = qkv[:, (hq + 2 * hkv) * d:]
        o = rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None, head_dim=d, eps=1e-5,
                          has_norm=True)
        return o.view(batch, length, hq * d)
    return fn


_FLUSH = None


def _time(fn, data, reps=25):
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    fn(data)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(data)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn(data)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for i in range(reps):
        _FLUSH.zero_()
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


SCORED = [  # (name, batch, layer, q_len, spread)
    ("d1s", 1, "sliding", 1, "fixed"), ("d1f", 1, "full", 1, "fixed"),
    ("d16s", 16, "sliding", 1, "agentic"), ("d16f", 16, "full", 1, "agentic"),
    ("d32s", 32, "sliding", 1, "agentic"), ("d32f", 32, "full", 1, "agentic"),
    ("v1s", 1, "sliding", 8, "fixed"), ("v1f", 1, "full", 8, "fixed"),
    ("v16s", 16, "sliding", 8, "agentic"), ("v16f", 16, "full", 8, "agentic"),
    ("v32s", 32, "sliding", 8, "agentic"), ("v32f", 32, "full", 8, "agentic"),
]


def _set(t):
    TUNE.clear()
    TUNE.update(t)


def _main():
    import triton as _tr
    _p("torch", torch.__version__, "triton", _tr.__version__, torch.cuda.get_device_name())
    t0 = _time_mod()
    # correctness
    for args in [dict(batch=1, ctx=10, layer="sliding"), dict(batch=3, ctx=2047, layer="sliding"),
                 dict(batch=5, ctx=5000, layer="sliding", q_len=8, spread="mixed"),
                 dict(batch=4, ctx=3000, layer="full", page=64),
                 dict(batch=2, ctx=9000, layer="full", kv_scale=1 / 64, page=64),
                 dict(batch=3, ctx=4000, layer="sliding", q_len=8, kv_scale=1 / 64),
                 dict(batch=2, ctx=20000, layer="full", q_len=8, heads="dp", page=64),
                 dict(batch=4, ctx=80000, layer="full", spread="agentic")]:
        try:
            _set({})
            data = _gen(**args)
            out = kernel_fn(data)
            ref = _ref(data)
            _p("corr", args, "rel", round(_rel(out, ref), 5))
        except Exception:
            _p("corr FAIL", args, _tb.format_exc()[-3000:])
    _p("corr time", _time_mod() - t0)
    for args in (("d32f", 32, "full", 1, "agentic", dict(pack=2, nsplit=8, bn=128, warps=4, stages=5, gearly=True, qk8=True)),
                 ("d32s", 32, "sliding", 1, "agentic", dict(pack=1, nsplit=8, bn=64, warps=4, stages=5, gearly=True)),
                 ("v32f", 32, "full", 8, "agentic", dict(pack=1, nsplit=8, bn=64, warps=4, stages=2, pref=True, qk8=True, sb=2, maxnreg=120)),
                 ("v32f", 32, "full", 8, "agentic", dict(pack=1, nsplit=8, bn=64, warps=4, stages=3, qk8=True, sb=2, maxnreg=120))):
        try:
            _prof(*args)
            _ptx_stats(args[0])
        except Exception:
            _p("prof FAIL", args[0], _tb.format_exc()[-2000:])
    base = _sgl()
    total_b = {}
    for name, batch, layer, q_len, spread in SCORED:
        if not SWEEP(name, batch, layer, q_len):
            continue
        data = _gen(batch, 80000, layer, q_len, spread)
        _set({})
        try:
            tb = _time(base, data)
        except Exception:
            _p("base FAIL", name, _tb.format_exc()[-2000:])
            tb = float("nan")
        res = []
        for t in SWEEP(name, batch, layer, q_len):
            t = {k: v for k, v in t.items() if v is not None}
            _set(t)
            try:
                tt = _time(kernel_fn, data)
                res.append((tt, dict(t, regs=LAST.n_regs, sp=LAST.n_spills)))
            except Exception:
                _p("FAIL", name, t, _tb.format_exc()[-1500:])
        _set({})
        td = _time(kernel_fn, data)
        res.sort(key=lambda x: x[0])
        _p(f"{name}: sglang {tb:.1f}  default {td:.1f}  best " +
           "  ".join(f"{x:.1f}{t}" for x, t in res[:6]))
        del data
        torch.cuda.empty_cache()
    _p("total time", _time_mod() - t0)


def _time_mod():
    return _timemod.time()


def SWEEP(name, batch, layer, q_len):
    out = []
    if batch != 1:
        return out
    if q_len == 1 and layer == "sliding":
        base = dict(pack=1, nsplit=16, bn=128, warps=8, stages=2, pref=True, gearly=True)
        for mst in (1, 2, 3):
            out.append(dict(base, mst=mst))
        out.append(dict(base, gs=4, sb=4))
        out.append(dict(base, nsplit=32, bn=64, gs=8, sb=8))
    elif q_len == 1:
        base = dict(pack=2, nsplit=256, bn=64, warps=4, stages=6, gearly=True, qk8=True, sb=4, gs=16)
        for mst in (1, 3):
            out.append(dict(base, mst=mst))
        out.append(dict(base, nsplit=128, bn=128, stages=5, mst=3))
        out.append(dict(base, nsplit=128, bn=256, warps=8, stages=3, mst=3))
        out.append(dict(base, nsplit=128, bn=128, warps=8, stages=4, mst=3))
        out.append(dict(base, nsplit=64, bn=256, warps=8, stages=4, mst=3))
    elif layer == "sliding":
        base = dict(pack=1, nsplit=32, bn=64, warps=8, stages=3, qk8=True, sb=4)
        for mst in (1, 2, 3):
            out.append(dict(base, mst=mst))
        out.append(dict(base, gs=4, mst=1))
        out.append(dict(base, gs=8, sb=2, mst=1))
        out.append(dict(base, nsplit=16, gs=4, mst=1))
        out.append(dict(base, nsplit=16, mst=3))
    else:
        base = dict(pack=1, nsplit=128, bn=64, warps=4, stages=2, pref=True, qk8=True, sb=4)
        for mst in (1, 3):
            out.append(dict(base, mst=mst))
        out.append(dict(base, gs=8, mst=1))
        out.append(dict(base, gs=12, mst=1))
        out.append(dict(base, gs=16, sb=4, mst=3))
        out.append(dict(base, nsplit=64, gs=8, mst=1))
    return out


def _ptx_stats(tag):
    k = LAST
    ptx = k.asm["ptx"]
    ttg = k.asm.get("ttgir", "")
    _p(tag, "regs", k.n_regs, "spills", k.n_spills, "cp.async", ptx.count("cp.async.cg") + ptx.count("cp.async.ca"),
       "ld.global", ptx.count("ld.global"), "wgmma", ptx.count("wgmma.mma_async"), "mma.sync", ptx.count("mma.sync"),
       "async_copy", ttg.count("async_copy_global_to_local"), "local_load", ttg.count("local_load"),
       "bar.sync", ptx.count("bar.sync"))


def _prof(name, batch, layer, q_len, spread, t):
    data = _gen(batch, 80000, layer, q_len, spread)
    t = dict(t)
    nprog = batch * 2 // t["pack"] * (t["nsplit"] + 1 if t["nsplit"] > 1 else 1)
    ts = torch.zeros(nprog * 8, dtype=torch.int64, device="cuda")
    t["ts"] = ts
    _set(t)
    kernel_fn(data)
    torch.cuda.synchronize()
    for _ in range(3):
        _FLUSH_INIT()
        _FLUSH.zero_()
        ts.zero_()
        kernel_fn(data)
        torch.cuda.synchronize()
    x = ts.view(nprog, 8).cpu().double()
    t0 = x[:, 0].min()
    rel = (x - t0) / 1000.0
    msg = []
    _p("  per-phase durations (median): " + " ".join(f"{float((x[:, k + 1] - x[:, k]).median()) / 1000:.2f}" for k in range(3)))
    for k in range(7):
        col = rel[:, k]
        col = col[x[:, k] > 0]
        if len(col):
            msg.append(f"p{k}: min {col.min():.2f} med {col.median():.2f} max {col.max():.2f}")
    _p(f"PROF {name} {t.get('nsplit')} n={nprog}: " + " | ".join(msg))
    _set({})


def _FLUSH_INIT():
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")


@triton.jit
def _gather_bw(KC, VC, TABLE, OUT, NPER: tl.constexpr, BN: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    c = tl.arange(0, RB)
    acc = tl.zeros((BN, RB), tl.float32)
    for st in range(pid * NPER, pid * NPER + NPER, BN):
        pg = tl.load(TABLE + st + tl.arange(0, BN))
        off = pg.to(tl.int64)[:, None] * 128 + c[None, :]
        k = tl.load(KC + off)
        v = tl.load(VC + off)
        acc += k.to(tl.float32) + v.to(tl.float32)
    tl.store(OUT + pid, tl.sum(tl.sum(acc, 1), 0))


def _bw():
    rows = 32 * 86016
    kc = torch.randn((rows, 128), device="cuda").to(torch.float8_e4m3fn)
    vc = torch.randn((rows, 128), device="cuda").to(torch.float8_e4m3fn)
    tab = torch.randperm(rows, device="cuda").to(torch.int32)
    outb = torch.empty(1 << 16, device="cuda")
    res = []
    for ctas in (336, 672, 1344, 2688):
        nper = rows // ctas
        for bn, w, st in ((64, 4, 3), (64, 4, 4), (128, 4, 3), (128, 4, 4), (128, 8, 4), (256, 8, 3), (64, 4, 6)):
            if nper % bn:
                continue
            f = lambda _d: _gather_bw[(ctas,)](kc, vc, tab, outb, nper, bn, 128, num_warps=w, num_stages=st)
            try:
                t = _time(f, None, reps=20)
                res.append((t, ctas, bn, w, st))
            except Exception:
                _p("bw fail", ctas, bn, w, st, _tb.format_exc()[-800:])
    res.sort()
    for t, *cfg in res[:10]:
        _p(f"gather {t:.1f}us {rows * 256 / t / 1e6:.2f} TB/s cfg(ctas,bn,w,st)={cfg}")
    src = torch.empty(rows * 256, dtype=torch.uint8, device="cuda")
    t = _time(lambda _d: src.view(torch.int32).sum(), None, reps=10)
    _p(f"seq read sum {t:.1f}us {rows * 256 / t / 1e6:.2f} TB/s")
    del kc, vc, src


try:
    _main()
except Exception:
    _p("BENCH FAIL", _tb.format_exc())
