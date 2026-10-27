"""Compile submission.py's kernels for sm_90a without torch or a GPU; print static counts.

  ../muse_bf16_proj/build/compiler-env/bin/python tools/audit.py [submission.py] [out_dir]
"""
import os, re, sys, types
from collections import Counter
from pathlib import Path

src = Path(sys.argv[1] if len(sys.argv) > 1 else "submission.py")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "build/audit")
out.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TRITON_CACHE_DIR", str((out / "cache").resolve()))

class _Any:
    def __getattr__(self, name): return _Any()
    def __call__(self, *a, **k): return _Any()
stub = types.ModuleType("torch"); stub.__getattr__ = lambda name: _Any()
sys.modules["torch"] = stub

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.experimental.gluon._runtime import GluonASTSource
import importlib.util
spec = importlib.util.spec_from_file_location("sub", src)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def compile_one(name, kernel, sig, consts, warps=4, stages=3, gluon=False):
    signature = {**sig, **{k: "constexpr" for k in consts}}
    attrs = {(i,): [["tt.divisibility", 16]] for i, a in enumerate(kernel.arg_names) if sig.get(a, "").startswith("*")}
    S = GluonASTSource if gluon else ASTSource
    c = triton.compile(S(fn=kernel, signature=signature, constexprs=consts, attrs=attrs),
                       target=GPUTarget("cuda", 90, 32), options={"num_warps": warps, "num_stages": stages})
    d = out / name; d.mkdir(exist_ok=True)
    for ext in ("ttgir", "ptx"):
        (d / f"k.{ext}").write_text(c.asm[ext])
    (d / "k.cubin").write_bytes(c.asm["cubin"])
    ptx = re.sub(r"//[^\n]*", "", c.asm["ptx"])
    ops = Counter(m[1] for m in re.finditer(r"^\s*(?:@!?%\w+\s+)?([a-z][\w.:]*)", ptx, re.M))
    keys = [k for k in ops if k.startswith(("wgmma", "mma", "ld.shared", "st.shared", "ldmatrix", "cp.async", "ld.global", "ld.local", "st.local", "st.global", "stmatrix", "bar."))]
    print(name, "smem", c.metadata.shared, {k: ops[k] for k in sorted(keys)},
          "cvt_layout", c.asm["ttgir"].count("convert_layout"), "local_alloc", c.asm["ttgir"].count("local_alloc"), flush=True)
    return c

P = "*i32"
which = sys.argv[3] if len(sys.argv) > 3 else "all"
for gated in (True, False):
    for bm, rep in [(8, 1), (16, 1), (32, 1)]:
        lw, lx, mma = mod._layouts(bm, rep)
        compile_one(f"gemm_{'gu' if gated else 'dn'}_{bm}_{rep}", mod._expert_gemm,
                    dict(X="*bf16", W="*i32", S="*i16", GS="*fp32", TILES=P, LIST=P, COUNT=P, OUT="*bf16", M="i32"),
                    dict(LOCAL_START=0, NW=mod.NW, GATED=gated, BM=bm, REP=rep, LW=lw, LX=lx, MMA=mma), mod.NW, 1, gluon=True)
compile_one("route", mod._route, dict(IDS=P, COUNT=P, LIST=P, TILES=P, M="i32"), dict(LOCAL_START=0, P=2048, BM=32, MT=8))
compile_one("finalize", mod._finalize, dict(Y="*bf16", ROUTE_W="*fp32", IDS=P, NORM_W="*fp32", OUT="*bf16"),
            dict(LOCAL_START=0, LOCAL_COUNT=32), 8)
