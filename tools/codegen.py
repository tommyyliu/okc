"""Compile projection probes for Hopper without importing Torch or using a GPU.

  python -m tools.codegen --plan
  python -m tools.codegen --preset tma --output build/codegen-tma

Requires an installed Triton compiler with its NVIDIA assembler. Extracts only
JIT definitions from local sources; prepare and kernel_fn are never executed.
Reports static PTX instruction counts, IR layout conversions, shared memory,
and ptxas resource usage when available. Static counts are not runtime costs.
"""

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback


from projection.configs import CONFIGS, TUNED_CONFIGS, fallback_config
from projection.shapes import SCORED_TOKENS, SHAPES, shape_config

HERE = Path(__file__).resolve().parents[1]
KERNEL_SOURCES = sorted((HERE / "projection/kernels").glob("*.py"))


def compile_jobs(preset="pdl", layers=tuple(SHAPES), tokens=SCORED_TOKENS):
    """Compile the same configurations selected by the preset, including reducers."""
    jobs = []
    for layer in layers:
        for m in tokens:
            config = shape_config(layer, m)
            n, k = config["n"], config["k"]
            key = (layer, m)
            option = TUNED_CONFIGS.get(key, CONFIGS[preset].get(key, fallback_config(config, preset)))
            family = option[0]
            if family == "torch":
                continue
            dtype = "fp32" if layer == "router" else "bf16"
            job = dict(name=f"{preset}-{layer}-{m}", source=preset)
            if family == "gemv":
                _, rows, warps = option
                job.update(kernel="_gemv_full", signature={"X": "*bf16", "W": f"*{dtype}", "Y": f"*{dtype}"},
                           constants={"N": n, "K": k, "ROWS": rows, "BLOCK_K": 1 << (k-1).bit_length()},
                           num_warps=warps, num_stages=3)
                jobs.append(job)
                continue
            _, bm, bn, bk, splits, warps, stages = option
            constants = dict(M=m, N=n, K=k, BM=bm, BN=bn, BK=bk)
            signature = dict(X="*bf16", W="*bf16", Y=f"*{dtype}")
            kernel = {"dot": "_gemm", "gemm": "_stream_gemm", "orientation": "_weight_major_gemm",
                      "tma": "_tma_weight_major"}[family]
            if family != "orientation":
                constants["S"] = splits
                signature["P"] = "*fp32"
            if family == "dot":
                signature.pop("W")
                signature.update(WH="*bf16", WL="*bf16")
                constants["ROUTER"] = layer == "router"
            elif family == "tma":
                signature.pop("X")
                signature.pop("W")
                signature.update(WD=f"tensordesc<bf16[{bn},{bk}]>", XD=f"tensordesc<bf16[{bm},{bk}]>")
            job.update(kernel=kernel, signature=signature, constants=constants,
                       num_warps=warps, num_stages=stages)
            jobs.append(job)
            if splits > 1:
                jobs.append(dict(name=f"{preset}-{layer}-{m}-reduce", source=preset, kernel="_reduce",
                                 signature={"P": "*fp32", "Y": f"*{dtype}"},
                                 constants={"TOTAL": m*n, "S": splits, "B": 256},
                                 num_warps=4, num_stages=3, launch_pdl=True))
    return jobs


def extract_jit_sources(paths):
    definitions = {}
    for path in paths:
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.FunctionDef) and any(
                    isinstance(d, ast.Attribute) and d.attr == "jit" for d in node.decorator_list):
                if node.name in definitions:
                    raise ValueError(f"duplicate JIT definition: {node.name}")
                definitions[node.name] = node
    body = "import triton\nimport triton.language as tl\n\n"
    return body + "\n\n".join(ast.unparse(n) for n in definitions.values()) + "\n"


def summarize_ptx(ptx, ttgir=""):
    counts = Counter()
    # Strip comments so an opcode mentioned in a comment cannot affect counts.
    clean = re.sub(r"/\*.*?\*/", "", ptx, flags=re.S)
    clean = re.sub(r"//[^\n]*", "", clean)
    for line in clean.splitlines():
        match = re.match(r"\s*(?:@!?%[\w$]+\s+)?([a-z][\w.:]*)(?=\s|;)", line)
        if match:
            counts[match[1]] += 1
    global_loads = {k: v for k, v in counts.items() if k.startswith(("ld.global", "ldu.global"))}
    local_ops = {k: v for k, v in counts.items() if k.startswith(("ld.local", "st.local"))}
    return {"global_load_instructions": global_loads, "local_memory_instructions": local_ops,
            "shared_memory_instructions": {k: v for k, v in counts.items() if k.startswith(("ld.shared", "st.shared"))},
            "mma_instructions": {k: v for k, v in counts.items() if k.startswith(("mma.", "wgmma."))},
            "cp_async_instructions": {k: v for k, v in counts.items() if k.startswith("cp.async")},
            "shuffle_instructions": {k: v for k, v in counts.items() if k.startswith("shfl.")},
            "grid_dependency_instructions": {k: v for k, v in counts.items() if k.startswith("griddepcontrol.")},
            "layout_conversions_in_ir": len(re.findall(r"(?:triton_gpu|ttg)\.convert_layout", ttgir)),
            "note": "Static assembly/IR counts, including loop bodies once; not executed instruction counts."}


def parse_ptxas(text):
    fields = {"registers_per_thread": r"Used\s+(\d+)\s+registers",
              "stack_frame_bytes": r"(\d+)\s+bytes stack frame",
              "spill_store_bytes": r"(\d+)\s+bytes spill stores",
              "spill_load_bytes": r"(\d+)\s+bytes spill loads"}
    return {key: int(match[1]) if (match := re.search(pattern, text)) else None
            for key, pattern in fields.items()}


def parse_cubin_resources(text):
    fields = {"registers_per_thread": "REG", "stack_bytes": "STACK",
              "shared_bytes": "SHARED", "local_bytes": "LOCAL"}
    return {name: int(match[1]) if (match := re.search(rf"\b{token}:\s*(\d+)", text)) else None
            for name, token in fields.items()}


def find_ptxas(triton, explicit):
    candidates = [explicit, shutil.which("ptxas"),
                  str(Path(triton.__file__).parent / "backends/nvidia/bin/ptxas")]
    return next((str(Path(p).resolve()) for p in candidates if p and Path(p).is_file()), None)


def audit(args, jobs):
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "target": "cuda:90:32 (sm_90a)",
              "status": "running", "planned_jobs": jobs, "jobs": [], "timing_claims": False}
    report_path = args.output / "report.json"
    try:
        os.environ.setdefault("TRITON_CACHE_DIR", str((args.output / "cache").resolve()))
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource
    except ImportError as error:
        report.update(status="blocked", error=str(error),
                      remedy="Install Triton with its NVIDIA backend on a CPU or GPU machine, then rerun. No GPU is required.")
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Compiler unavailable: {error}\nReport: {report_path}", file=sys.stderr)
        return 2
    report["triton_version"] = triton.__version__
    ptxas = find_ptxas(triton, args.ptxas)
    cuobjdump = args.cuobjdump or shutil.which("cuobjdump")
    nvdisasm = shutil.which("nvdisasm")
    if ptxas:
        report["ptxas_version"] = subprocess.run([ptxas, "--version"], capture_output=True, text=True).stdout
    bundles = {job["source"]: KERNEL_SOURCES for job in jobs}
    modules = {}
    for name, paths in bundles.items():
        if not any(job["source"] == name for job in jobs):
            continue
        path = args.output / f"extracted_{name}.py"
        path.write_text(extract_jit_sources(paths))
        spec = importlib.util.spec_from_file_location(f"audit_{name}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    for job in jobs:
        entry = dict(job)
        directory = args.output / job["name"]
        directory.mkdir(exist_ok=True)
        entry["source_sha256"] = {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in bundles[job["source"]]}
        try:
            kernel = getattr(modules[job["source"]], job["kernel"])
            signature = {**job["signature"], **{k: "constexpr" for k in job["constants"]}}
            attrs = {(i,): [["tt.divisibility", 16]] for i, name in enumerate(kernel.arg_names)
                     if job["signature"].get(name, "").startswith("*")}
            src = ASTSource(fn=kernel, signature=signature, constexprs=job["constants"], attrs=attrs)
            options = {"num_warps": job["num_warps"], "num_stages": job["num_stages"]}
            if "launch_pdl" in job:
                options["launch_pdl"] = job["launch_pdl"]
            compiled = triton.compile(src, target=GPUTarget("cuda", 90, 32), options=options)
            for extension in ("ttir", "ttgir", "llir", "ptx", "cubin"):
                artifact = compiled.asm.get(extension)
                if artifact is not None:
                    path = directory / f"kernel.{extension}"
                    path.write_bytes(artifact) if isinstance(artifact, bytes) else path.write_text(artifact)
            metadata = compiled.metadata._asdict()
            entry["metadata"] = json.loads(json.dumps(metadata, default=str))
            entry["static_summary"] = summarize_ptx(compiled.asm["ptx"], compiled.asm.get("ttgir", ""))
            if job["kernel"] == "_tma_weight_major":
                summary = entry["static_summary"]
                entry["uses_tma_and_wgmma"] = (
                    any(k.startswith("cp.async.bulk.tensor") for k in summary["cp_async_instructions"])
                    and any(k.startswith("wgmma.mma_async") for k in summary["mma_instructions"]))
                if not entry["uses_tma_and_wgmma"]:
                    raise RuntimeError("TMA/WGMMA instructions missing from generated kernel")
            if job["kernel"] == "_reduce" or job["constants"].get("S", 1) > 1:
                ptx = compiled.asm["ptx"]
                if job["kernel"] == "_reduce":
                    wait = ptx.index("griddepcontrol.wait;")
                    first_load = re.search(r"\bld\.global[.\s]", ptx).start()
                    entry["dependency_order"] = {"wait_before_global_load": wait < first_load,
                                                 "consumer_launch_pdl": metadata["launch_pdl"]}
                else:
                    release = ptx.index("griddepcontrol.launch_dependents;")
                    first_store = re.search(r"\bst\.global[.\s]", ptx).start()
                    mma_positions = [m.start() for m in re.finditer(r"\b(?:wgmma|mma)\.", ptx)]
                    entry["dependency_order"] = {"release_before_partial_store": release < first_store,
                                                 "release_after_last_mma": release > max(mma_positions),
                                                 "producer_uses_normal_launch": not metadata["launch_pdl"]}
                if not all(entry["dependency_order"].values()):
                    raise RuntimeError(f"PDL instruction/launch ordering failed: {entry['dependency_order']}")
            if cuobjdump:
                for flag, filename in (("--dump-sass", "kernel.sass"),
                                       ("--dump-resource-usage", "cubin-resources.log")):
                    dumped = subprocess.run([cuobjdump, flag, str(directory / "kernel.cubin")],
                                            capture_output=True, text=True)
                    (directory / filename).write_text(dumped.stdout + dumped.stderr)
                    if dumped.returncode:
                        raise RuntimeError(f"cuobjdump failed; see {directory / filename}")
                    if flag == "--dump-resource-usage":
                        entry["original_cubin_resources"] = parse_cubin_resources(dumped.stdout)
            elif nvdisasm:
                dumped = subprocess.run([nvdisasm, str(directory / "kernel.cubin")], capture_output=True, text=True)
                (directory / "kernel.sass").write_text(dumped.stdout + dumped.stderr)
                entry["disassembly_returncode"] = dumped.returncode
            if ptxas:
                command = [ptxas, "--gpu-name=sm_90a", "--verbose", "--warn-on-spills",
                           str(directory / "kernel.ptx"), "--output-file", str(directory / "reassembled.cubin")]
                if metadata.get("enable_fp_fusion") is False:
                    command.append("--fmad=false")
                assembled = subprocess.run(command, capture_output=True, text=True)
                diagnostics = assembled.stdout + assembled.stderr
                (directory / "ptxas.log").write_text(diagnostics)
                entry["ptxas"] = {"command": command, "returncode": assembled.returncode,
                                   "scope": "Reassembled PTX; other backend flags may differ. Prefer original_cubin_resources when present.",
                                   **parse_ptxas(diagnostics)}
                # A reassembly failure is not a successful resource audit.
                if assembled.returncode:
                    raise RuntimeError(f"ptxas reassembly failed; see {directory / 'ptxas.log'}")
            entry["status"] = "ok" if ptxas else "compiled_without_resource_report"
            print(f"{job['name']}: shared={metadata.get('shared')} bytes; "
                  f"resources={entry.get('ptxas', 'assembler report unavailable')}", flush=True)
        except Exception as error:
            entry.update(status="error", error=str(error))
            (directory / "error.log").write_text(traceback.format_exc())
            print(f"{job['name']}: ERROR {error}", file=sys.stderr, flush=True)
        report["jobs"].append(entry)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "ok" if all(j["status"] == "ok" for j in report["jobs"]) else "incomplete"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {report_path}")
    return 0 if report["status"] == "ok" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "build/codegen")
    parser.add_argument("--preset", choices=("pdl", "combined", "tma"), default="pdl")
    parser.add_argument("--layers", nargs="+", choices=SHAPES, default=list(SHAPES))
    parser.add_argument("--tokens", nargs="+", type=int, default=list(SCORED_TOKENS))
    parser.add_argument("--ptxas", help="optional path to the NVIDIA assembler")
    parser.add_argument("--cuobjdump", help="optional path for original cubin resources and SASS")
    parser.add_argument("--plan", action="store_true", help="print jobs without importing Triton")
    args = parser.parse_args()
    if any(m <= 0 for m in args.tokens):
        parser.error("token counts must be positive")
    jobs = compile_jobs(args.preset, args.layers, args.tokens)
    if args.plan:
        print(json.dumps(jobs, indent=2))
        return 0
    return audit(args, jobs)


if __name__ == "__main__":
    raise SystemExit(main())
