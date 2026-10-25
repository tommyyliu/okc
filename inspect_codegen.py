"""Compile projection probes for Hopper without importing Torch or using a GPU.

  python inspect_codegen.py --plan
  python inspect_codegen.py --output profiles/codegen-h200

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


HERE = Path(__file__).resolve().parent
CASES = ("gemv-qkv", "gemv-pooler", "gemm-pooler-16", "gemm-pooler-256",
         "gemm-unpooler-128", "decoder")
EXPERIMENT_CASES = ("orientation", "pdl", "tma")


def compile_jobs(cases=CASES):
    jobs = []
    for case in cases:
        if case == "tma":
            shapes = [(layer, m) for layer in ("qkv", "pooler", "unpooler")
                      for m in (8, 16, 32, 128, 256) if layer != "unpooler" or m > 8]
            shapes += [("pooler", 7), ("unpooler", 130), ("qkv", 15),
                       ("qkv", 17), ("unpooler", 9)]
            for layer, m in shapes:
                n, k = (2304, 8192) if layer == "qkv" else (4096, 8192) if layer == "pooler" else (8192, 4096)
                bm = 16 if m <= 16 else 32 if m <= 32 else 64
                tiles = ((m + bm - 1) // bm) * ((n + 63) // 64)
                s = 1
                while tiles * s < 128 and s < 8 and k % (s * 2 * 128) == 0:
                    s *= 2
                jobs.append(dict(name=f"tma-{layer}-{m}", source="tma", kernel="_tma_weight_major",
                                 signature={"WD": "tensordesc<bf16[64,128]>",
                                            "XD": f"tensordesc<bf16[{bm},128]>",
                                            "Y": "*bf16", "P": "*fp32"},
                                 constants={"M": m, "N": n, "K": k, "S": s,
                                            "BM": bm, "BN": 64, "BK": 128},
                                 num_warps=4, num_stages=3))
            continue
        if case == "orientation":
            for m in (2, 8, 16, 17, 32):
                jobs.append(dict(name=f"orientation-o-{m}", source="orientation",
                                 kernel="_weight_major_gemm",
                                 signature={"W": "*bf16", "X": "*bf16", "Y": "*bf16"},
                                 constants={"M": m, "N": 8192, "K": 1024,
                                            "BM": 16 if m <= 16 else 32, "BN": 64, "BK": 128},
                                 num_warps=4, num_stages=4))
            continue
        if case == "pdl":
            # Launch settings match selective, including irregular correctness cases.
            for m in (8, 9, 16, 32, 128, 256):
                bm = 16 if m <= 16 else 32 if m <= 32 else 64
                s = 32 if m <= 128 else 16
                jobs.append(dict(name=f"pdl-router-{m}", source="pdl", kernel="_gemm",
                                 signature={"X": "*bf16", "WH": "*bf16", "WL": "*bf16",
                                            "Y": "*fp32", "P": "*fp32"},
                                 constants={"M": m, "N": 256, "K": 8192, "S": s,
                                            "ROUTER": True, "BM": bm, "BN": 64, "BK": 64},
                                 num_warps=4, num_stages=3))
            for layer, m, n, s in (("qkv", 8, 2304, 8), ("qkv", 16, 2304, 8),
                                    ("pooler", 7, 4096, 4), ("pooler", 8, 4096, 4),
                                    ("pooler", 16, 4096, 4), ("pooler", 32, 4096, 4),
                                    ("unpooler", 5, 1024, 16), ("unpooler", 8, 1024, 16)):
                sliced = layer == "unpooler"
                signature = ({"X": "*bf16", "WH": "*bf16", "WL": "*bf16"}
                             if sliced else {"X": "*bf16", "W": "*bf16"})
                signature.update(Y="*bf16", P="*fp32")
                constants = {"M": m, "N": n, "K": 4096 if sliced else 8192, "S": s,
                             "BM": 16 if m <= 16 else 32, "BN": 64, "BK": 64 if sliced else 128}
                if sliced:
                    constants["ROUTER"] = False
                jobs.append(dict(name=f"pdl-{layer}-{m}", source="pdl",
                                 kernel="_gemm" if sliced else "_stream_gemm",
                                 signature=signature, constants=constants,
                                 num_warps=4, num_stages=3))
            for total, s, dtype in ((2048, 32, "fp32"), (32768, 8, "bf16"),
                                     (131072, 8, "fp32"), (8192, 16, "bf16")):
                jobs.append(dict(name=f"pdl-reduce-{total}-{s}", source="pdl", kernel="_reduce",
                                 signature={"P": "*fp32", "Y": f"*{dtype}"},
                                 constants={"TOTAL": total, "S": s, "B": 256},
                                 num_warps=4, num_stages=3, launch_pdl=True))
            continue
        if case == "decoder":
            signature = {"Payload": "*u8", "Offsets": "*i32", "Bases": "*i16", "Y": "*bf16"}
            for aligned in (False, True):
                jobs.append(dict(name=f"decoder-{'aligned' if aligned else 'original'}", source="probes",
                                 kernel="_decode_matrix", signature=signature,
                                 constants={"GROUPS": 131072, "G": 4, "ALIGNED": aligned},
                                 num_warps=4, num_stages=1))
            jobs.append(dict(name="decoder-group-branch", source="probes", kernel="_decode_group",
                             signature=signature, constants={}, num_warps=4, num_stages=1))
            jobs.append(dict(name="raw-copy", source="probes", kernel="_copy_matrix",
                             signature={"W": "*bf16", "Y": "*bf16"},
                             constants={"TOTAL": 33554432, "BLOCK": 1024}, num_warps=4, num_stages=1))
            continue
        gemv = case.startswith("gemv")
        if gemv:
            n, k = (2304, 8192) if case == "gemv-qkv" else (4096, 8192)
            constants = {"N": n, "K": k, "ROWS": 1}
            raw_constants = dict(constants, BLOCK_K=k)
            packed_constants = dict(constants, GROUPS_K=k // 256)
            raw_kernel, packed_kernel = "_gemv_full", "_lossless_gemv"
            raw_signature = {"X": "*bf16", "W": "*bf16", "Y": "*bf16"}
            packed_signature = {"X": "*bf16", "Payload": "*u8", "Offsets": "*i32",
                                "Bases": "*i16", "Y": "*bf16"}
            warps, stages = 4, 3
        else:
            m = int(case.rsplit("-", 1)[1])
            n, k = (8192, 4096) if "unpooler" in case else (4096, 8192)
            bm = 16 if m <= 16 else 128
            tiles = ((m + bm - 1) // bm) * ((n + 63) // 64)
            splits = 1
            while tiles * splits < 256 and splits < 16 and k % (splits * 2 * 256) == 0:
                splits *= 2
            raw_constants = packed_constants = {"M": m, "N": n, "K": k, "S": splits,
                                                 "BM": bm, "BN": 64, "BK": 128}
            raw_kernel, packed_kernel = "_stream_gemm", "_lossless_gemm"
            raw_signature = {"X": "*bf16", "W": "*bf16", "Y": "*bf16", "P": "*fp32"}
            packed_signature = {"X": "*bf16", "Payload": "*u8", "Offsets": "*i32",
                                "Bases": "*i16", "Y": "*bf16", "P": "*fp32"}
            warps, stages = (8 if bm == 128 else 4), 3
        for variant, source, kernel, signature, const in (
                ("raw", "selective", raw_kernel, raw_signature, raw_constants),
                ("lossless", "lossless", packed_kernel, packed_signature, packed_constants)):
            jobs.append(dict(name=f"{case}-{variant}", source=source, kernel=kernel,
                             signature=signature, constants=const, num_warps=warps, num_stages=stages))
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


def runtime_reports(text):
    """Read diagnostic JSON from plain logs or an okc result's retained log tail."""
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        document = None
    if isinstance(document, dict):
        result = document.get("result") or document
        text = result.get("log_tail", "")
    records = {}
    decoder = json.JSONDecoder()
    for match in re.finditer(r"LOSSLESS_CODEGEN\s+", text):
        try:
            record, _ = decoder.raw_decode(text[match.end():])
        except json.JSONDecodeError:
            continue  # A retained log may start/end inside a record.
        if not isinstance(record, dict):
            continue
        key = json.dumps(record, sort_keys=True)
        if key not in records:
            records[key] = {"metadata": record, "occurrences": 0}
        records[key]["occurrences"] += 1
    return list(records.values())


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
    bundles = {"selective": [HERE / "submission_triton_selective.py"],
               "lossless": [HERE / "submission_triton_lossless.py"],
               "probes": [HERE / "submission_triton_lossless.py", HERE / "lossless_probes.py"],
               "orientation": [HERE / "submission_triton_orientation.py"],
               "pdl": [HERE / "submission_triton_pdl.py"],
               "tma": [HERE / "submission_triton_tma.py"]}
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
            if job["source"] == "tma":
                summary = entry["static_summary"]
                entry["uses_tma_and_wgmma"] = (
                    any(k.startswith("cp.async.bulk.tensor") for k in summary["cp_async_instructions"])
                    and any(k.startswith("wgmma.mma_async") for k in summary["mma_instructions"]))
                if not entry["uses_tma_and_wgmma"]:
                    raise RuntimeError("TMA/WGMMA instructions missing from generated kernel")
            if job["source"] == "pdl" or (job["source"] == "tma" and job["constants"]["S"] > 1):
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
    parser.add_argument("--output", type=Path, default=HERE / "profiles/codegen-h200")
    parser.add_argument("--cases", nargs="+", choices=CASES + EXPERIMENT_CASES, default=list(CASES))
    parser.add_argument("--ptxas", help="optional path to the NVIDIA assembler")
    parser.add_argument("--cuobjdump", help="optional path for original cubin resources and SASS")
    parser.add_argument("--read-result", type=Path, help="extract runtime metadata from a diagnostic check JSON or log; no Triton needed")
    parser.add_argument("--plan", action="store_true", help="print jobs without importing Triton")
    args = parser.parse_args()
    if args.read_result:
        records = runtime_reports(args.read_result.read_text())
        report = {"status": "ok" if records else "no_metadata", "records": records,
                  "source": str(args.read_result),
                  "scope": "Actual compiled-kernel metadata retained in the log; may omit earlier truncated records."}
        args.output.mkdir(parents=True, exist_ok=True)
        output = args.output / "runtime_report.json"
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        print(f"Report: {output}")
        return 0 if records else 1
    jobs = compile_jobs(args.cases)
    if args.plan:
        print(json.dumps(jobs, indent=2))
        return 0
    return audit(args, jobs)


if __name__ == "__main__":
    raise SystemExit(main())
