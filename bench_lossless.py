"""Separate decoding, GEMM scheduling and fusion costs on a development GPU.

  python bench_lossless.py --plan
  python bench_lossless.py --layers pooler --tokens 16 256

This is not a submission. Uses cold-L2 CUDA graphs, ABBA comparisons, exact
decoded-weight checks, and fresh-input matrix correctness. Decoder timings
include writing a full BF16 matrix; fused kernels do not incur that write.
"""

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SHAPES = {"qkv_gate": (2304, 8192), "o": (8192, 1024),
          "pooler": (4096, 8192), "unpooler": (8192, 4096)}


def raw_option(m, n, k):
    if m == 1:
        return ("gemv", 4 if k <= 1024 else 1, 4)
    bm = 16 if m <= 16 else (32 if m <= 32 else (64 if m <= 64 else 128))
    tiles = ((m + bm - 1) // bm) * ((n + 63) // 64)
    splits = 1
    while tiles * splits < 256 and splits < 16 and k % (splits * 2 * 256) == 0:
        splits *= 2
    return ("gemm", bm, 64, 128, splits, 8 if bm >= 128 else 4, 3)


def run(args):
    def blocked(message):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"status": "blocked", "error": message, "cases": []}, indent=2) + "\n")
        print(f"{message}\nReport: {args.output}")
        return 2

    try:
        import torch
        import triton
        import reference
        import lossless_probes as probes
        import submission_triton_lossless as candidate
        import submission_triton_selective as selective
        from tune_streaming import capture, compare_graphs
    except ImportError as error:
        return blocked(f"GPU benchmark dependencies unavailable: {error}")
    if not torch.cuda.is_available():
        return blocked("Timing requires a CUDA GPU; --plan and inspect_codegen.py do not.")
    if torch.cuda.get_device_capability()[0] < 9:
        raise SystemExit("Use Hopper or newer for these large shared-memory launch settings.")
    report = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "triton": triton.__version__, "cuda": torch.version.cuda, "seed": args.seed,
              "source_sha256": {p: hashlib.sha256((HERE / p).read_bytes()).hexdigest()
                                for p in ("submission_triton_lossless.py", "submission_triton_selective.py",
                                          "lossless_probes.py", "tune_streaming.py")},
              "scope": "development cold-L2 CUDA graphs; not judge timings", "cases": []}
    report["status"] = "running"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        weights = reference.generate_weights(args.seed)
        state = candidate.prepare(weights)
        flush = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
        for index, (m, layer) in enumerate((m, layer) for layer in args.layers for m in args.tokens):
            data = reference.generate_input(m, layer, args.seed + index + 1)
            fresh = reference.generate_input(m, layer, args.seed + index + 10001, dist="outliers")
            n, k = data["config"]["n"], data["config"]["k"]
            w = weights[layer][:n]
            representation = state["compressed"].get(layer)
            if representation is None:
                representation = candidate.compress_weight(weights[layer])
                if representation is None:
                    raise RuntimeError(f"unsupported compression shape: {layer}")
                state["compressed"][layer] = representation
            payload, offsets, bases, ratio = representation
            groups = n * k // 256
            active_payload = int(offsets[groups].item()) if groups < offsets.numel() else payload.numel()
            handles = {}
            buffers = {name: torch.empty_like(w) for name in
                       ("copy", "decode_original", "decode_aligned", "decode_group")}

            def copy_weight():
                handles["copy"] = probes._copy_matrix[(triton.cdiv(n * k, 1024),)](
                    w, buffers["copy"], n * k, BLOCK=1024, num_warps=4, num_stages=1)
                return buffers["copy"]

            def decode_weight(mode):
                if mode == "decode_group":
                    handles[mode] = probes._decode_group[(groups,)](
                        payload, offsets, bases, buffers[mode], num_warps=4, num_stages=1)
                else:
                    handles[mode] = probes._decode_matrix[(triton.cdiv(groups, args.groups_per_cta),)](
                        payload, offsets, bases, buffers[mode], groups,
                        G=args.groups_per_cta, ALIGNED=mode == "decode_aligned", num_warps=4, num_stages=1)
                return buffers[mode]

            def staged_mm():
                decoded = decode_weight("decode_original")
                return torch.mm(data["x"], decoded.T)

            paths = {"copy": copy_weight,
                     "decode_original": lambda: decode_weight("decode_original"),
                     "decode_aligned": lambda: decode_weight("decode_aligned"),
                     "decode_group": lambda: decode_weight("decode_group"),
                     "selective": lambda: selective.kernel_fn(data, state),
                     "raw_matching_tiles": lambda: candidate.run_config(data, state, raw_option(m, n, k)),
                     "fused_lossless": lambda: candidate.kernel_fn(data, state),
                     "decode_then_cublas": staged_mm}
            graphs = {}
            for name, fn in paths.items():
                # Warming/capture may compile; all compilation precedes timing.
                graphs[name] = capture(torch, fn, flush)
                graphs[name][0].replay()
                torch.cuda.synchronize()
                output = graphs[name][3]
                if name in buffers:
                    if not torch.equal(output.view(torch.int16), w.view(torch.int16)):
                        raise AssertionError(f"{name}: decoded weight bits differ")
                else:
                    ok, message = reference.check_implementation(data, output, weights)
                    if not ok:
                        raise AssertionError(f"{name}: {message}")
            # Every graph sees the same new activations at its captured address.
            data["x"].copy_(fresh["x"])
            for name, graph in graphs.items():
                if name in buffers:
                    continue
                graph[0].replay()
                torch.cuda.synchronize()
                ok, message = reference.check_implementation(data, graph[3], weights)
                if not ok:
                    raise AssertionError(f"{name} fresh graph replay: {message}")
            row = {"layer": layer, "m": m, "n": n, "k": k,
                   "active_compressed_bytes": active_payload + groups * 6,
                   "raw_weight_bytes": n * k * 2, "whole_layer_byte_ratio": ratio,
                   "matching_raw_option": raw_option(m, n, k), "paths": {}}
            for name, graph in graphs.items():
                baseline = "copy" if name in buffers else "selective"
                stats = compare_graphs(graphs[baseline], graph, args.pairs)
                row["paths"][name] = {"baseline": baseline, "correct": True, **stats}
                print(f"m={m:3d} {layer:9s} {name:22s} "
                      f"{stats['candidate_us']:9.2f} us; {stats['ratio']:.3f}x {baseline}", flush=True)
            # Runtime resource attributes are read only after actual GPU launches.
            row["decoder_resources"] = {
                name: {"registers_per_thread": getattr(kernel, "n_regs", None),
                       "n_spills_raw": getattr(kernel, "n_spills", None),
                       "shared_bytes": getattr(kernel.metadata, "shared", None)}
                for name, kernel in handles.items()}
            report["cases"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            del graphs, buffers, handles
    report["status"] = "ok"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {args.output}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", nargs="+", choices=SHAPES, default=["qkv_gate", "pooler", "unpooler"])
    parser.add_argument("--tokens", nargs="+", type=int, choices=(1, 8, 16, 32, 128, 256), default=[1, 16, 256])
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--groups-per-cta", type=int, choices=(1, 4, 16), default=4)
    parser.add_argument("--seed", type=int, default=80137)
    parser.add_argument("--output", type=Path, default=HERE / "profiles/lossless_benchmark.json")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.pairs < 10:
        parser.error("--pairs must be at least 10")
    if args.plan:
        for layer in args.layers:
            full_n, k = SHAPES[layer]
            for m in args.tokens:
                n = 1024 if layer == "unpooler" and m <= 8 else full_n
                print(f"{layer} m={m} n={n} k={k}; matching raw option {raw_option(m, n, k)}")
        print("Paths: copy, original/aligned/group decoder, selective, matching raw tiles, fused lossless, decode + cuBLAS")
    else:
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
