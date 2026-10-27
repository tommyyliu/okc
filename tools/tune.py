"""Offline cold-L2 CUDA graph tuning; this file is never submitted.

Run on H200 for meaningful choices:
  python -m tools.tune
  python -m tools.validate build/tuned.py --nearby
  okc check build/tuned.py --json

No GPU is needed for --plan. All selected settings are exported as literals
into a standalone submission; neither timing nor reference imports go into it.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import statistics
from types import SimpleNamespace


from projection.shapes import SHAPES, SCORED_TOKENS, shape_config

HERE = Path(__file__).resolve().parents[1]
BUCKETS = SCORED_TOKENS


def options(m, n, k, layer, *, stages=(2, 3, 4), k_tiles=(64, 128, 256),
            families=("gemm", "weight_major"), splits=None):
    if layer != "router":
        yield ("torch",)
    if m == 1:
        for rows in (1, 2, 4):
            for warps in (4, 8):
                if rows * k <= warps * 4096:
                    yield ("gemv", rows, warps)
    elif layer != "router":
        bm = 16 if m <= 16 else (32 if m <= 32 else 128)
        for bn, warps in ((32, 4), (64, 4), (128, 8)):
            tiles = ((m + bm - 1) // bm) * ((n + bn - 1) // bn)
            occupancy_split = 1
            while tiles * occupancy_split < 264 and occupancy_split < 16:
                occupancy_split *= 2
            partitions = (splits if splits is not None else
                          sorted({1, max(1, occupancy_split // 4),
                                  max(1, occupancy_split // 2), occupancy_split,
                                  min(32, occupancy_split * 2)}))
            for bk in k_tiles:
                for split in partitions:
                    if k % (split * bk) == 0:
                        for family in families:
                            for depth in stages:
                                yield (family, bm, bn, bk, split, warps, depth)


def read_source_choices(source, layers=tuple(SHAPES), tokens=SCORED_TOKENS):
    """Inspect standalone configuration without importing Torch or GPU kernels."""
    tree = ast.parse(source)
    host = [n for n in tree.body if isinstance(n, (ast.Assign, ast.FunctionDef))
            and not (isinstance(n, ast.FunctionDef) and (n.decorator_list or
                     n.name in ("prepare", "run_config", "kernel_fn")))]
    ns = {}
    exec(compile(ast.Module(body=host, type_ignores=[]), "<source configuration>", "exec"), ns)
    return source_choices(SimpleNamespace(**ns), layers, tokens)


def write_ablations(source, directory, layers, tokens):
    """Export one-factor small-GEMM experiments for remote checks, without a GPU.

    The streaming reference changes cuBLAS cases to the existing streaming tile;
    all other variants change exactly one field of that reference. The original
    source remains a separate control, including its cuBLAS choices.
    """
    choices = read_source_choices(source)
    keys = [(layer, m) for layer in layers for m in tokens]
    if any(layer not in ("qkv_gate", "pooler") or m not in (8, 16, 32)
           for layer, m in keys):
        raise ValueError("ablations require QKV/pooler at M=8/16/32")
    streaming = dict(choices)
    for layer, m in keys:
        option = choices[(layer, m)]
        if option[0] == "torch":
            option = ("gemm", 16 if m <= 16 else 32, 64, 128,
                      8 if layer == "qkv_gate" else 4, 4, 3)
        if option[0] != "gemm":
            raise ValueError("ablations require a streaming GEMM or cuBLAS source control")
        streaming[(layer, m)] = option
    directory.mkdir(parents=True, exist_ok=True)
    control = directory / "control.py"
    control.write_text(source)
    manifest = {"source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "measured": False, "control": "control.py", "experiments": []}
    changes = (("streaming", None, None), ("stages2", 6, 2), ("stages4", 6, 4),
               ("bk64", 3, 64), ("bk256", 3, 256),
               ("splits_half", 4, "half"), ("splits_double", 4, "double"),
               ("weight_major", 0, "weight_major"))
    for name, field, value in changes:
        selected = dict(streaming)
        for key in keys:
            option = list(selected[key])
            if field is not None:
                option[field] = (max(1, option[field] // 2) if value == "half" else
                                 option[field] * 2 if value == "double" else value)
            if SHAPES[key[0]][1] % (option[3] * option[4]):
                raise ValueError(f"{name}: K must be divisible by splits * BK: {key}")
            selected[key] = tuple(option)
        path = directory / f"{name}.py"
        export_submission(source, path, selected)
        comparison = choices if name == "streaming" else streaming
        manifest["experiments"].append({
            "file": path.name,
            "compare_to": "control.py" if name == "streaming" else "streaming.py",
            "changed_cases": [{"layer": layer, "m": m, "before": comparison[(layer, m)],
                               "after": selected[(layer, m)]}
                              for layer, m in keys if selected[(layer, m)] != comparison[(layer, m)]]})
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def export_submission(source, destination, choices):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "TUNED_CONFIGS"
                        for t in n.targets))
    lines = source.splitlines(keepends=True)
    literal = "TUNED_CONFIGS = {\n" + "".join(
        f"    {key!r}: {value!r},\n" for key, value in choices.items()) + "}\n"
    result = "".join(lines[:node.lineno - 1]) + literal + "".join(lines[node.end_lineno:])
    ast.parse(result)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(result)


def source_choices(module, layers=tuple(SHAPES), tokens=SCORED_TOKENS):
    """Preserve exact overrides and every unswept scored configuration."""
    choices = dict(module.TUNED_CONFIGS)
    for layer in layers:
        for m in tokens:
            choices[(layer, m)] = module.select_config(shape_config(layer, m))
    return choices


def load_candidate(path):
    spec = importlib.util.spec_from_file_location("streaming_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture(torch, fn, flush):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    # External event nodes keep cache eviction inside the graph but outside
    # the measured interval. No CPU launch gap separates the two operations.
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flush.zero_()
        start.record()
        output = fn()
        end.record()
    return graph, start, end, output


def measure(graph_info):
    graph, start, end, _ = graph_info
    graph.replay()
    end.synchronize()
    return start.elapsed_time(end) * 1000  # microseconds


def compare_graphs(baseline, candidate, pairs):
    ratios, base_times, candidate_times = [], [], []
    for _ in range(pairs):
        # ABBA balances a gradual clock or temperature drift. Both graphs
        # independently evict L2 before every invocation.
        b1 = measure(baseline)
        c1 = measure(candidate)
        c2 = measure(candidate)
        b2 = measure(baseline)
        ratios.append(((b1 / c1) * (b2 / c2)) ** 0.5)
        base_times.extend((b1, b2))
        candidate_times.extend((c1, c2))
    return {"ratio": statistics.median(ratios),
            "p10_ratio": sorted(ratios)[int(0.1 * len(ratios))],
            "baseline_us": statistics.median(base_times),
            "candidate_us": statistics.median(candidate_times)}


def correct(reference, data, output, weights):
    ok, message = reference.check_implementation(data, output, weights)
    if not ok:
        raise AssertionError(message)


def run(args):
    import torch
    import reference

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required; use --plan to inspect the sweep.")
    if torch.cuda.get_device_capability()[0] < 8:
        raise SystemExit("BF16 tensor cores require compute capability >= 8.0.")
    module = load_candidate(args.source)
    weights = reference.generate_weights(args.seed)
    state = module.prepare(weights)
    # A full 128 MiB memset evicts the H200's 50 MiB L2; it is never timed.
    flush = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    choices = source_choices(module)
    report = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "seed": args.seed, "source": str(args.source),
              "search": {"stages": args.stages, "k_tiles": args.k_tiles,
                         "families": args.families, "splits": args.splits}, "cases": []}
    print(f"GPU: {report['gpu']}; measuring cold-L2 CUDA graph nodes", flush=True)
    with torch.no_grad():
        for index, (m, layer) in enumerate((m, layer) for m in args.tokens for layer in args.layers):
            data = reference.generate_input(m, layer, args.seed + index + 1)
            cfg = data["config"]
            baseline_option = choices[(layer, m)]
            baseline = capture(torch, lambda: module.run_config(data, state, baseline_option), flush)
            baseline[0].replay()
            torch.cuda.synchronize()
            correct(reference, data, baseline[3], weights)
            case = {"m": m, "layer": layer, "baseline": baseline_option, "trials": []}
            candidates = [option for option in dict.fromkeys(options(m, cfg["n"], cfg["k"], layer,
                          stages=args.stages, k_tiles=args.k_tiles,
                          families=args.families, splits=args.splits))
                          if option != baseline_option]
            random.Random(args.seed + index).shuffle(candidates)
            best = baseline_option
            best_ratio = 1.0
            for option in candidates:
                try:
                    trial = capture(torch, lambda: module.run_config(data, state, option), flush)
                except Exception as error:
                    if type(error).__name__ not in ("OutOfResources", "CompilationError",
                                                   "CompileTimeAssertionFailure"):
                        raise
                    case["trials"].append({"option": option, "skipped": str(error)})
                    print(f"SKIP {m=} {layer} {option}: {error}", flush=True)
                    continue
                trial[0].replay()
                torch.cuda.synchronize()
                correct(reference, data, trial[3], weights)
                stats = compare_graphs(baseline, trial, args.pairs)
                case["trials"].append({"option": option, **stats})
                print(f"m={m:3d} {layer:9s} {option!s:40s} "
                      f"{stats['ratio']:.3f}x source dispatch", flush=True)
                if stats["ratio"] > max(1.03, best_ratio) and stats["p10_ratio"] > 1.015:
                    best, best_ratio = option, stats["ratio"]
                del trial
            # Confirm on independent outlier activations, with more samples.
            # Winner selection and confirmation never share their inputs.
            if best != baseline_option:
                fresh = reference.generate_input(m, layer, args.seed + index + 10001, dist="outliers")
                data["x"].copy_(fresh["x"])
                trial = capture(torch, lambda: module.run_config(data, state, best), flush)
                trial[0].replay()
                torch.cuda.synchronize()
                correct(reference, data, trial[3], weights)
                confirmation = compare_graphs(baseline, trial, max(40, args.pairs))
                case["confirmation"] = confirmation
                if confirmation["ratio"] < 1.03 or confirmation["p10_ratio"] < 1.015:
                    best = baseline_option
                else:
                    # Also validate the candidate on nearby shapes before retaining it.
                    position = BUCKETS.index(m)
                    lower = BUCKETS[position - 1] + 1 if position else 1
                    nearby_tokens = {lower, max(lower, m - 1)}
                    if m > 1:
                        nearby_tokens.add(m + 1)
                    for nearby in sorted(nearby_tokens - {m}):
                        nearby_data = reference.generate_input(
                            nearby, layer, args.seed + index + nearby + 20001, dist="outliers")
                        out = module.run_config(nearby_data, state, best)
                        correct(reference, nearby_data, out, weights)
                del trial
            choices[(layer, m)] = best
            case["selected"] = best
            report["cases"].append(case)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
            print(f"SELECT m={m} {layer}: {best}", flush=True)
            del baseline
    # Export only after the requested sweep has finished successfully.
    export_submission(args.source.read_text(), args.output, choices)
    print(f"Wrote {args.output}; report: {args.report}")
    print(f"Next: python -m tools.validate {args.output} --nearby")
    print(f"Then: okc check {args.output} --json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE / "submission.py")
    parser.add_argument("--output", type=Path, default=HERE / "build/tuned.py")
    parser.add_argument("--report", type=Path, default=HERE / "build/tuning.json")
    parser.add_argument("--layers", nargs="+", choices=SHAPES, default=list(SHAPES))
    parser.add_argument("--tokens", nargs="+", type=int, choices=BUCKETS, default=[1, 8, 16, 32, 128, 256])
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=27641)
    parser.add_argument("--stages", nargs="+", type=int, choices=(1, 2, 3, 4, 5), default=[2, 3, 4])
    parser.add_argument("--k-tiles", nargs="+", type=int, choices=(32, 64, 128, 256), default=[64, 128, 256])
    parser.add_argument("--families", nargs="+", choices=("gemm", "weight_major"),
                        default=["gemm", "weight_major"])
    parser.add_argument("--splits", nargs="+", type=int, choices=(1, 2, 4, 8, 16, 32),
                        help="default: occupancy-derived partitions plus quarter, half and double")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--ablations", type=Path,
                      help="export small QKV/pooler experiments and a manifest without a GPU")
    args = parser.parse_args()
    if args.pairs < 10:
        parser.error("--pairs must be at least 10")
    if args.source.resolve() == args.output.resolve():
        parser.error("--output must differ from --source")
    if args.ablations:
        if args.source.resolve().parent == args.ablations.resolve():
            parser.error("--ablations must differ from the source directory")
        try:
            manifest = write_ablations(args.source.read_text(), args.ablations, args.layers, args.tokens)
        except ValueError as error:
            parser.error(str(error))
        print(f"Wrote control and {len(manifest['experiments'])} experiments to {args.ablations}")
    elif args.plan:
        choices = read_source_choices(args.source.read_text())
        for m in args.tokens:
            for layer in args.layers:
                n, k = SHAPES[layer]
                if layer == "unpooler" and m <= 8:
                    n = 1024
                current = choices[(layer, m)]
                count = len(set(options(m, n, k, layer, stages=args.stages,
                                       k_tiles=args.k_tiles, families=args.families,
                                       splits=args.splits)) | {current})
                print(f"m={m:3d} {layer:9s}: {current}; {count} choices")
    else:
        run(args)


if __name__ == "__main__":
    main()
