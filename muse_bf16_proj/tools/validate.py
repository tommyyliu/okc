"""GPU correctness and CUDA graph checks outside the submitted file.

Usage: python -m tools.validate submission.py --nearby

Run from the project root, beside reference.py and checks.py. No SGLang or
okc login is needed. This is a development check; okc check remains authoritative.
"""

import argparse
import importlib.util
from pathlib import Path
import time

import torch

import reference


HARD_CASES = [
    (1, "qkv_gate", "normal"),
    (7, "pooler", "normal"),
    (33, "o", "normal"),
    (130, "unpooler", "normal"),
    (5, "unpooler", "normal"),
    (8, "router", "normal"),
    (9, "router", "outliers"),
    (512, "qkv_gate", "outliers"),
    (1000, "unpooler", "outliers"),
]


def load_submission(path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_correct(data, output, weights, stage):
    ok, message = reference.check_implementation(data, output, weights)
    if not ok:
        raise AssertionError(f"{stage}: {message}")
    return message


def check_case(module, state, weights, m, layer, dist, seed):
    data = reference.generate_input(m, layer, seed=seed, dist=dist)
    output = module.kernel_fn(data, state)
    torch.cuda.synchronize()
    message = assert_correct(data, output, weights, "eager")

    # Warm compilation and allocator paths before capture, on a side stream.
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            module.kernel_fn(data, state)
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = module.kernel_fn(data, state)
    graph.replay()
    torch.cuda.synchronize()
    assert_correct(data, captured_output, weights, "graph replay")

    # Reuse the captured addresses with genuinely new activations. This catches
    # accidentally retaining outputs or computing activation work in prepare.
    fresh = reference.generate_input(m, layer, seed=seed + 10000, dist=dist)
    data["x"].copy_(fresh["x"])
    graph.replay()
    torch.cuda.synchronize()
    assert_correct(data, captured_output, weights, "graph replay with fresh inputs")
    print(f"PASS m={m:4d} {layer:10s} {dist:8s} eager + graph + fresh: {message}",
          flush=True)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--nearby", action="store_true",
                        help="also check nearby and irregular batch sizes with outliers")
    parser.add_argument("--seed", type=int, default=9173,
                        help="fresh weight/input seed; differs from the example weights")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("An NVIDIA GPU runtime is required.")
    if torch.cuda.get_device_capability()[0] < 8:
        raise SystemExit("BF16 tensor cores require compute capability >= 8.0; "
                         "choose an Ampere-or-newer GPU runtime.")

    print(f"GPU: {torch.cuda.get_device_name()}; PyTorch: {torch.__version__}", flush=True)
    module = load_submission(args.submission.resolve())
    weights = reference.generate_weights(seed=args.seed)
    start = time.perf_counter()
    state = module.prepare(weights)
    torch.cuda.synchronize()
    print(f"prepare: {time.perf_counter() - start:.2f} s", flush=True)

    cases = HARD_CASES + [(m, layer, "normal")
                          for m in (1, 8, 16, 32, 128, 256)
                          for layer in reference.LAYERS]
    if args.nearby:
        cases += [(m, layer, "outliers")
                  for m in (2, 15, 17, 31, 34, 127, 257)
                  for layer in reference.LAYERS]
    cases = list(dict.fromkeys(cases))
    for index, (m, layer, dist) in enumerate(cases):
        check_case(module, state, weights, m, layer, dist, args.seed + index + 1)
    print(f"All {len(cases)} cases passed eager execution and graph replay "
          "with original and fresh activations.")


if __name__ == "__main__":
    main()
