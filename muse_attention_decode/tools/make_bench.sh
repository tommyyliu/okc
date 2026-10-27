#!/usr/bin/env bash
# Build a self-benchmarking check file: the kernel plus a dev tail that runs at import.
#   tools/make_bench.sh bench   -> build/bench.py   (correctness + per-shape config sweep vs SGLang)
#   tools/make_bench.sh ablate  -> build/ablate.py  (verify-loop ablation microbenchmark)
# Then: okc check build/bench.py --json > build/bench.json   (log lands in build/profiles/)
set -euo pipefail
cd "$(dirname "$0")/.."
kind="${1:-bench}"
mkdir -p build
cat submission.py "tools/${kind}_tail.py" > "build/${kind}.py"
echo "wrote build/${kind}.py"
