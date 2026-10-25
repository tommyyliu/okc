"""Calculate score budgets from an actual check, using the task's load formula.

  python projection_budget.py --result submission-4 --target 1.61

Floors are optimistic bounds from the result, not predictions. Removing router
time is an impossible oracle used only to bound the value of router tuning.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


def geometric_mean(values):
    values = list(values)
    return math.exp(sum(math.log(v) for v in values) / len(values))


def score_rows(rows, replacement=None):
    loads = defaultdict(lambda: [0.0, 0.0])
    for row in rows:
        case = row["case"]
        weight = case["weight"]
        time = row["median_ns"] if replacement is None else replacement(row)
        if not math.isfinite(time) or time < 0:
            raise ValueError("replacement times must be finite and nonnegative")
        load = loads[case["load"]]
        load[0] += row["baseline_ns"] * weight
        load[1] += time * weight
    if not loads or any(time <= 0 for _, time in loads.values()):
        raise ValueError("need positive total times for at least one load")
    return geometric_mean(base / time for base, time in loads.values()), loads


def build_budget(document, target):
    if not math.isfinite(target) or target <= 0:
        raise ValueError("target must be finite and positive")
    result = document.get("result", document)
    if not result.get("passed"):
        raise ValueError("input must be a passing check result")
    rows = [r for r in result["records"] if r.get("event") == "benchmark"]
    current, loads = score_rows(rows)
    if not math.isclose(current, result["speedup"], rel_tol=1e-6):
        raise ValueError("per-shape score reconstruction does not match the reported speedup")
    report = {"target_speedup": target, "target_source": "user supplied; not a verified live leaderboard query",
              "measured_speedup": current, "reported_score_us": result["score_us"],
              "target_score_us": result["baseline_us"] / target,
              "required_throughput_factor": target / current,
              "required_latency_reduction_fraction": 1 - current / target,
              "load_budgets": {}, "optimistic_scenarios": {}}
    for name, (base, time) in loads.items():
        report["load_budgets"][name] = {
            "baseline_us": base / 1000, "measured_us": time / 1000,
            "uniform_reduction_budget_us": time * current / target / 1000,
            "note": "One sufficient uniform allocation; individual loads may trade gains."}
    scenarios = {"router_free_impossible_oracle": lambda r: 0 if r["case"]["layer"] == "router" else r["median_ns"],
                 "all_shapes_at_reported_floors": lambda r: r["speed_of_light_ns"]}
    layers = sorted({r["case"]["layer"] for r in rows})
    for layer in layers:
        scenarios[f"only_{layer}_at_floor"] = lambda r, layer=layer: (
            r["speed_of_light_ns"] if r["case"]["layer"] == layer else r["median_ns"])
    for name, replace in scenarios.items():
        speedup, _ = score_rows(rows, replace)
        report["optimistic_scenarios"][name] = {"speedup": speedup, "reaches_target": speedup >= target}
    report["caveat"] = "Floor/oracle scenarios are mathematical bounds, not implementable performance claims."
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=Path(__file__).with_name("submission-4"))
    parser.add_argument("--target", type=float, default=1.61)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "profiles/target_budget.json")
    args = parser.parse_args()
    report = build_budget(json.loads(args.result.read_text()), args.target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Measured {report['measured_speedup']:.4f}x; target {args.target:.4f}x")
    print(f"Required latency reduction: {100 * report['required_latency_reduction_fraction']:.2f}%")
    print(f"Target score: {report['target_score_us']:.2f} us")
    for name, result in report["optimistic_scenarios"].items():
        print(f"{name}: {result['speedup']:.4f}x")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
