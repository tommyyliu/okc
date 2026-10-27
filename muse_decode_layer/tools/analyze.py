"""Summarize a check and rank scored shapes by their contribution to load latency."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


def geometric_mean(values):
    values = list(values)
    return math.exp(sum(math.log(v) for v in values) / len(values))


def case_key(case):
    ignored = {"seed", "weight", "load", "router_load", "router_distinct"}
    return json.dumps({k: v for k, v in case.items() if k not in ignored}, sort_keys=True)


def analyze(document, target_us=9580.0):
    if not math.isfinite(target_us) or target_us <= 0:
        raise ValueError("target time must be finite and positive")
    result = document.get("result", document)
    records = result.get("records", [])
    tests = [r for r in records if r.get("event") == "test"]
    failures = [r for r in records if r.get("status") == "fail"]
    report = {
        "job": document.get("id"),
        "content_hash": document.get("content_hash"),
        "passed": result.get("passed") is True,
        "tests_passed": sum(r.get("status") == "pass" for r in tests),
        "tests_total": len(tests),
        "failures": [{k: r[k] for k in ("event", "index", "case", "message", "error") if k in r}
                     for r in failures],
        "error": result.get("error"),
        "score_us": result.get("score_us"),
        "speedup": result.get("speedup"),
        "speedup_ci": result.get("speedup_ci"),
        "has_timings": False,
    }
    if not report["passed"]:
        report["next_step"] = "Resolve the reported failure before performance tuning."
        return report
    rows = [r for r in records if r.get("event") == "benchmark"]
    if not rows:
        raise ValueError("passing result has no benchmark records")
    loads = defaultdict(lambda: {"baseline_ns": 0., "candidate_ns": 0.})
    keys = set()
    for row in rows:
        case = row["case"]
        key = case_key(case)
        if key in keys:
            raise ValueError(f"duplicate benchmark case: {key}")
        keys.add(key)
        if row.get("status") != "pass":
            raise ValueError("benchmark records must all pass")
        for value in (row["baseline_ns"], row["median_ns"], case["weight"]):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("benchmark times and weights must be finite and positive")
        load = loads[case["load"]]
        load["baseline_ns"] += row["baseline_ns"] * case["weight"]
        load["candidate_ns"] += row["median_ns"] * case["weight"]
    speedup = geometric_mean(v["baseline_ns"] / v["candidate_ns"] for v in loads.values())
    if not math.isclose(speedup, result["speedup"], rel_tol=1e-6):
        raise ValueError("reconstructed load score does not match reported speedup")
    score_us = result["score_us"]
    if not math.isfinite(score_us) or score_us <= 0:
        raise ValueError("reported score must be finite and positive")
    shape_reports = []
    for row in rows:
        case = row["case"]
        load = loads[case["load"]]
        fraction = row["median_ns"] * case["weight"] / load["candidate_ns"]
        # Conditional payoff: changing one shape changes just one load total.
        hypothetical_score = score_us * (1. - .2 * fraction) ** (1. / len(loads))
        shape_reports.append({
            "case": case,
            "candidate_us": row["median_ns"] / 1000,
            "baseline_us": row["baseline_ns"] / 1000,
            "normalized_latency": row["median_ns"] / row["baseline_ns"],
            "within_load_share": fraction,
            "score_sensitivity": fraction / len(loads),
            "hypothetical_score_us_if_shape_20pct_faster": hypothetical_score,
        })
    report.update({
        "has_timings": True,
        "reconstructed_speedup": speedup,
        "loads": {name: {"baseline_us": v["baseline_ns"] / 1000,
                         "candidate_us": v["candidate_ns"] / 1000,
                         "speedup": v["baseline_ns"] / v["candidate_ns"]}
                  for name, v in loads.items()},
        "shapes_by_sensitivity": sorted(shape_reports, key=lambda r: r["score_sensitivity"], reverse=True),
        "target_score_us": target_us,
        "required_uniform_latency_reduction": 1. - target_us / score_us,
        "target_source": "User-reported 9580 us leaderboard time, or CLI override; no live verification.",
        "caveat": "Sensitivity ranks whole shapes, not kernel stages. Conditional savings are not measured predictions. Profile timings are a separate pass; router and pooler overlap.",
    })
    return report


def compare_reports(current, previous):
    if not current["has_timings"] or not previous["has_timings"]:
        raise ValueError("both results need passing benchmark timings to compare")
    before = {case_key(r["case"]): r for r in previous["shapes_by_sensitivity"]}
    after = {case_key(r["case"]): r for r in current["shapes_by_sensitivity"]}
    if before.keys() != after.keys():
        raise ValueError("comparison requires matching benchmark shapes")
    return [{"case": row["case"],
             "normalized_latency_ratio": row["normalized_latency"] / before[key]["normalized_latency"]}
            for key, row in after.items()]


def read_document(path):
    text = path.read_text()
    if not text.strip():
        raise ValueError(f"{path} is empty; the check may still be running")
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--compare", type=Path, help="previous passing check for per-shape normalized ratios")
    parser.add_argument("--target-us", type=float, default=9580.)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = analyze(read_document(args.result), args.target_us)
        if args.compare:
            report["comparison"] = compare_reports(report, analyze(read_document(args.compare), args.target_us))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Cannot analyze: {error}\n")
    print(f"Job {report['job']}: {report['tests_passed']}/{report['tests_total']} correctness tests passed")
    if report["has_timings"]:
        print(f"Measured {report['score_us']:.2f} us / {report['speedup']:.4f}x; interval {report['speedup_ci']}")
        print(f"Target {report['target_score_us']:.2f} us; uniform latency reduction needed {report['required_uniform_latency_reduction']:.2%}")
        for row in report["shapes_by_sensitivity"]:
            case = row["case"]
            print(f"  {case['load']} {case['layer']}: {row['candidate_us']:.2f} us; sensitivity {row['score_sensitivity']:.2%}")
    else:
        print("No validated timing result. " + report["next_step"])
        if report["failures"]:
            print(report["failures"][0].get("message") or report["failures"][0].get("error") or report["failures"][0])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
