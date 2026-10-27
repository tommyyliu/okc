"""CPU checks for refactor equivalence, standalone exports, and score accounting."""

import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from tools.analyze import analyze, compare_reports, read_document
from tools.build import ROOT, SOURCES, build_source, source_manifest


class BuildTests(unittest.TestCase):
    def test_refactor_preserves_fixed_candidate_definitions_and_constants(self):
        checkpoint = json.loads((ROOT / "tests/fixtures/fp8_fix.json").read_text())
        tree = ast.parse(build_source())
        nodes = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        # The original attention implementation is renamed, not retuned. The
        # only new GPU math is the FA3 output norm/gate. Runtime stays intact.
        self.assertEqual(set(nodes), set(checkpoint["definitions"]) |
                         {"_split_attention", "_fa3_attention", "_attention_gate", "_small_expert_finalize",
                          "_decode_nvfp4", "_nvfp4_gate_up", "_nvfp4_down", "_single_token_experts"})
        split = copy.deepcopy(nodes["_split_attention"])
        split.name = "_attention"
        experts = copy.deepcopy(nodes["_experts"])
        guards = [n for n in experts.body if isinstance(n, ast.If) and isinstance(n.test, ast.BoolOp)
                  and any(isinstance(v, ast.Name) and v.id in ("SMALL_EXPERT_FINALIZE", "DIRECT_SINGLE_EXPERTS") for v in n.test.values)]
        self.assertEqual(len(guards), 2)
        for guard in guards:
            experts.body.remove(guard)
        prepare = copy.deepcopy(nodes["prepare"])
        raw_guards = [n for n in prepare.body if isinstance(n, ast.If)
                      and isinstance(n.test, ast.Name) and n.test.id == "DIRECT_SINGLE_EXPERTS"]
        self.assertEqual(len(raw_guards), 2)
        for guard in raw_guards:
            prepare.body.remove(guard)
        versions = dict(nodes, _attention=split, _experts=experts, prepare=prepare)
        definitions = {name: hashlib.sha256(ast.dump(versions[name]).encode()).hexdigest()
                       for name in checkpoint["definitions"]}
        self.assertEqual(definitions, checkpoint["definitions"])
        constants = {n.targets[0].id: ast.literal_eval(n.value)
                     for n in tree.body if isinstance(n, ast.Assign)}
        self.assertEqual(constants.pop("ATTENTION_BACKEND"), "full_fa3")
        self.assertIs(constants.pop("SMALL_EXPERT_FINALIZE"), True)
        self.assertIs(constants.pop("DIRECT_SINGLE_EXPERTS"), False)
        expected = dict(checkpoint["constants"], BLOCK_SIZE_M_CANDIDATES=[8, 16])
        self.assertEqual(json.loads(json.dumps(constants)), expected)

    def test_current_standalone_export(self):
        source = build_source()
        self.assertEqual((ROOT / "submission.py").read_text(), source)
        self.assertTrue(source.startswith("# okc: problem=muse_decode_layer\n"))
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0)
                self.assertNotIn(node.module.split(".")[0], ("decode", "tools", "tests", "reference", "baseline"))
            elif isinstance(node, ast.Import):
                self.assertTrue(all(alias.name.split(".")[0] not in ("decode", "tools", "tests", "reference", "baseline")
                                    for alias in node.names))

    def test_export_manifest_binds_to_exact_source(self):
        source = build_source()
        manifest = source_manifest(source)
        self.assertEqual(manifest["submission_sha256"], hashlib.sha256(source.encode()).hexdigest())
        self.assertEqual(set(manifest["sources"]), set(SOURCES))
        self.assertEqual(manifest["preset"], "full_fa3_m16_finalize")

    def test_presets_differ_only_in_backend_constant(self):
        trees = [ast.parse(build_source(preset)) for preset in ("control", "full_fa3")]
        for tree in trees:
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ATTENTION_BACKEND" for t in node.targets):
                    node.value = ast.Constant("normalized-preset")
        self.assertEqual(ast.dump(trees[0]), ast.dump(trees[1]))
        self.assertEqual(source_manifest(build_source("control"))["preset"], "control")
        with self.assertRaisesRegex(ValueError, "unknown preset"):
            build_source("invalid")


def passing_result(scale=1.):
    rows = []
    # Unequal weights and baseline times exercise the actual per-load formula.
    for load, layer, weight, baseline, candidate in (
        ("decode 1", "sliding", 46, 100., 50.),
        ("decode 1", "full", 16, 200., 150.),
        ("verify 1", "sliding", 46, 200., 100.),
        ("verify 1", "full", 16, 400., 300.),
    ):
        rows.append({"event": "benchmark", "status": "pass", "baseline_ns": baseline,
                     "median_ns": candidate * scale,
                     "case": {"load": load, "layer": layer, "weight": weight,
                              "batch": 1, "q_len": 1 if load == "decode 1" else 8, "seed": 42}})
    speedup = 7800. / (4700. * scale)
    return {"id": "synthetic", "result": {"passed": True, "records": rows,
            "score_us": 1000. * scale, "speedup": speedup, "speedup_ci": None}}


class AnalysisTests(unittest.TestCase):
    def test_failed_run_never_becomes_a_performance_measurement(self):
        result = analyze({"result": {"passed": False, "records": [
            {"event": "test", "status": "fail", "message": "CompilationError"}]}})
        self.assertFalse(result["has_timings"])
        self.assertEqual(result["failures"][0]["message"], "CompilationError")
        self.assertNotIn("shapes_by_sensitivity", result)

    def test_load_score_sensitivity_and_conditional_saving(self):
        report = analyze(passing_result())
        self.assertTrue(report["has_timings"])
        self.assertAlmostEqual(report["reconstructed_speedup"], 7800. / 4700.)
        self.assertAlmostEqual(sum(r["score_sensitivity"] for r in report["shapes_by_sensitivity"]), 1.)
        row = next(r for r in report["shapes_by_sensitivity"] if r["case"]["load"] == "decode 1" and r["case"]["layer"] == "full")
        self.assertAlmostEqual(row["within_load_share"], 2400. / 4700.)
        self.assertAlmostEqual(row["hypothetical_score_us_if_shape_20pct_faster"], 1000. * math.sqrt(1. - .2 * 2400. / 4700.))

    def test_rejects_inconsistent_score_and_duplicate_shapes(self):
        bad = passing_result()
        bad["result"]["speedup"] = 99.
        with self.assertRaisesRegex(ValueError, "reconstructed"):
            analyze(bad)
        duplicate = passing_result()
        duplicate["result"]["records"].append(duplicate["result"]["records"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analyze(duplicate)

    def test_normalized_comparison_rejects_failed_or_different_workloads(self):
        before = analyze(passing_result())
        after = analyze(passing_result(.8))
        self.assertTrue(all(math.isclose(r["normalized_latency_ratio"], .8) for r in compare_reports(after, before)))
        with self.assertRaisesRegex(ValueError, "passing"):
            compare_reports(after, analyze({"result": {"passed": False}}))
        after["shapes_by_sensitivity"][0]["case"]["batch"] = 99
        with self.assertRaisesRegex(ValueError, "matching"):
            compare_reports(after, before)

    def test_empty_result_is_reported_as_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "running.json"
            path.touch()
            with self.assertRaisesRegex(ValueError, "still be running"):
                read_document(path)


if __name__ == "__main__":
    unittest.main()
