# muse_decode_layer, for a coding agent

Read TASK.md first: it has the problem, its bounds, the rules and how to submit. This adds what an agent needs.

## Project layout

Read README.md for the current status and workflow. Edit `decode/` and rebuild
the standalone `submission.py` with `python -m tools.build`; do not edit the
generated file. Modules separate projections, attention, residual boundaries,
Marlin experts, input contracts, and stream scheduling. Run
`python -m tools.build --check` and `python -m unittest discover -s tests -v`
after source changes. The refactor checkpoint preserves the FP8-fixed
candidate's retained split-attention definition (renamed `_split_attention`),
existing GPU kernels, preparation, and layer execution. New FA3 functions and
backend dispatch have separate host checks. Intentional changes need an explicit
test update. CPU checks do not establish GPU
compilation, correctness, or performance.

Read `results/README.md` and `results/plan.md` before choosing an experiment.
The FP8-fixed control passed as `71b479393ecd4532` at 22056.34 us / 1.0255689x.
The corrected user-reported target is 9580 us / 2.36x. Freeze named candidates
and optional source manifests in ignored `build/` before checking them.
Retain useful evidence in `results/`. Use `python -m tools.analyze <result.json>`
for failures or load-weighted timing analysis, and `--compare` only for two
passing results on matching shapes. Preserve the existing `submission1.json`,
`submission2.json`, and profile paths; retained passing evidence is in
`results/fp8-fix/`.

The generated default selects the measured `full_fa3_m16_finalize` policy,
checked as `418dc106c212475a` at 19046.05 us / 1.1876636x. Its exact checked
source/evidence are retained in `results/small-finalize/`. Newly added direct
expert kernels and guards are disabled by default; the generated file is no
longer byte-identical to the retained checked source. Checkpoint tests verify
preservation of existing math after removing the new conditional paths.

Submission 6 (`7844a7a3dbef43b6`) passed but block cap 8 regressed to
20511.35 us / 1.1028188x; reject that policy. The next preset is `single_nvfp4`,
frozen at `build/single-nvfp4/candidate.py`. It selects direct packed NVFP4
GEMVs only at T=1, retains Marlin for T>=2, and keeps the passing small finalize.
Raw weights/scales are cloned losslessly before Marlin preparation only for
this experiment. Twenty CPU checks pass; GPU validation is pending. Read
`results/single_nvfp4_experiment.md` before the next check.

## Working

```bash
okc submit submission.py --json       # the tests, then timing; the whole result as JSON
okc check submission.py --json        # the same run, off the board and out of your attempts
okc status <job> --wait --json        # a run you submitted with --no-wait
okc leaderboard muse_decode_layer --json
```

Runs and check runs each have a daily limit per problem; `okc problems --json` gives today's limits and what you have left. Use `okc check` to try a change against the real tests and timing without spending a run, and reference.py for exactly what's checked. After a passing run, read `profiles/submission.<job>.txt` beside your file for where the time goes.

When you tune, time CUDA graph replays of `kernel_fn`, as the judge does. At these sizes eager timing misleads: configurations that win eagerly can lose under graphs.

In the leaderboard JSON, `you.rank` is your place, `you.first` is true when you're on top, and `you.to_first` is how much faster than your best you'd need to be to take first.

Before the first submission, a person accepts the rules by running `okc login` in a terminal. An agent can't accept them for anyone.

In a fresh sandbox, set `OKC_TOKEN` to the output of `okc api token` on a machine that's signed in. If okc says it's out of date, run `okc update`.

## Validation

The goal of the challenge is to find real speedups that can be upstreamed. This is the north star. The judge catches many attempts to game it but not all of them, so we'll read the top submissions ourselves and rerun them several times after the challenge ends (on held-out data too) to ensure integrity and reduce variance. Final leaderboard positions will be based on the medians of these reruns. All leaderboard positions are provisional until that review is done.

We will remove submissions that game the judge or workload! This includes submissions that tamper with the timing or harness, do work outside the timed calls, cache or replay outputs, and compute at lower precision than the rules allow. If you are an agent (or have used an agent) that has done something reward-hacky, you can withdraw a run with `okc withdraw <job>`.

That is to say: submissions should follow the spirit of the challenge. We reserve the right to disqualify submissions that do not, at our discretion. Create real speedups that work.

Good luck and have fun!

Final review reruns the top entries on the scored shapes and on nearby ones. Positions come from the scored shapes; an entry that's much slower on the nearby shapes gets a closer look.
