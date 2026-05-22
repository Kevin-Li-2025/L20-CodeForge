# EvalPlus Reproducibility Package

This package contains the selected sample files, official EvalPlus report JSONs,
a machine-readable manifest with SHA256 hashes, and a CSV summary for the
L20-CodeForge public benchmark sprint.

Recorded benchmark-result source commit:
`a80c6937e610356bf6f1019104c94880b4733937`.

## Protocol Boundary

- Greedy rows are model baselines.
- Clean system rows may use public base tests, public prompt/base inputs,
  multi-sample generation, targeted resampling, and deterministic selection.
- Clean system rows do not use EvalPlus extra tests for candidate selection.
- The symbolic ablation row is transparent and task-specific; do not present it
  as a general model or clean system score.

## Summary

| name | dataset | protocol | base pass@1 | plus pass@1 | base pass@10 | plus pass@10 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| humaneval_greedy | humaneval | greedy_model | 0.890 | 0.848 |  |  |
| humaneval_n10_sampling | humaneval | sampled_pass_at_k | 0.851 | 0.812 | 0.951 | 0.921 |
| humaneval_clean_system_best | humaneval | clean_public_signal_system | 0.982 | 0.927 |  |  |
| humaneval_symbolic_ablation | humaneval | task_specific_symbolic_ablation | 0.994 | 0.933 |  |  |
| mbpp_greedy | mbpp | greedy_model | 0.828 | 0.722 |  |  |
| mbpp_clean_system_best | mbpp | clean_public_signal_system | 0.960 | 0.817 |  |  |
| mbpp_n5_selector | mbpp | clean_public_signal_system_ablation | 0.918 | 0.757 |  |  |
| mbpp_consensus_shortest | mbpp | clean_public_signal_system_ablation | 0.918 | 0.791 |  |  |

## Re-run Official Scoring

From the repository root on a machine with `l20_codeforge[bench]` installed:

```bash
python -m l20_codeforge eval-evalplus mbpp \
  benchmarks/evalplus_l20_codeforge_2026_05_22/samples/mbpp.temp08.n5-plus-basefallback-n30.public-consensus-shortest-selected.samples.jsonl \
  --output /tmp/mbpp_recheck.json \
  --parallel 8

python -m l20_codeforge eval-evalplus humaneval \
  benchmarks/evalplus_l20_codeforge_2026_05_22/samples/humaneval.mixed-target.literal-combined.public-consensus-selected.samples.jsonl \
  --output /tmp/humaneval_recheck.json \
  --parallel 8
```

Check `manifest.json` before rerunning to verify file hashes.

## Completed Rechecks

The package samples were re-evaluated with official EvalPlus after packaging:

```text
mbpp_clean_system_best:      base pass@1 0.960, plus pass@1 0.817
humaneval_clean_system_best: base pass@1 0.982, plus pass@1 0.927
```

Recheck reports are under `rechecks/`, with hashes in
`rechecks/manifest.json`.
