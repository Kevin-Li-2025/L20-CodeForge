# Base -> Verified SFT -> RLVR Campaign (2026-08-29)

## Outcome

**Status: `FAIL_EVALPLUS_NO_REGRESSION`.**

The frozen-development improvement target was reached, but no audited RLVR
candidate passed the full EvalPlus no-regression gate. This package therefore
does not claim a successful final model.

| stage | frozen dev greedy n=1 | delta vs Base | HumanEvalPlus | MBPPPlus |
| --- | ---: | ---: | ---: | ---: |
| Base | 70/200 (35.0%) | - | 137/164 (0.835 reported) | 271/378 (0.717 reported) |
| verified SFT | 74/200 (37.0%) | +2.0 points | 133/164 (0.811) | 272/378 (0.720) |
| RLVR interpolation 0.25 | 74/200 (37.0%) | +2.0 points | 134/164 (0.817) | 272/378 (0.720) |
| RLVR interpolation 0.75 | 75/200 (37.5%) | +2.5 points | 136/164 (0.829) | 270/378 (0.714) |

The 0.25 interpolation was the unique candidate selected by the preregistered
"smallest eligible fraction" rule. It failed HumanEvalPlus by three exact
tasks. A second, explicitly post-failure audit used the highest-development
candidate, interpolation 0.75. It failed both Plus suites by one exact task.
The second audit means EvalPlus was consumed twice; even a pass would have had
weaker evidentiary value than a single untouched final audit.

## Claim boundaries

- Development results use the same frozen 200 tasks, temperature 0, greedy
  `n=1`, and sample index 0. Development data was used for model selection.
- EvalPlus v0.3.1 used HumanEvalPlus v0.1.10 (164 tasks) and MbppPlus v0.2.0
  (378 tasks). Plus pass counts require both base and extra tests to pass.
- EvalPlus extra tests were not used for training, reward construction, scale
  selection, or checkpoint selection. They were used for two final audits.
- The historical benchmark package in this repository used a different
  environment. Only the fresh `base-current` reports in this package are used
  for no-regression comparisons.
- The paired Base-to-RLVR 0.75 development delta is +2.5 points, but its paired
  bootstrap 95% interval is `[-1.5, +6.5]` points and exact McNemar p is
  `0.3593`. This 200-task development result is not statistically conclusive.
- Training completion, nonzero gradients, local tests, and Slurm success are
  engineering evidence; none substitutes for the failed model-quality gate.

## Data and training

- Base model: `Qwen/Qwen2.5-Coder-7B-Instruct`, frozen snapshot
  `c03e6d358207e414f1eca0bb1891e29f1db0e242`.
- Source data: `microsoft/rStar-Coder`, `synthetic_rl_testcase`, CC-BY-4.0.
  The source server was partially converted, so this campaign froze 1,600 rows
  obtained through the rows API rather than claiming a full-dataset snapshot.
- Train: 800 tasks, SHA256
  `6cf24bc6da5bd2111c6b4ba730fa679b009ecd5808507cd7484fd903dbc2ec1e`.
- Dev: 200 tasks, SHA256
  `67c9fb2f5bbba31ae3886f49dbcdc98d7231784c559103e328fa931b3e418bda`.
- LiveCodeBench overlap audit: zero exact or thresholded near matches; maximum
  five-gram Jaccard `0.086486`.
- Base train rollouts: 800 tasks, 3,200 samples, pass@4 414/800, 263 mixed-
  reward tasks.
- Verified SFT: 414 all-tests-passed records, completion-only loss, 100 steps,
  train loss `0.1057876`.
- RLVR: 263 mixed-reward tasks, dense executable reward, 100 GRPO steps on two
  RTX 4090 GPUs, train loss `0.0014296`, epoch `1.5385`. All 100 logged steps
  had nonzero gradient norm.

## Selection history

1. Full RLVR adapter scaling `{0.5, 0.75, 1.0}` produced 68, 69, and 73 dev
   passes. No candidate met 74/200, so EvalPlus was not run.
2. SFT-to-RLVR parameter interpolation `{0.25, 0.5, 0.75}` produced 74, 74,
   and 75 passes. The preregistered smallest-eligible rule selected 0.25.
3. Interpolation 0.25 failed the fresh EvalPlus gate: HumanEvalPlus 134 vs 137
   Base; MBPPPlus 272 vs 271 Base.
4. A post-failure global-retention grid on interpolation 0.75, with scales
   `{0.75, 0.875, 0.9375}`, produced 71, 70, and 70 passes. No candidate was
   audited on EvalPlus.
5. A second and final audit used interpolation 0.75 because it had the highest
   development score. It failed by one exact task on each Plus suite: 136 vs
   137 HumanEvalPlus and 270 vs 271 MBPPPlus.

No additional EvalPlus query should be used to tune this campaign. The next
credible attempt should add unrelated broad-code retention data or a base-model
KL/replay objective, select only on development data, and use a newly sealed
generalization guardrail once.

## Reproducibility

The implementation path is on branch `codex/l20-codeforge-gpu-campaign`.
Campaign implementation through adapter interpolation is commit `5b28b19`.
Key entry points are:

- `scripts/slurm_code_verified_sft.sbatch`
- `scripts/slurm_code_grpo.sbatch`
- `scripts/slurm_code_rlvr_rollout_shard.sbatch`
- `scripts/interpolate_lora_adapters.py`
- `scripts/scale_lora_adapter.py`
- `scripts/select_code_retention_scale.py`
- `scripts/merge_evalplus_shards.py`
- `scripts/slurm_evalplus_official.sbatch`
- `scripts/compare_evalplus_guardrail.py`

Formal Slurm jobs:

- SFT train `1491928`; SFT dev `1491938`
- RLVR train `1491941`
- full-RLVR dev scale grid `1492031`-`1492033`
- SFT-to-RLVR interpolation dev grid `1492063`-`1492065`
- first RLVR EvalPlus generation `1492128`-`1492129`; scoring
  `1492143`-`1492144`
- global-retention dev grid `1492153`-`1492155`
- second RLVR EvalPlus generation `1492282`-`1492283`; scoring
  `1492347`-`1492348`

All formal jobs listed above completed with exit code `0:0`. Machine-readable
reports, hashes, official scorer stdout/stderr, negative controls, and selection
receipts are under `receipts/`. `campaign_summary.json` is the compact status
record.
