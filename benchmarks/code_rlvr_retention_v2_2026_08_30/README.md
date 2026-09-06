# Retention-aware verified SFT + RLVR v2 (2026-08-30)

## Outcome

**Status: `FAIL_EVALPLUS_NO_REGRESSION`.**

The preregistered development target was reached: selected RLVR seed42 gained
four exact passes (`+2.0` points) over Base on the newly frozen rStar dev set
and exactly matched Base on the 90-task MBPP-validation retention set. It then
failed both EvalPlus no-regression guardrails, so this package does not claim a
successful retained model.

| stage | new rStar dev | delta vs Base | MBPP validation | delta vs Base |
| --- | ---: | ---: | ---: | ---: |
| Base | 76/200 (38.0%) | - | 64/90 (71.1%) | - |
| SFT seed42 | 82/200 (41.0%) | +3.0 points | 63/90 (70.0%) | -1 task |
| RLVR seed42 | 80/200 (40.0%) | +2.0 points | 64/90 (71.1%) | 0 tasks |
| SFT seed43 | 75/200 (37.5%) | -0.5 points | 62/90 (68.9%) | -2 tasks |
| RLVR seed43 | 76/200 (38.0%) | 0.0 points | 62/90 (68.9%) | -2 tasks |
| SFT seed44 | 81/200 (40.5%) | +2.5 points | 60/90 (66.7%) | -4 tasks |
| RLVR seed44 | 81/200 (40.5%) | +2.5 points | 61/90 (67.8%) | -3 tasks |

The frozen selector chose seed42. Seed44 had the highest RLVR target score but
was correctly rejected for retention regression. For seed42, RLVR gave back two
rStar passes relative to SFT while recovering the one lost MBPP-validation task.
This is an observed retention tradeoff, not a causal replay effect: no matched
replay-off ablation was run. It does not show that RLVR improved on SFT overall.

The selected Base-to-RLVR dev comparison has 20 gains and 16 losses. Its paired
bootstrap 95% interval is `[-4.0, +8.0]` points and exact McNemar p is `0.6177`,
so the observed `+2.0` points are not statistically conclusive.

## Final guardrails

| guardrail | Base | selected RLVR seed42 | delta | result |
| --- | ---: | ---: | ---: | --- |
| HumanEvalPlus | 137/164 | 110/164 | -27 | fail |
| MBPPPlus | 271/378 | 257/378 | -14 | fail |
| new rStar final | not queried | 72/200 (36.0%) | n/a | report only |

The rStar final split was never used for selection. A Base comparator was not
run because the protocol allowed only the selected adapter to touch this final
split; therefore 72/200 is a report-only score, not a measured final delta.

The date-held-out LiveCodeBench run (`2025-02-01` through `2025-04-30`) could
not be completed. The prior 4.2 GB materialized hidden-test JSONL was not present
on current GPU storage and the ParaCloud host could not reach the official
dataset endpoint. The saved prompt-only and old evaluation artifacts cannot
grade new generations, so no public-test proxy is mislabeled as hidden-test
evidence.

## Failure diagnosis

The single selected EvalPlus audit was not followed by tuning, reselection, or
a second generation run.

- HumanEvalPlus had 2 gained and 29 lost tasks. Nineteen of the 29 losses were
  compile-invalid bare function-body completions under the completion-style
  prompt; selected outputs had 22 such invalid samples overall. Ten additional
  lost tasks compiled and failed semantically. Exact paired McNemar p is
  `4.63e-7`.
- MBPPPlus had 26 gained and 40 lost tasks. All 378 selected samples compiled,
  so its net 14-task regression is semantic. Equal aggregate performance on the
  90-task selector masked substantial task-level churn on the larger Plus set.

Under `0.01` MBPP replay CE plus `beta=0.01` SFT-KL, one small retention-set
task was recovered, but broad function-completion behavior was not preserved.
The HumanEval format failure motivates checking completion-style output
contracts in a new development-only experiment. Neither replay strength nor
SFT mixture coverage is isolated causally by this run.

## Data and training receipts

- Base: `Qwen/Qwen2.5-Coder-7B-Instruct`, snapshot
  `c03e6d358207e414f1eca0bb1891e29f1db0e242`.
- SFT mixture: 414 verified rStar trajectories, 138 historical full-harness
  LiveCodeBench trajectories, and 138 official MBPP-train references (690
  records, deterministic 60/20/20 mixture).
- Three independent QLoRA SFT runs: 150 steps on two RTX 4090 GPUs.
- Three matching RLVR runs: 100 Dr. GRPO steps, four sampled completions,
  executable reward, `beta=0.01`, and MBPP-train replay CE weight `0.01` on two
  RTX 4090 GPUs.
- All 300 formal RLVR log rows were finite and had nonzero gradient norm. Each
  run produced 100 completion receipts and a complete final adapter. This is
  engineering evidence only.
- Implementation through the runtime compatibility fix is commit `c81f95b` on
  branch `codex/l20-retention-aware-rlvr-v2`.

Machine-readable receipts are grouped under `receipts/`: frozen data manifests,
per-seed train reports, training-health ranges, all Base/SFT/RLVR dev and
retention reports, the executable selection report, final rStar report, official
EvalPlus reports, paired guardrail comparison, and post-audit failure diagnosis.

## Claim boundary

Run `python scripts/verify_rlvr_v2_receipts.py` from the repository root to
cross-check the published development counts, linked output hashes, selected
seed, and failed guardrails. CI runs the same check. A receipt-consistency
`PASS` means the aggregate reports agree, not that remote raw generations
were replayed or the model-quality target passed.

EvalPlus had already been consumed by the 2026-08-29 campaign, so it is a
repeated guardrail rather than a previously untouched test. This v2 campaign
used it once, only after seed42 passed development selection. Training success,
finite gradients, and the development threshold do not override the failed
general-code guardrail.
