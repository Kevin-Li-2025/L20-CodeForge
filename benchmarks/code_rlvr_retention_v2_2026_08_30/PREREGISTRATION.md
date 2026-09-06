# Retention-aware verified SFT + RLVR v2 preregistration

**Status: `PREREGISTERED_BEFORE_RUNTIME_SMOKE_OR_FORMAL_TRAINING`.** An initial
data-feasibility audit accidentally read 1,500 instead of 1,600 source rows and
was retracted before model training or evaluation. See
`SOURCE_RECONSTRUCTION_CORRECTION.md`; the original 200-development + 200-final
v2 contract is feasible and remains authoritative.

Frozen before formal model training or any v2 development evaluation.

## Objective

Compare the frozen Qwen2.5-Coder-7B-Instruct Base model, retention-aware verified
SFT, and replay-regularized RLVR at greedy `n=1`. A successful RLVR candidate
must improve a newly frozen 200-task rStar development split by at least four
exact passes (`+2.0` absolute points) relative to Base and must not regress on
the official 90-task MBPP validation split. Only then may it be evaluated on
the final guardrails.

## Frozen data contract

- rStar source: `microsoft/rStar-Coder`, config `synthetic_rl_testcase`, 1,600
  Dataset Server rows, split seed `20260829`, tests `8..24`.
- Historical train 800 and dev 200 assignments must retain SHA-256
  `6cf24bc6da5bd2111c6b4ba730fa679b009ecd5808507cd7484fd903dbc2ec1e`
  and `67c9fb2f5bbba31ae3886f49dbcdc98d7231784c559103e328fa931b3e418bda`.
- The remaining admitted rows are scanned in the already frozen order. The
  first 200 passing the 5-gram Jaccard `<0.85` disjointness check become the
  new development split; the next 200 become the new final split. All four
  rStar splits must be pairwise task-disjoint.
- Target SFT source: all 414 previously verified Base trajectories; each
  admitted program passed every frozen executable testcase.
- Historical L20 replay: 138 pre-existing system-selected LiveCodeBench
  trajectories dated no later than `2024-08-31T23:59:59`, each passing the
  full LiveCodeBench harness. These records are training data and cannot be
  benchmark evidence for overlapping dates.
- MBPP replay: official task IDs 601--974 only, pinned to Google Research commit
  `041338718b4e8151372fd63677104c65b73a0a4e` and raw SHA-256
  `ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f`.
- MBPP retention development: official IDs 511--600. IDs 11--510, MBPP test,
  MBPP+, and HumanEval+ are prohibited from optimization and checkpoint
  selection.
- SFT mixture: exactly 414 target + 138 historical LCB + 138 MBPP train records
  (`60/20/20`, 690 total), deterministically ordered with seed `20260830`.

If 200 new development plus 200 final rStar tasks cannot be frozen under this
contract, the formal campaign is blocked; split sizes or thresholds are not
changed after observing any model output.

## Training contract

Three independent seed-matched runs use seeds `{42, 43, 44}`.

Verified SFT uses two RTX 4090 GPUs, QLoRA, completion-only loss, max length
3,072, 150 steps, learning rate `1e-4`, per-device batch 1, and gradient
accumulation 4. RLVR starts from the matching SFT adapter and uses two RTX 4090
GPUs, 100 GRPO steps, learning rate `1e-6`, four generations, temperature
`0.7`, dense executable reward, `dr_grpo`, KL coefficient `0.01` to the frozen
SFT reference adapter, and MBPP-train replay CE weight `0.01` at max length
3,072.

Runtime smoke is limited to two SFT and two GRPO steps on at most eight target
records. Its outputs are engineering evidence only and cannot change the
frozen hyperparameters based on model quality. Runtime fixes are allowed only
for crashes, tensor-shape errors, non-finite loss, or receipt correctness, and
must be documented before restarting formal runs.

## Development evaluation and selection

Base, every SFT seed, and every RLVR seed are evaluated at temperature 0,
greedy `n=1`, on exactly the new 200-task rStar development split and the 90
official MBPP-validation tasks. No interpolation, adapter scaling, checkpoint
grid, extra seed, or post-result hyperparameter change is allowed.

The executable selector `scripts/select_retention_aware_rlvr_v2.py` admits an
RLVR seed only when:

1. new rStar development passes are at least `Base + 4`; and
2. MBPP-validation passes are at least `Base`.

Among eligible RLVR seeds, select maximum rStar passes, then maximum MBPP
passes, then lexically smallest seed name. If no seed is eligible, the campaign
ends as a negative result without querying final guardrails.

## Final report-only guardrails

The single selected RLVR adapter, if any, is evaluated once on:

- the new 200-task rStar final split;
- LiveCodeBench tasks dated `2025-02-01` through `2025-04-30`, which are outside
  the LCB replay window; and
- EvalPlus v0.3.1 HumanEvalPlus v0.1.10 and MbppPlus v0.2.0.

EvalPlus is a strict no-regression gate against the already frozen current-Base
counts: HumanEvalPlus `137/164` and MBPPPlus `271/378`. Because this repository
has already consumed these suites in the 2026-08-29 campaign, this is a repeated
guardrail, not a previously untouched test. EvalPlus and the final splits are
report-only: no tuning, reselection, interpolation, or second audit follows.

## Success and claim boundary

Overall success requires one RLVR seed to pass the two development gates and
both EvalPlus suites to meet or exceed the frozen Base counts. The new rStar
final and date-held-out LiveCodeBench results are reported with paired deltas
where possible but have no post-hoc acceptance threshold. Training completion,
finite gradients, smoke outputs, and Slurm completion are engineering receipts,
not model-improvement evidence.
