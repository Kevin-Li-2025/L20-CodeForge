# Milestone 10: Failure-Driven Verifier RLVR

Date: 2026-08-29

## Decision

The next model-improvement path is verifier-first algorithmic code RLVR. The
generic SWE-Gym QLoRA campaign remains a negative control: it improved held-out
assistant-token NLL but does not have an executable patch success receipt. The
X-Coder probes also show that formatting and extraction fixes do not resolve the
remaining algorithmic failures on the medium control slice.

The working research question is:

```text
Can model-specific, failure-driven tests improve greedy code capability for a
7B model under a four-consumer-GPU budget, and can the gain be separated from
prompting, multi-sample selection, and test overfitting?
```

This milestone does not claim that RL training has run or that model capability
has improved. It establishes the executable reward and verifier-quality gates
that must pass before a GPU pilot is allowed.

## Why Verifier Quality Comes First

X-Coder and rStar-Coder show that verified synthetic tasks, solutions, and tests
can move 7B code models substantially. HardTests shows that weak tests can make
post-training worse, while adversarial generator-based tests improve verifier
precision and downstream training. RobustTests further targets near-correct
faulty programs and uses their behavioral differences to synthesize and filter
tests before RL.

Sources:

- https://arxiv.org/abs/2601.06953
- https://github.com/microsoft/rStar
- https://arxiv.org/abs/2505.24098
- https://arxiv.org/abs/2608.24135

The local control evidence is consistent with that literature:

- automatic and strict starter-prefix runs scored `0/12` on medium control12;
- one public-feedback repair round raised the combined gate to `4/12`;
- a second feedback round produced a public-only pass and `0/8` hidden passes;
- healthy syntax did not remove the remaining algorithmic failures.

Therefore public examples alone are not promoted into an RL reward. Training
tests must first pass the labeled false-positive/false-negative audit below.

## Implemented Phase-A Surface

`l20_codeforge.rewards.code_execution` now provides a standalone Python
stdin/stdout verifier with:

- deterministic fenced-code extraction;
- a compile gate;
- per-test pass/fail, timeout, runtime-error, and output-limit results;
- separate compile and per-test wall-clock timeouts;
- whitespace-token or normalized-exact comparison;
- CPU, address-space, output-size, and file-descriptor limits;
- process-group termination after wall-clock timeout;
- dense and binary rewards for controlled ablations;
- a behavioral signature such as `PPWT` for clustering faulty candidates.

`l20_codeforge.evals.verifier_audit` audits labeled JSONL before tests may be
used for training. It reports:

- reference-solution acceptance;
- known-correct candidate acceptance and false-negative rate;
- faulty-candidate kill rate and false-positive rate;
- unique faulty behavior count;
- per-test faulty-code kill counts;
- exact input SHA-256 and explicit claim boundaries.

The subprocess resource limits reduce accidental damage but are not a security
sandbox. Large untrusted rollouts still require isolated workers or containers.

## Audit Input Contract

Each JSONL record has this shape:

```json
{
  "task_id": "square",
  "tests": [{"input": "-3\n", "output": "9\n"}],
  "reference_solutions": ["x = int(input())\nprint(x * x)"],
  "candidates": [
    {
      "candidate_id": "near-correct-0",
      "code": "x = int(input())\nprint(x + 2)",
      "expected_correct": false,
      "source": "qwen25-seed42",
      "tags": ["boundary-sign"]
    }
  ]
}
```

Run the committed synthetic smoke:

```bash
python -m l20_codeforge audit-code-verifier \
  examples/verifier_audit.square.jsonl \
  --output artifacts/verifier/square-audit.json \
  --min-reference-solutions 2 \
  --min-faulty-kill-rate 1.0 \
  --max-false-positive-rate 0.0 \
  --max-false-negative-rate 0.0 \
  --fail-on-gates
```

This command validates only the synthetic smoke fixture. It is not data-scale,
GPU-training, LiveCodeBench, or model-improvement evidence.

## Promotion Gates

Before SFT or GRPO, the real 2K-task candidate pool must satisfy:

1. Exact and near-duplicate LiveCodeBench exclusion receipts.
2. Dataset source, license, and content hashes.
3. Two independent reference solutions for admitted tasks.
4. `100%` reference acceptance on admitted tests.
5. At least `70%` faulty-candidate kill rate.
6. At most `5%` false-positive and `5%` false-negative rate on the labeled audit.
7. A held-out verifier split that is not used to generate or repair tests.

These thresholds are engineering promotion gates, not published benchmark
claims. A failed gate blocks GPU training and routes the task back to validator,
oracle, or adversarial-test construction.

## Reward Ablation

The first 100-step pilot compares two rewards on identical prompts and seeds:

```text
dense = 0.10 compile + 0.70 pass_fraction + 0.20 all_pass
        - 0.20 any_timeout - 0.10 any_runtime_or_output_error

binary = 1.0 only when all verifier tests pass, else 0.0
```

Patch-quality and self-reported verification rewards have zero weight for this
algorithmic-code experiment. Reward curves alone are not a promotion signal.
Most sampled groups must have non-zero within-group reward variance, and reward
growth must correlate with held-out executable accuracy.

## Planned GPU Pilot

The planned first allocation is four GPUs:

```text
2 GPUs: QLoRA/LoRA policy training
2 GPUs: dedicated vLLM rollout server
num_generations: 4
beta: 0
loss_type: dr_grpo
max_completion_length: 2048 initially
steps: 100, then 500 only after reward and recovery audit
```

The exact plan is pinned in
`configs/qwen25_coder_7b_failure_driven_rlvr.yaml`. All model, dataset, cache,
execution, and checkpoint paths must live under `/ssd/scxi253`; the cluster root
filesystem is currently full.

## Evaluation Boundary

- Development/model selection: verifier dev, control12 public-only signals, and
  EvalPlus guardrails.
- Sealed final: a new algorithmic final split, never used for test generation.
- Legacy comparison: LiveCodeBench release_v6 greedy `n=1`, once after the
  training recipe is frozen.
- `pass@4` and public-selection results remain separate system metrics.
- No hidden/frozen test is used to choose data, prompts, checkpoints, or reward
  weights.
