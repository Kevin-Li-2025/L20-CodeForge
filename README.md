# l20-codeforge

Resource-constrained coding-agent RL and post-training stack for a single NVIDIA L20.

The design target is not broad pretraining. It is a compact loop that turns limited
GPU time into measurable software-engineering capability:

1. Build executable repo-level evaluation first.
2. Collect and normalize agent trajectories.
3. Train small adapters with soft-verified SFT and preference data.
4. Run guided GRPO/RLVR only on short, high-signal tasks.
5. Use self-verification and verifier-guided candidate selection at inference time.

## Why This Shape

The stack follows current evidence from:

- TRL GRPO custom reward support:
  https://huggingface.co/docs/trl/grpo_trainer
- mini-SWE-agent's simple bash-only trajectory shape:
  https://github.com/SWE-agent/mini-swe-agent
- SWE-bench's Docker-based executable evaluation:
  https://www.swebench.com/SWE-bench/installation/
- Qwen2.5-Coder model family and long-context code capabilities:
  https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
- Unsloth VRAM guidance for QLoRA/LoRA:
  https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements
- vLLM's recommendation for fresh uv-managed environments:
  https://docs.vllm.ai/en/latest/getting_started/installation/gpu/

## Repository Layout

```text
configs/                 L20-first experiment configs
docs/                    architecture and runbooks
scripts/                 setup, smoke, and model utility scripts
src/l20_codeforge/
  context/               repo context packing and omission policies
  data/                  task and trajectory schemas
  envs/                  local repo execution adapters
  evals/                 eval cards and reporting
  gpu/                   L20 profile and memory policy
  inference/             candidate verification and selection
  rewards/               executable and patch-quality rewards
  training/              TRL-compatible reward functions and config helpers
tests/                   lightweight unit tests
```

## First Remote Setup

On the L20 host:

```bash
cd ~/l20-codeforge
bash scripts/bootstrap_remote.sh
source .venv/bin/activate
python scripts/check_gpu.py
pytest -q
```

The bootstrap script installs uv if missing, creates a Python 3.12 venv, installs a
CUDA 12.4 PyTorch wheel, then installs this package with training and dev
dependencies. It does not download large model weights.

## Model Strategy

Primary training target:

- `Qwen/Qwen2.5-Coder-7B-Instruct`

L20 policy:

- Use QLoRA for 7B/14B work.
- Use short-context GRPO first: 2K-4K completion budgets, small groups.
- Keep vLLM in a separate environment if needed, because vLLM pins binary stacks.
- Keep all generated checkpoints under `artifacts/`, never under the repo root.

## Smoke Gates

Before real training, these must pass:

```bash
python scripts/check_gpu.py
python -m l20_codeforge profile
pytest -q
```

After that, run a toy reward loop before loading a large model:

```bash
python -m l20_codeforge init-dirs
python -m l20_codeforge eval-card smoke pass
python -m l20_codeforge smoke-loop
```

`smoke-loop` creates 36 executable repo-repair tasks with visible and hidden
tests, evaluates known-good patches in isolated worktrees, writes trajectories,
builds a report, and builds chat SFT JSONL:

```text
data/raw/smoke_tasks/
artifacts/trajectories/smoke_reference.jsonl
artifacts/reports/smoke_reference_report.json
data/processed/smoke_sft.jsonl
```

This is the first quality gate. Do not start GPU training until this local loop
passes on the target machine.

Milestone notes:

- `docs/MILESTONE_1_DATA_FACTORY.md`: first executable data loop.
- `docs/MILESTONE_2_REPAIR_SUITE.md`: 36-task visible/hidden repair suite.
- `docs/MILESTONE_3_MINI_SWE_ADAPTER.md`: mini-SWE-agent trajectory adapter and DPO pairs.
- `docs/MILESTONE_4_REAL_DATA.md`: real SWE dataset registry and fetch path.
- `benchmarks/evalplus_l20_codeforge_2026_05_22/`: reproducibility package
  for the EvalPlus HumanEval+/MBPP+ benchmark sprint, including selected
  samples, official reports, rechecks, hashes, and a summary table.
- `benchmarks/livecodebench_full_release_v6_2026_05_22/`: full 1,055-task
  LiveCodeBench `release_v6` greedy baseline with saved generations, hidden-test
  evaluator outputs, hashes, and breakdowns. It also includes the `n=4`
  public-test selection result and a candidate-aware behavior-test prompt bank
  for the next verifier loop.

LiveCodeBench verifier loop:

```bash
python scripts/build_lcb_behavior_test_prompts.py \
  --lcb-repo /tmp/LiveCodeBench \
  --prompt-public-parquet data/raw/livecodebench/full_release_v6/release_v6_test_prompt_public_only.parquet \
  --generations benchmarks/livecodebench_full_release_v6_2026_05_22/qwen25_coder_7b_temp08_n4_full_generate_only/generations.json \
  --public-selection benchmarks/livecodebench_full_release_v6_2026_05_22/qwen25_coder_7b_temp08_n4_public_select_full_eval/public_selection.json \
  --output-dir benchmarks/livecodebench_full_release_v6_2026_05_22/qwen25_coder_7b_temp08_n4_candidate_aware_behavior_prompts64 \
  --limit 64 \
  --target-priority public-fragility \
  --max-samples 4
```

After a local model fills the generated-test prompts, parse them to
`behavior_inputs.json` and pass that file to `scripts/evaluate_lcb_generations.py`
with `--behavior-inputs` and `--behavior-public-scores`.

For a local Transformers generation pass:

```bash
python scripts/generate_lcb_behavior_tests.py \
  --prompts benchmarks/livecodebench_full_release_v6_2026_05_22/qwen25_coder_7b_temp08_n4_candidate_aware_behavior_prompts64/prompts.jsonl \
  --output-dir artifacts/lcb_behavior_tests/qwen25_coder_7b_prompts64 \
  --model /path/to/Qwen2.5-Coder-7B-Instruct \
  --limit 64 \
  --resume
```

The generated `behavior_inputs.json` is then used only as extra public-side
candidate discrimination; hidden tests remain reserved for final measurement.

Current generated-test probe:

- `benchmarks/livecodebench_full_release_v6_2026_05_22/qwen25_coder_7b_temp08_n4_candidate_aware_behavior_tests64/`
  records the first 64-prompt local-model pass.
- It parsed `54` usable task records with `558` generated behavior inputs.
- On those 54 targets, generated behavior consensus improved selection from
  `51/54` to `52/54`.
- A stricter `conservative-public-pass` selector with consensus margin `1`
  keeps the same `52/54` result while requiring public-test pass and strong
  generated-test agreement before overriding the public-selected candidate.
- The full-suite headline score remains `378/1055` until the full evaluator has
  retry-stabilized replay for flaky unchanged selections.
- Post-hoc recheck of the two unchanged-code full hybrid flips gives a
  stabilized audit count of `379/1055`; this stays separate from the headline
  score until retry logic is built into the evaluator path.
- `target_priority_analysis_2026_05_23` adds a public-signal-only targeter for
  the next behavior-test batch. At the same 64-prompt budget, the new
  `public-fragility` ordering raises retrospective public-pass/hidden-fail
  target density from `3/64` to `20/64` without using hidden labels for prompt
  construction or candidate selection.
- `qwen25_coder_7b_temp08_n4_candidate_aware_behavior_tests128_fragility`
  records the first expanded `public-fragility` generation run: `128` prompt
  records, `112` parsed non-empty behavior-test records, and `1130` generated
  input-only behavior tests in `1227.057s`.
- The targeted 112-task conservative replay was a stabilized-neutral result:
  public-selection baseline `68/112`, raw behavior replay `66/112` due to two
  known unchanged-code flaky tasks, and rechecked audit `68/112`. The selector
  made four public-pass overrides but did not change hidden outcomes, so the
  full-suite headline remains `378/1055`.
- A stricter `conservative-differential-medoid` selector was added for the same
  112-target batch. It made `0` overrides and stabilized to `68/112`, exposing
  the real bottleneck: `89/112` targets had no valid generated behavior tests
  that differentiated public-passing candidates. The next verifier step should
  synthesize adaptive pairwise distinguishing inputs or add expected-output
  verification before another full merged replay.

Generalization gate:

```bash
python scripts/build_generalization_scorecard.py \
  --output-dir benchmarks/generalization_scorecard_2026_05_23
```

This builds a cross-benchmark scorecard over full LiveCodeBench `release_v6`
and EvalPlus HumanEval+/MBPP+. Any future selector, repair, SFT, or RLVR change
should improve the target benchmark without failing this gate.

Full hidden-test materialization:

```bash
python scripts/materialize_lcb_release_jsonl.py \
  --release-version release_v6 \
  --output-jsonl data/raw/livecodebench/full_release_v6/release_v6_test_full.jsonl \
  --manifest data/raw/livecodebench/full_release_v6/release_v6_test_full.manifest.json
```

The generated full JSONL contains private tests and is intentionally not
committed. Commit hashes, manifests, evaluator outputs, and compact summaries
only.

Agent trajectory bridge:

```bash
python -m l20_codeforge export-mini-tasks
python -m l20_codeforge convert-mini data/raw/smoke_tasks/<task>/task.json artifacts/mini_swe/trajectories/<task>.traj.json
python -m l20_codeforge build-dpo artifacts/trajectories/mini_swe_converted.jsonl
```

Real data entry:

```bash
python -m l20_codeforge list-real-sources
python -m l20_codeforge fetch-real-tasks swe-bench-lite --output data/raw/real/swe_bench_lite_sample.jsonl --limit 25
python -m l20_codeforge build-real-sft data/raw/real/swe_bench_lite_sample.jsonl --output data/processed/real_sft/swe_bench_lite_sample_sft.jsonl
```
