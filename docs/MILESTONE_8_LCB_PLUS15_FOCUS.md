# Milestone 8: LiveCodeBench +15 Focus

## Target

The target is a real greedy model-capability result, not a larger public-test
selection number.

```text
Qwen2.5-Coder-7B-Instruct published reference: 37.6 LiveCodeBench Pass@1
Target delta:                                      +15.0
Target score:                                      52.6+
```

The local full-suite baseline remains:

```text
full release_v6 greedy: 297/1055 = 28.15%
full release_v6 n=8 public selection: 403/1055 = 38.20%
```

The `n=8` result is useful infrastructure signal, but it does not prove the
weights are better. This milestone treats `greedy` or `single sampled solution`
as the headline score.

## Research Signal

The most relevant current route is not generic repository SFT. The strongest
public evidence points to competitive-programming-specialized post-training:

- rStar-Coder builds a 418K verified competitive-programming corpus with long
  reasoning solutions and synthesized tests, reporting a Qwen2.5-7B
  LiveCodeBench improvement to 57.3.
- X-Coder releases public Qwen2.5-Coder-derived SFT and GRPO/RLVR checkpoints.
  Its model cards report a GRPO-trained 7B model reaching 62.9 on
  LiveCodeBench v5.
- HardTests and related verifier papers argue that code RLVR is bottlenecked by
  test quality; better synthesized tests matter much more for self-distillation
  and RL than for plain teacher distillation.
- LiveCodeBench is time-segmented and specifically intended to expose
  contamination and HumanEval-style overfitting, so hidden tests stay reserved
  for measurement only.

Sources:

- https://arxiv.org/abs/2505.21297
- https://huggingface.co/IIGroup/X-Coder-RL-Qwen2.5-7B
- https://huggingface.co/datasets/IIGroup/X-Coder-SFT-376k
- https://huggingface.co/papers/2505.24098
- https://proceedings.iclr.cc/paper_files/paper/2025/hash/94074dd5a072d28ff75a76dabed43767-Abstract-Conference.html

## Execution Plan

1. Run a protocol sanity check with `IIGroup/X-Coder-RL-Qwen2.5-7B` on the same
   local LiveCodeBench `release_v6` harness. This tells us whether the +15
   target is reachable under our exact evaluator, prompt wrapper, truncation,
   and L20 memory constraints.
2. If the sanity check clears the target, use it as a teacher/baseline, not as a
   claimed original training result.
3. Build a local QLoRA route from `IIGroup/X-Coder-SFT-376k` with
   LiveCodeBench exact-prompt exclusion, then evaluate the resulting adapter
   with the same greedy protocol.
4. Add RLVR only after the SFT adapter improves greedy held-out behavior and
   does not regress EvalPlus/MBPP+ guardrails.

## New Infrastructure

`scripts/build_lcb_public_selection.py` creates reusable `public_selection.json`
payloads from saved generations without touching hidden tests. Full hidden
replay can then use the chunked evaluator.

`scripts/build_xcoder_sft_jsonl.py` materializes a capped chat-SFT JSONL from
the public X-Coder dataset. It supports exact LiveCodeBench JSONL exclusion so
we can avoid training on known benchmark prompts.

## Active Probe

The first active L20 probe is:

```text
model: IIGroup/X-Coder-RL-Qwen2.5-7B
scope: release_v6 stratified60
decode: temperature=0.6, top_p=0.95, n=1
max_new_tokens: 8192
purpose: confirm whether a Qwen2.5-Coder-derived RLVR model is above the +15
         target region under this harness before spending GPU time on local SFT.
```
