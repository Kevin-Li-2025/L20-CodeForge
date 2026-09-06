# Retracted v2 data-feasibility receipt and v2.1 replacement

**Status: `RETRACTED_INVALID_INCOMPLETE_SOURCE`.** This audit consumed only
1,500 source rows because converted parquet shard `0002` was omitted. It is
preserved as an audit trail, but none of its feasibility conclusions or the
v2.1 replacement is authoritative. See `SOURCE_RECONSTRUCTION_CORRECTION.md`.

## v2 fail-closed result

No v2 model training or model evaluation had started when this check ran. The
pinned 1,600-row rStar Dataset Server view was reconstructed from the official
Hugging Face `refs/convert/parquet` files as physical rows 0--999 of `0000`
followed by rows 0--599 of `0001`:

- `0000.parquet`: 398,180,786 bytes; SHA-256
  `9a6741885dc8556c51d377f2b2fb38bb8f587bc149e3f3a62939ee8d946290be`
- `0001.parquet`: 497,224,109 bytes; SHA-256
  `2f94ddd239a3e0974219f963fb2407e4fdfd341e8c977edd92e1094abb6ea7b4`

The v2 splitter preserved 200 new-development tasks but, after checking every
remaining candidate against historical train/dev and earlier holdouts at the
preregistered 5-gram Jaccard threshold `<0.85`, admitted only 145 of the
requested 200 final tasks. It stopped with:

```text
ValueError: insufficient holdout tasks after filtering:
retention=200/200, final=145/200
```

The original v2 protocol is therefore closed as `BLOCKED_DATA_FEASIBILITY`.
No threshold was weakened and no task was added after looking at model output.

## v2.1 preregistration

Frozen before runtime smoke, formal training, or any v2.1 model evaluation.
v2.1 inherits every data, training, development-selection, EvalPlus, and
no-post-hoc-tuning rule in `PREREGISTRATION.md`, with exactly one change:

- the report-only new rStar final split contains 100 tasks instead of 200.

The new rStar development split remains 200 tasks. Overall success still
requires an RLVR seed to improve it by at least 4 exact passes (`+2.0` points),
not regress on all 90 MBPP-validation tasks relative to Base, and meet or exceed
Base on both EvalPlus suites. The 100-task rStar final result remains report-only
and cannot select a seed or trigger a second audit.

If 200 development plus 100 final tasks cannot be frozen with the unchanged
split order and near-duplicate threshold, v2.1 is blocked and no further split
revision is allowed in this campaign.
