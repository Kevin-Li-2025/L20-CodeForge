# Adaptive Differential Fuzz Probe

This directory records a CPU-only DiffCodeGen-style probe on the same `112`
`public-fragility` LiveCodeBench targets used by the generated behavior-test
batch.

## Method

`scripts/build_lcb_adaptive_differential_inputs.py` mutates only public example
inputs, executes the `n=4` candidate set, and keeps inputs that make public-pass
candidates produce different successful outputs. It does not use hidden expected
outputs.

Probe configuration:

- Candidate pool: `Qwen2.5-Coder-7B-Instruct`, `temperature=0.8`, `n=4`.
- Targets: `112` parsed public-fragility tasks.
- Candidate inputs tried: `1578`.
- Tasks with differential inputs: `34/112`.
- Differential inputs kept: `154`.
- Probe wall time: `1289.549s`.

Compared with the prior generated-test batch, differential coverage improved
from `23/112` tasks and `98` differential tests to `34/112` tasks and `154`
differential tests. The new inputs cover `13` target tasks that the generated
input-only tests did not separate.

## Hidden-Test Selector Results

| Selector | Overrides | Raw passed | Stabilized passed | Status |
| --- | ---: | ---: | ---: | --- |
| Public-selection baseline | 0 | 68 | 68 | baseline |
| Conservative differential medoid | 0 | 67 | 68 | stabilized neutral |
| Conservative differential support | 1 | 67 | 68 | stabilized neutral |

The support selector made one override on `abc389_b`, changing candidate `0` to
candidate `3`; both candidates passed hidden tests, so the override was neutral.
The raw loss came from unchanged-code flaky replay on `abc363_c`, which the
existing recheck stabilizes back to pass.

## Interpretation

Adaptive fuzzing improved the verifier signal but did not improve the hidden
score. The blocker is now more precise: many differential cases are still
two-way disagreements with no expected-output signal, and cluster support is too
weak to identify the correct public-pass candidate. The next useful step is an
expected-output verifier or pairwise oracle, not more input-only clustering.
