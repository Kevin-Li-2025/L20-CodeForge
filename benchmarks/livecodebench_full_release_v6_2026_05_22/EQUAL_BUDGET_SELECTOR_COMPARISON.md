# Equal-budget LiveCodeBench selector comparison

This is a post-hoc comparison over one frozen candidate pool. Hidden tests are used only
to score frozen choices; the hidden oracle is reported only as a non-deployable ceiling.

## Budget and provenance boundary

- Tasks: `60` stratified release-v6 tasks.
- Candidate budget: exactly `4` candidates per task (`240` total).
- Every selector is rescored on the same saved candidate strings and order. Generation is shared, not rerun; this is not a measured equal-total-compute trial.
- Selector report generation hashes verified: `True`.
- Selection overhead is not equal: public tests and behavior tests add CPU sandbox work.
- Recorded selection wall times: public_tests = 24.008 s; behavior_consensus = 152.367 s. These are historical elapsed times, not normalized CPU/GPU costs.

## Results

| Selector | Passed | pass@1 | Success given available solution | Wilson 95% CI |
| --- | ---: | ---: | ---: | ---: |
| `first_candidate` | 19/60 | 0.3167 | 0.8261 | [0.2131, 0.4423] |
| `public_tests` | 21/60 | 0.3500 | 0.9130 | [0.2417, 0.4764] |
| `behavior_consensus` | 21/60 | 0.3500 | 0.9130 | [0.2417, 0.4764] |
| `uniform_random_seed_42` | 17/60 | 0.2833 | n/a | [0.1851, 0.4077] |

Uniform-random analytical expectation: `0.3000`. Across `10000` deterministic seeds, its 95% trial range was `[0.2500, 0.3500]`.

Hidden oracle ceiling: `23/60` (`0.3833`); this is not a deployable selector.

## Paired evidence

- Public vs first: `2` public-only passes and `0` first-only passes; exact paired `p=0.5000`.
- Behavior vs public: `0` behavior-only passes and `0` public-only passes; exact paired `p=1.0000`.

This retrospective candidate-pool comparison is not a fresh held-out benchmark.
The random-seed trial interval describes selector randomness, not uncertainty over new tasks.
Paired tests above must be read alongside effect size, sample size and selection overhead.
