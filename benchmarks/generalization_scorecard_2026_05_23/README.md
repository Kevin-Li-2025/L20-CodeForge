# Generalization Scorecard

This scorecard is a guardrail against LiveCodeBench-only overfitting. Hidden/private tests are used only for final measurement; selector, repair, and training changes should continue to pass this cross-benchmark gate.

Gate status: `PASS`

## LiveCodeBench Full `release_v6`

| slice | greedy passed | candidate passed | total | greedy | candidate | delta |
| --- | --- | --- | --- | --- | --- | --- |
| overall | 297 | 378 | 1055 | 0.2815 | 0.3583 | +0.0768 |
| difficulty:easy | 206 | 251 | 322 | 0.6398 | 0.7795 | +0.1398 |
| difficulty:hard | 9 | 17 | 350 | 0.0257 | 0.0486 | +0.0229 |
| difficulty:medium | 82 | 110 | 383 | 0.2141 | 0.2872 | +0.0731 |
| platform:atcoder | 146 | 183 | 602 | 0.2425 | 0.3040 | +0.0615 |
| platform:codeforces | 2 | 5 | 9 | 0.2222 | 0.5556 | +0.3333 |
| platform:leetcode | 149 | 190 | 444 | 0.3356 | 0.4279 | +0.0923 |

## EvalPlus Holdout

| dataset | greedy | clean system | greedy plus | system plus | delta |
| --- | --- | --- | --- | --- | --- |
| humaneval | humaneval_greedy | humaneval_clean_system_best | 0.848 | 0.927 | +0.079 |
| mbpp | mbpp_greedy | mbpp_clean_system_best | 0.722 | 0.817 | +0.095 |

## Gate Checks

All checks passed.
