# Generalization Scorecard

This scorecard is a guardrail against LiveCodeBench-only overfitting. Hidden/private tests are used only for final measurement; selector, repair, and training changes should continue to pass this cross-benchmark gate.

Gate status: `PASS`

## LiveCodeBench Full `release_v6`

| slice | greedy passed | candidate passed | total | greedy | candidate | delta |
| --- | --- | --- | --- | --- | --- | --- |
| overall | 297 | 403 | 1055 | 0.2815 | 0.3820 | +0.1005 |
| difficulty:easy | 206 | 260 | 322 | 0.6398 | 0.8075 | +0.1677 |
| difficulty:hard | 9 | 19 | 350 | 0.0257 | 0.0543 | +0.0286 |
| difficulty:medium | 82 | 124 | 383 | 0.2141 | 0.3238 | +0.1097 |
| platform:atcoder | 146 | 199 | 602 | 0.2425 | 0.3306 | +0.0880 |
| platform:codeforces | 2 | 5 | 9 | 0.2222 | 0.5556 | +0.3333 |
| platform:leetcode | 149 | 199 | 444 | 0.3356 | 0.4482 | +0.1126 |

## EvalPlus Holdout

| dataset | greedy | clean system | greedy plus | system plus | delta |
| --- | --- | --- | --- | --- | --- |
| humaneval | humaneval_greedy | humaneval_clean_system_best | 0.848 | 0.927 | +0.079 |
| mbpp | mbpp_greedy | mbpp_clean_system_best | 0.722 | 0.817 | +0.095 |

## Gate Checks

All checks passed.
