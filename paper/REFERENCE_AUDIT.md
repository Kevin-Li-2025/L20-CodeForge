# Reference Audit

This note records why the paper cites each research line and how the entries
were checked. The bibliography is intentionally selective: it covers the
closest benchmark, verifier, code-RL, verified-data, and feedback-loop work
without padding the paper with loosely related citations.

## Coverage

| Area | References | Why they are included |
| --- | --- | --- |
| Code benchmark hygiene | LiveCodeBench, EvalPlus | Defines the contamination and test-adequacy risks that motivate full-suite hidden replay and EvalPlus guardrails. |
| Test-time scaling | AlphaCode, CodeT, LEVER, S* | Covers sampling, generated tests, execution-aware verifiers, and recent code-specific test-time scaling. |
| Code RL and reward modeling | CodeRL, ACECoder | Covers unit-test/critic feedback, test-case synthesis, reward modeling, and RL for code models. |
| Verified synthetic data | rStar-Coder, HardTests, X-Coder | Covers the closest recent direction for durable model-weight gains from verified tasks, tests, and solutions. |
| Feedback/refinement loops | OpenCodeInterpreter, Reflexion | Covers execution feedback, refinement, and verbal reflection; this anchors the paper's negative repair-loop findings. |
| Base model | Qwen2.5-Coder technical report | Identifies the open 7B model used in the main experiments. |

## Primary Sources Checked

- LiveCodeBench: https://arxiv.org/abs/2403.07974
- EvalPlus: https://arxiv.org/abs/2305.01210
- AlphaCode: https://arxiv.org/abs/2203.07814
- CodeT: https://arxiv.org/abs/2207.10397
- LEVER: https://arxiv.org/abs/2302.08468
- S*: https://arxiv.org/abs/2502.14382
- CodeRL: https://arxiv.org/abs/2207.01780
- ACECoder: https://arxiv.org/abs/2502.01718
- rStar-Coder: https://arxiv.org/abs/2505.21297
- HardTests: https://arxiv.org/abs/2505.24098
- X-Coder: https://arxiv.org/abs/2601.06953
- OpenCodeInterpreter: https://arxiv.org/abs/2402.14658
- Reflexion: https://arxiv.org/abs/2303.11366
- Qwen2.5-Coder: https://qwenlm.github.io/blog/qwen2.5-coder-family/

## Submission Sources

The arXiv and checklist sources are used for packaging and metadata, not cited
as research contributions in the paper body:

- arXiv submission overview: https://info.arxiv.org/help/submit/index.html
- arXiv TeX submission help: https://info.arxiv.org/help/submit_tex.html
- arXiv license help: https://info.arxiv.org/help/license/index.html
- arXiv category taxonomy: https://arxiv.org/category_taxonomy
- NeurIPS paper checklist: https://nips.cc/public/guides/PaperChecklist
