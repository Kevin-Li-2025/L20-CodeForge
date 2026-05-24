# arXiv Submission Metadata

This file records the current intended metadata for the L20-CodeForge preprint.
Verify every field in the arXiv submission UI before submission.

## Title

L20-CodeForge: Auditable Test-Time Scaling for Code Generation on a Single GPU

## Authors

Yin Li

University of Birmingham

## Abstract

Use the abstract from `main.tex` verbatim unless the paper body changes.

## Suggested Categories

Primary category:

- `cs.CL` - Computation and Language

Cross-lists:

- `cs.SE` - Software Engineering
- `cs.LG` - Machine Learning

Rationale:

- The paper studies code generation with language models, which fits current
  LLM/code-generation work commonly posted in `cs.CL`.
- The benchmark and execution-harness material is also software-engineering
  work, especially the LiveCodeBench and EvalPlus evaluation boundary.
- The post-training and verifier direction justifies `cs.LG` as a cross-list,
  but the present paper is not primarily a new learning algorithm.

## Comments Field

Suggested text:

```text
Preprint. Code, benchmark artifacts, CI, and reproducibility notes are available at https://github.com/Kevin-Li-2025/L20-CodeForge.
```

Update the page count after the final compile.

## License

Recommended default for an open preprint:

- Creative Commons Attribution 4.0 International (`CC BY 4.0`)

Reason:

- arXiv lists CC BY 4.0 as an available license and encourages authors to
  consider liberal licenses for reuse.
- CC BY 4.0 preserves attribution while allowing broad reuse.

Before submission, check whether any later venue or funder imposes a different
license requirement. arXiv states that license choices are irrevocable for a
posted version.

## Source Package

The arXiv source package should contain only:

```text
main.tex
references.bib
```

Do not include:

```text
main.pdf
*.aux
*.bbl
*.blg
*.log
*.out
*.toc
*.tar.gz
```

Build the source archive locally with:

```bash
cd paper
make arxiv-source
```

The generated `l20-codeforge-arxiv-source.tar.gz` is ignored by git.

## Official References Checked

- https://info.arxiv.org/help/submit/index.html
- https://info.arxiv.org/help/submit_tex.html
- https://info.arxiv.org/help/license/index.html
- https://arxiv.org/category_taxonomy
