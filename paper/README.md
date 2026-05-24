# Paper Draft

This directory contains the LaTeX draft for an arXiv-style paper about
L20-CodeForge.

The draft is intentionally written as a measured research report. It claims
system-level gains, not a new model checkpoint.

## Build

With Tectonic:

```bash
cd paper
tectonic main.tex
```

With a standard TeX installation:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## arXiv Notes

Before uploading source to arXiv:

- Submit TeX source, not a PDF generated from TeX.
- Do not include generated files such as `.aux`, `.log`, `.out`, `.pdf`, or
  `.toc`.
- Keep file names portable and case-consistent.
- Use PDF/PNG/JPG figures when compiling with PDFLaTeX.
- Verify title, abstract, authors, license, and category metadata in the arXiv
  submission UI.

Primary arXiv guidance:

- https://info.arxiv.org/help/submit/index.html
- https://info.arxiv.org/help/submit_tex.html
- https://info.arxiv.org/help/license/index.html
