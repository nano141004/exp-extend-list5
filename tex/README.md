# ACL LaTeX Paper Template

This folder is trimmed for writing an ACL-format paper in Overleaf.

## Files to upload

- `latex/acl_latex.tex` - main paper file
- `latex/sections/` - modular section files included by the main paper
- `latex/acl.sty` - official ACL style file
- `latex/acl_natbib.bst` - ACL bibliography style
- `latex/custom.bib` - your BibTeX entries

Keep `acl.sty` and `acl_natbib.bst` unchanged to comply with the ACL format.

## Overleaf setup

1. Upload the `latex` folder to Overleaf.
2. Set `acl_latex.tex` as the main file.
3. Compile with pdfLaTeX.
4. Use `\usepackage[review]{acl}` for anonymous review submissions.
5. Use `\usepackage{acl}` for final/camera-ready submissions.

## Paper structure

The main file uses `\input{sections/...}` so each part can be edited separately:

- `00_abstract.tex`
- `01_introduction.tex`
- `02_related_work.tex`
- `03_method.tex`
- `04_experimental_setup.tex`
- `05_results.tex`
- `06_analysis.tex`
- `07_conclusion.tex`
- `08_limitations.tex`
- `09_ethics_statement.tex`
- `10_acknowledgments.tex`
- `appendix.tex`

Remove or rename sections only if the target venue instructions allow it.
