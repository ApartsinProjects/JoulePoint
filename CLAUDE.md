# JoulePoint project rules

## Deliverable propagation (source-of-truth policy)

The **LaTeX two-column PDF (`docs/JoulePoint_2col_latex.pdf`) is the canonical
typeset deliverable**: any change agreed on that version is authoritative.
Mechanically, content still lives in ONE master source, `docs/GPTEnergy.html`
(written by `power_shaping/build_gptenergy.py`), and every format regenerates
from it. Therefore, for ANY change:

1. Land the change in the master: `power_shaping/build_gptenergy.py` (and the
   built `docs/GPTEnergy.html` stays in sync with it).
2. If the change originated as a LaTeX-side front-matter or layout decision
   (e.g. corresponding author, affiliations, back-matter conventions), mirror
   its CONTENT into the HTML master too; layout-only mechanics stay in
   `latex/build_tex.py` / `docs/build_docx.py`.
3. Regenerate EVERY deliverable, never a subset:
   - `python docs/build_docx.py` -> JoulePoint_1col.{docx,pdf} + JoulePoint_2col.{docx,pdf}
   - `python latex/build_tex.py` -> docs/JoulePoint_2col_latex.pdf
4. Verify (render pages, canaries), then commit and push (GitHub Pages serves
   docs/ directly; the badges on the paper page must all resolve).
5. Mirror shared files (`GPTEnergy.html`, `build_gptenergy.py`, deliverables)
   to the private working repo `DataCenterEnergy` in the same session.

A commit that changes the paper without rebuilding all five deliverables is
incomplete.

## Skills

Use the `TwoColPaper` skill for any two-column layout work (it carries the
elsarticle route, the Word packing discipline, and the verification loop),
`paper-build`/`html2doc` for Word/HTML builds, `html2tex` for LaTeX.
