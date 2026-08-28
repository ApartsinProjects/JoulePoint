# -*- coding: utf-8 -*-
"""Build both Word/PDF deliverables from docs/GPTEnergy.html via the html2doc skill.

  GPTEnergy_1col.docx/pdf  -- PRIMARY: single-column, horizontal multi-panel figures,
                              figures height-capped so figure+caption bumps less (less white).
  GPTEnergy_2col.docx/pdf  -- best-effort: two-column, multi-panel figures swapped to their
                              VERTICAL stacked variants so they flow single-column (no float gaps).

Run from docs/:  python build_docx.py
Word (win32com) is used to render PDFs; on non-Windows, the .docx are still produced.
"""
import os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = r"C:\Users\apart\.claude\skills\html2doc"
PY = sys.executable
NODE_ENV = {**os.environ, "NODE_PATH": os.path.join(SKILL, "node_modules")}


def sh(cmd, env=None):
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=HERE, env=env, check=True)


def stage1(src, out):
    sh(["node", os.path.join(SKILL, "scripts", "katex_to_mathml.js"), "--input", src, "--output", out], env=NODE_ENV)


def stage2(src, out, profile):
    sh([PY, os.path.join(SKILL, "scripts", "convert_to_docx.py"), "--input", src, "--output", out, "--profile", profile])


def stage3(src, out, profile, extra):
    sh([PY, os.path.join(SKILL, "scripts", "apply_academic_style.py"), "--input", src, "--output", out,
        "--profile", profile, "--font-family", "Georgia"] + extra)


def render_pdf(docx, pdf):
    try:
        import win32com.client as w, pythoncom
        pythoncom.CoInitialize()
        app = w.DispatchEx("Word.Application"); app.Visible = False
        d = app.Documents.Open(os.path.join(HERE, docx), False, True)
        d.ExportAsFixedFormat(os.path.join(HERE, pdf), 17); d.Close(False); app.Quit()
        print("  rendered", pdf)
    except Exception as e:
        print("  [skip PDF render]", type(e).__name__, str(e)[:80])


def main():
    # ---- 1-column (primary): horizontal figures, capped height to reduce page-bottom white ----
    stage1("GPTEnergy.html", "_gpte_1col_mathml.html")
    stage2("_gpte_1col_mathml.html", "GPTEnergy_1col_conv.docx", "camera-ready-generic")
    stage3("GPTEnergy_1col_conv.docx", "GPTEnergy_1col.docx", "camera-ready-generic",
           ["--figure-max-height-in", "3.5"])
    render_pdf("GPTEnergy_1col.docx", "GPTEnergy_1col.pdf")

    # ---- 2-column (best-effort): swap multi-panel figures to their stacked variants ----
    html = open(os.path.join(HERE, "GPTEnergy.html"), encoding="utf-8").read()
    html = html.replace("fig_law_fit.png", "fig_law_fit_stacked.png").replace("fig_frontier.png", "fig_frontier_stacked.png")
    open(os.path.join(HERE, "GPTEnergy_2col_src.html"), "w", encoding="utf-8").write(html)
    stage1("GPTEnergy_2col_src.html", "_gpte_2col_mathml.html")
    stage2("_gpte_2col_mathml.html", "GPTEnergy_2col_conv.docx", "two-column")
    stage3("GPTEnergy_2col_conv.docx", "GPTEnergy_2col.docx", "two-column",
           ["--max-span-height-frac", "0.30"])
    render_pdf("GPTEnergy_2col.docx", "GPTEnergy_2col.pdf")

    # content canary
    from docx import Document
    import zipfile
    for f in ("GPTEnergy_1col.docx", "GPTEnergy_2col.docx"):
        z = zipfile.ZipFile(os.path.join(HERE, f))
        media = len([n for n in z.namelist() if n.startswith("word/media/")])
        print(f"  {f}: {media} figures, {len(Document(os.path.join(HERE, f)).tables)} tables")
        assert media == 6, f"{f}: expected 6 figures, got {media}"
    print("done: 1col (primary) + 2col built, 6/6 figures each.")


if __name__ == "__main__":
    main()
