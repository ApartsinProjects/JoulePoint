# -*- coding: utf-8 -*-
"""Build the elsarticle two-column LaTeX PDF from docs/GPTEnergy.html.

Runs the html2tex pipeline (convert -> pack elsarticle) and then applies the
paper-specific front-matter grafting on main.tex: the abstract moves from the
inlined body into elsarticle's frontmatter, the web badges and byline are
dropped, and the real author/affiliation block and journal name replace the
anonymous placeholders.  Output: latex/main.pdf -> docs/JoulePoint_2col_latex.pdf.
"""
import os, re, subprocess, sys, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SKILL = r"C:\Users\apart\.claude\skills\html2tex"
PY = sys.executable


def sh(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    sh([PY, os.path.join(SKILL, "scripts", "convert_to_tex.py"),
        "--input", "docs/GPTEnergy.html", "--out-dir", "latex", "--columns", "2"])
    sh([PY, os.path.join(SKILL, "scripts", "pack_tmlr_bundle.py"),
        "--in-dir", "latex", "--template", "elsarticle"])

    main_p = os.path.join(HERE, "main.tex")
    tex = open(main_p, encoding="utf-8").read()

    # carve out the stray front matter the converter left before Introduction
    fm_end = tex.index("\\end{frontmatter}") + len("\\end{frontmatter}")
    intro = tex.index("\\section{Introduction}")
    front_seg = tex[fm_end:intro]
    m = re.search(r"^Abstract\s*$(.*)", front_seg, re.M | re.S)
    abstract = m.group(1).strip() if m else ""
    assert len(abstract) > 500, "abstract extraction failed"
    tex = tex[:fm_end] + "\n\n" + tex[intro:]

    tex = tex.replace("\\author{Anonymous Authors}\n\\address{Anonymous Affiliations}",
        "\\author[hit]{Alexander Apartsin}\n"
        "\\author[afeka]{Yehudit Aperstein\\corref{cor1}}\n"
        "\\cortext[cor1]{Corresponding author}\n"
        "\\ead{apersteiny@afeka.ac.il}\n"
        "\\address[hit]{Holon Institute of Technology, Holon, Israel}\n"
        "\\address[afeka]{Afeka Academic College of Engineering, Tel Aviv, Israel}")
    tex = tex.replace("__JOURNAL__", "Sustainable Computing: Informatics and Systems")
    tex = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}",
                 lambda _: "\\begin{abstract}\n" + abstract + "\n\\end{abstract}",
                 tex, count=1, flags=re.S)
    # Pull the title block up: the 3p title box otherwise leaves a tall white
    # margin above the paper title on page 1.
    tex = tex.replace("\\title{The Joule Point",
                      "\\title{\\vspace*{-2\\baselineskip}The Joule Point")

    # Journal (non-ACL) back matter: Limitations is a NUMBERED section here
    # (the converter stars it per the ACL convention), Data availability is
    # unnumbered back matter, and the \section{References} heading goes away
    # entirely (thebibliography prints its own, so it would appear twice).
    tex = tex.replace("\\section*{Limitations}", "\\section{Limitations}")
    tex = tex.replace("\\section{Data and code availability}",
                      "\\section*{Data and code availability}")
    tex = re.sub(r"\\section\{References\}\\label\{references\}\n?", "", tex)

    # Figure 6 (power budget) floats [b]: at [tbp] it takes the top of its
    # column and splits the "Pricing" bullet, whose continuation then resumes
    # under the caption; bottom placement keeps the bullet text contiguous
    # above the figure.
    fig6 = tex.index("fig_power_budget_stacked")
    head = tex.rindex("\\begin{figure}[tbp]", 0, fig6)
    tex = tex[:head] + "\\begin{figure}[b]" + tex[head + len("\\begin{figure}[tbp]"):]

    open(main_p, "w", encoding="utf-8").write(tex)

    sh([PY, os.path.join(SKILL, "scripts", "compile_local.py"), "--in-dir", "latex", "--auto-patch"])
    shutil.copy(os.path.join(HERE, "main.pdf"), os.path.join(ROOT, "docs", "JoulePoint_2col_latex.pdf"))
    print("done -> docs/JoulePoint_2col_latex.pdf")


if __name__ == "__main__":
    main()
