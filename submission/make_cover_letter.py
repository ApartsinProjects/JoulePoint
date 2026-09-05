# -*- coding: utf-8 -*-
"""Build submission/cover_letter.docx + .pdf for the SUSCOM submission."""
import os
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

HERE = os.path.dirname(os.path.abspath(__file__))

PARAS = [
    ("right", "Yehudit Aperstein\nAfeka Academic College of Engineering, Tel Aviv, Israel\napersteiny@afeka.ac.il"),
    ("left", "September 5, 2026"),
    ("left", "Editor-in-Chief\nSustainable Computing: Informatics and Systems"),
    ("left", "Dear Editor,"),
    ("left", "We are pleased to submit our manuscript, \"The Joule Point: an Energy-Optimal "
             "Operating Point for AI Inference\" by Alexander Apartsin and Yehudit Aperstein, "
             "for consideration in Sustainable Computing: Informatics and Systems."),
    ("left", "The paper shows by direct measurement that the standard practice of running "
             "data-center GPUs at full rated power wastes a quarter to a third of the energy "
             "spent per inference. For 20 inference models on multiple NVIDIA GPU classes, "
             "board power follows a superlinear response law over the operating point (GPU, "
             "power cap), so energy per inference is U-shaped with a minimum, the Joule Point, "
             "at 43 to 46 per cent of rated power. Under load this optimum is nearly a per-card "
             "constant, so a single static cap per card type, set once, captures the saving "
             "that prior energy-aware systems search for per job at runtime. A trace-driven "
             "fleet simulation over the measured curves serves equal work for 18 to 45 per "
             "cent less energy under a power budget, and an incentive analysis locates the "
             "obstacle to adoption in per-hour GPU pricing rather than in the technology."),
    ("left", "The work sits squarely in the journal's scope of energy-efficient computing "
             "systems and green data centers. Alongside the paper we release ELF, the "
             "measurement dataset behind every figure and number, together with the exact "
             "measurement harnesses, archived at Zenodo (doi:10.5281/zenodo.22058568) under "
             "CC-BY-4.0 (data) and MIT (code)."),
    ("left", "This manuscript is original, has not been published previously, and is not "
             "under consideration by any other journal. All authors have approved the "
             "manuscript and agree with its submission. The authors declare no competing "
             "interests."),
    ("left", "Thank you for considering our submission."),
    ("left", "Sincerely,\n\nYehudit Aperstein (corresponding author)\non behalf of both authors"),
]


def main():
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Georgia"
    st.font.size = Pt(11)
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(1.1)
        s.top_margin = Inches(1.0)
    for align, text in PARAS:
        p = doc.add_paragraph(text)
        p.paragraph_format.space_after = Pt(10)
        if align == "right":
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    docx_path = os.path.join(HERE, "cover_letter.docx")
    doc.save(docx_path)
    print("wrote", docx_path)

    try:
        import win32com.client as w, pythoncom
        pythoncom.CoInitialize()
        app = w.DispatchEx("Word.Application"); app.Visible = False
        d = app.Documents.Open(docx_path, False, True)
        d.ExportAsFixedFormat(os.path.join(HERE, "cover_letter.pdf"), 17)
        d.Close(False); app.Quit()
        print("wrote cover_letter.pdf")
    except Exception as e:
        print("[skip PDF render]", e)


if __name__ == "__main__":
    main()
