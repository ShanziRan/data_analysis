from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUT = Path("output/hitl/OCR_Human_Review_Model_Leader_Brief.docx")
ASSET_DIR = Path("output/hitl/doc_assets")


def diagram_font(size, bold=False):
    choices = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for path in choices:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def multiline_center(draw, box, text, size=25, bold=False):
    font = diagram_font(size, bold)
    left, top, right, bottom = box
    lines = text.split("\n")
    heights = []
    for line in lines:
        bounds = draw.textbbox((0, 0), line, font=font)
        heights.append(bounds[3] - bounds[1])
    total = sum(heights) + (len(lines) - 1) * 6
    y = top + (bottom - top - total) / 2
    for line, height in zip(lines, heights):
        bounds = draw.textbbox((0, 0), line, font=font)
        width = bounds[2] - bounds[0]
        draw.text((left + (right - left - width) / 2, y), line, font=font, fill="black")
        y += height + 6


def arrow(draw, start, end):
    draw.line((start, end), fill="black", width=4)
    x, y = end
    draw.polygon([(x, y), (x - 16, y - 9), (x - 16, y + 9)], fill="black")


def training_diagram(path):
    image = Image.new("RGB", (1800, 340), "white")
    draw = ImageDraw.Draw(image)
    boxes = [(30 + i * 295, 85, 255 + i * 295, 255) for i in range(6)]
    labels = [
        "Reviewed data\nhuman.csv",
        "Compare\nCaptured vs Published",
        "Create\ntraining labels",
        "Calculate\n16 features",
        "Split by scan_id\nand train",
        "Save model\nand threshold",
    ]
    for box, label in zip(boxes, labels):
        draw.rounded_rectangle(box, radius=10, outline="black", width=3, fill="white")
        multiline_center(draw, box, label, 23, bold=False)
    for left, right in zip(boxes, boxes[1:]):
        arrow(draw, (left[2] + 8, 170), (right[0] - 8, 170))
    image.save(path)


def scoring_diagram(path):
    image = Image.new("RGB", (1800, 520), "white")
    draw = ImageDraw.Draw(image)
    top_boxes = [(25 + i * 340, 55, 285 + i * 340, 210) for i in range(5)]
    labels = [
        "Unreviewed data\nno_human.csv",
        "Parse key and\ncalculate features",
        "Predict review\nprobability",
        "Apply saved\nthreshold",
        "Risk and\nreview flag",
    ]
    for box, label in zip(top_boxes, labels):
        draw.rounded_rectangle(box, radius=10, outline="black", width=3, fill="white")
        multiline_center(draw, box, label, 23)
    for left, right in zip(top_boxes, top_boxes[1:]):
        arrow(draw, (left[2] + 8, 132), (right[0] - 8, 132))
    low = (1190, 320, 1450, 465)
    review = (1510, 320, 1770, 465)
    draw.rounded_rectangle(low, radius=10, outline="black", width=3, fill="white")
    draw.rounded_rectangle(review, radius=10, outline="black", width=3, fill="white")
    multiline_center(draw, low, "Low risk\nquality-control sample", 22)
    multiline_center(draw, review, "Medium / high\nhuman review", 22)
    source = top_boxes[-1]
    draw.line((source[0] + 70, source[3] + 8, source[0] + 70, 285, 1320, 285, 1320, low[1] - 8), fill="black", width=4)
    draw.polygon([(1320, low[1]), (1311, low[1] - 16), (1329, low[1] - 16)], fill="black")
    draw.line((source[2] - 55, source[3] + 8, source[2] - 55, 285, 1640, 285, 1640, review[1] - 8), fill="black", width=4)
    draw.polygon([(1640, review[1]), (1631, review[1] - 16), (1649, review[1] - 16)], fill="black")
    image.save(path)


def set_alt_text(shape, description):
    doc_pr = shape._inline.docPr
    doc_pr.set("descr", description)
    doc_pr.set("title", description)


def font(run, size=10.5, bold=False):
    run.font.name = "Arial"
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), "Arial")
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), "Arial")
    run.font.size = Pt(size)
    run.bold = bold
    run.font.color.rgb = RGBColor(0, 0, 0)


def para(doc, text, after=5):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.08
    font(p.add_run(text))
    return p


def heading(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    font(p.add_run(text), size=12, bold=True)
    return p


def bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Inches(.32)
    p.paragraph_format.first_line_indent = Inches(-.18)
    p.paragraph_format.space_after = Pt(2.5)
    p.paragraph_format.line_spacing = 1.04
    font(p.add_run(text))
    return p


def numbered(doc, text):
    p = doc.add_paragraph(style="List Number")
    p.paragraph_format.left_indent = Inches(.32)
    p.paragraph_format.first_line_indent = Inches(-.18)
    p.paragraph_format.space_after = Pt(2.5)
    p.paragraph_format.line_spacing = 1.04
    font(p.add_run(text))
    return p


ASSET_DIR.mkdir(parents=True, exist_ok=True)
training_path = ASSET_DIR / "training_workflow.png"
scoring_path = ASSET_DIR / "scoring_workflow.png"
training_diagram(training_path)
scoring_diagram(scoring_path)

doc = Document()
section = doc.sections[0]
section.page_width = Inches(8.5)
section.page_height = Inches(11)
section.top_margin = section.bottom_margin = Inches(.7)
section.left_margin = section.right_margin = Inches(.8)
section.header_distance = section.footer_distance = Inches(.3)

normal = doc.styles["Normal"]
normal.font.name = "Arial"
normal.font.size = Pt(10.5)
normal.font.color.rgb = RGBColor(0, 0, 0)
normal.paragraph_format.space_after = Pt(5)
normal.paragraph_format.line_spacing = 1.08

title = doc.add_paragraph()
title.paragraph_format.space_after = Pt(2)
font(title.add_run("OCR human-review model"), size=18, bold=True)
subtitle = doc.add_paragraph()
subtitle.paragraph_format.space_after = Pt(8)
font(subtitle.add_run("Short discussion guide: training, scoring and use"), size=11)

heading(doc, "The aim")
para(doc, "Use previous human review decisions to identify which new OCR-captured answers are most likely to need human checking. The model prioritises work; it does not correct answers or replace the reviewer.")

heading(doc, "What it is trained on")
bullet(doc, "Source: data/hitl/human.csv - 341,190 reviewed rows.")
bullet(doc, "Captured is the original OCR answer. Published is the human-verified answer.")
bullet(doc, "Training label - no correction: normalised Captured equals Published.")
bullet(doc, "Training label - review needed: normalised Captured differs from Published.")
bullet(doc, "There are 7,857 corrected rows (about 2.3%), so corrected examples receive extra weight.")
bullet(doc, "Validation is split by scan_id, keeping answers from the same scan together.")

heading(doc, "Training workflow")
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(3)
shape = p.add_run().add_picture(str(training_path), width=Inches(6.75))
set_alt_text(shape, "Training workflow from reviewed data through labels and features to the saved model and threshold")

heading(doc, "Important rules")
bullet(doc, "Published creates training labels but is never a scoring input.")
bullet(doc, "Answer-key parsing is independent of Captured and bounded for very large keys.")
bullet(doc, "Blank non-MCQ captures have no distance or similarity; the blank indicator is used.")
bullet(doc, "MCQ correction types can only be assigned to single-letter MCQ answer keys.")

doc.add_page_break()

heading(doc, "Features used by the main review model")
for item in [
    "ocr_confidence - OCR engine confidence.",
    "edit_similarity - normalised similarity to the closest accepted variation.",
    "char_ngram_similarity - similarity between short overlapping character groups.",
    "token_similarity - overlap between words/tokens.",
    "answer_exact - whether Captured exactly matches an accepted variation.",
    "is_blank - whether Captured is empty or --blank--.",
    "is_mcq - whether the answer key is a single-letter MCQ key.",
    "captured_length and best_answer_length.",
    "length_difference - difference between the two lengths.",
    "digit_fraction, alpha_fraction and punctuation_fraction.",
    "variation_count - number of bounded accepted variations considered.",
    "known_confusion_score - strength of the strongest historical character confusion.",
    "known_confusion_fraction - share of required edits recognised from history.",
]:
    bullet(doc, item)
para(doc, "min_distance is also logged in the scored CSV as supporting evidence. It is left empty for blank non-MCQ captures.", after=4)

heading(doc, "Possible labels and correction types")
bullet(doc, "Review flag: requires_human_review is True or False.")
bullet(doc, "Risk label: low, medium or high.")
bullet(doc, "Correction type: no_correction, substitution, insertion, deletion, transposition or multiple_edits.")
bullet(doc, "Blank-related type: blank_to_text or text_to_blank.")
bullet(doc, "MCQ type: mcq_correction, only for MCQ answer keys.")
para(doc, "The correction type is supporting guidance. It is not an automatic correction and does not determine the risk label.")

heading(doc, "Current risk rules")
bullet(doc, "Low: probability is below review_threshold; requires_human_review is False.")
bullet(doc, "Medium: probability is at or above review_threshold but below max(0.8, review_threshold).")
bullet(doc, "High: probability is at or above max(0.8, review_threshold).")
para(doc, "The threshold targets 95% correction recall by default. This favours catching errors but can create a larger review queue.")

doc.add_page_break()

heading(doc, "Scoring and review workflow")
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(4)
shape = p.add_run().add_picture(str(scoring_path), width=Inches(6.75))
set_alt_text(shape, "Scoring workflow from unreviewed data through probability and threshold to low-risk sampling or human review")

heading(doc, "How the flagged CSV should be used")
bullet(doc, "Filter requires_human_review = True and sort review_probability from highest to lowest.")
bullet(doc, "Show Captured, ANSWER Key, OCR confidence, similarity evidence and the original scan image.")
bullet(doc, "Record the verified result in new human-review fields; do not overwrite Captured.")
bullet(doc, "Audit low-risk rows, especially those just below the threshold.")
bullet(doc, "Add completed reviews to future training data after quality checks.")

heading(doc, "Outputs to keep")
bullet(doc, "review_probability, requires_human_review and risk_label.")
bullet(doc, "predicted_correction_type, as supporting guidance only.")
bullet(doc, "min_distance, edit_similarity, char_ngram_similarity and token_similarity.")
bullet(doc, "known_confusion_score and known_confusion_fraction.")
bullet(doc, "UID, scan_id, model version, threshold, reviewer decision and review timestamp.")

heading(doc, "Points to agree with the lead")
bullet(doc, "What missed-correction rate is acceptable?")
bullet(doc, "How many rows can reviewers handle?")
bullet(doc, "What percentage of low-risk rows should be quality checked?")
bullet(doc, "Who can approve threshold changes and retraining?")
bullet(doc, "Which measures will be reported: recall, review precision, review rate, low-risk miss rate and reviewer time?")

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print(OUT.resolve())
