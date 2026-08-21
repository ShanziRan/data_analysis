from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math


OUT = Path("output/hitl/ocr_review_decision_flowchart.jpg")
W, H = 3000, 4400
BG = "white"
INK = "#111111"
LINE = "#333333"
DECISION = "#EAF2F8"
ACTION = "#F8F9F9"
LOW = "#E8F5E9"
MEDIUM = "#FFF4D6"
HIGH = "#FDEDEC"


def font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


F = font(36)
FB = font(38, True)
FE = font(29)


img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)


def wrapped(text, max_width, use_font=F):
    lines = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        current = ""
        for word in words:
            trial = word if not current else f"{current} {word}"
            if d.textbbox((0, 0), trial, font=use_font)[2] <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word
        lines.append(current)
    return lines


def node(cx, cy, width, height, text, kind="action", fill=None):
    x1, y1, x2, y2 = cx-width/2, cy-height/2, cx+width/2, cy+height/2
    fill = fill or (DECISION if kind == "decision" else ACTION)
    if kind == "decision":
        points = [(cx, y1), (x2, cy), (cx, y2), (x1, cy)]
        d.polygon(points, fill=fill, outline=INK, width=4)
        text_width = width * .62
    else:
        d.rounded_rectangle((x1, y1, x2, y2), radius=24, fill=fill, outline=INK, width=4)
        text_width = width - 70
    use_font = FB if any(text.startswith(p) for p in ("LOW RISK", "MEDIUM RISK", "HIGH RISK")) else F
    lines = wrapped(text, text_width, use_font)
    line_h = use_font.size + 12
    y = cy - len(lines) * line_h / 2
    for line in lines:
        box = d.textbbox((0, 0), line, font=use_font)
        d.text((cx-(box[2]-box[0])/2, y), line, font=use_font, fill=INK)
        y += line_h


def arrow(a, b, label="", offset=(0, 0)):
    x1, y1 = a
    x2, y2 = b
    d.line((x1, y1, x2, y2), fill=LINE, width=5)
    ang = math.atan2(y2-y1, x2-x1)
    size = 22
    p1 = (x2-size*math.cos(ang-.55), y2-size*math.sin(ang-.55))
    p2 = (x2-size*math.cos(ang+.55), y2-size*math.sin(ang+.55))
    d.polygon([(x2, y2), p1, p2], fill=LINE)
    if label:
        mx, my = (x1+x2)/2 + offset[0], (y1+y2)/2 + offset[1]
        box = d.textbbox((0, 0), label, font=FE)
        pad = 8
        d.rectangle((mx-(box[2]-box[0])/2-pad, my-(box[3]-box[1])/2-pad,
                     mx+(box[2]-box[0])/2+pad, my+(box[3]-box[1])/2+pad), fill=BG)
        d.text((mx-(box[2]-box[0])/2, my-(box[3]-box[1])/2), label, font=FE, fill=INK)


# Top split and feature calculation.
node(1500, 140, 760, 150, "New OCR answer")
node(1500, 400, 860, 230, "Is the answer key MCQ?", "decision")
node(760, 730, 760, 220, "Is captured answer blank?", "decision")
node(2240, 730, 760, 220, "Is captured answer blank?", "decision")
arrow((1500, 215), (1500, 285))
arrow((1230, 480), (900, 620), "Yes", (-25, -20))
arrow((1770, 480), (2100, 620), "No", (25, -20))

node(390, 1080, 650, 210, "Calculate MCQ similarities and model features")
node(1130, 1080, 650, 210, "Compare OCR answer with accepted MCQ options")
node(1870, 1080, 650, 230, "Do not calculate distance or similarity\nRecord is_blank = 1")
node(2610, 1080, 650, 210, "Compare OCR answer with parsed answer-key variations")
arrow((620, 820), (460, 975), "Yes", (-30, -5))
arrow((900, 820), (1060, 975), "No", (30, -5))
arrow((2100, 820), (1940, 965), "Yes", (-30, -5))
arrow((2380, 820), (2540, 975), "No", (30, -5))

node(1500, 1400, 720, 150, "Predict review probability")
for x in (390, 1130, 1870, 2610):
    arrow((x, 1195), (1500, 1325))

# Risk decision.
node(1500, 1690, 880, 230, "Probability below review threshold?", "decision")
arrow((1500, 1475), (1500, 1575))
node(570, 2020, 740, 230, "LOW RISK\nNo human review\nType: no_correction", fill=LOW)
node(1880, 2020, 900, 260, "Probability at least\nmax(0.80, review threshold)?", "decision")
arrow((1170, 1760), (770, 1910), "Yes", (-30, -12))
arrow((1700, 1790), (1810, 1890), "No", (30, -8))
node(1390, 2380, 720, 220, "MEDIUM RISK\nAdd to normal review queue", fill=MEDIUM)
node(2370, 2380, 720, 220, "HIGH RISK\nPrioritise for human review", fill=HIGH)
arrow((1690, 2130), (1450, 2270), "No", (-20, -8))
arrow((2070, 2130), (2310, 2270), "Yes", (20, -8))

# Correction-type guidance.
node(1880, 2720, 720, 220, "MCQ?", "decision")
arrow((1390, 2490), (1740, 2610))
arrow((2370, 2490), (2020, 2610))
node(760, 3050, 840, 240, "Suggested type: mcq_correction\nCheck selected option against answer key")
node(2080, 3050, 900, 250, "Blank-state change expected?", "decision")
arrow((1610, 2790), (980, 2940), "Yes", (-20, -15))
arrow((2080, 2830), (2080, 2925), "No", (45, 0))

node(720, 3420, 820, 250, "Suggested type: blank_to_text\nInspect original image and enter missing answer")
node(1500, 3420, 820, 250, "Suggested type: text_to_blank\nCheck for OCR noise or stray marks")
node(2450, 3420, 880, 240, "Likely character-level change", "decision")
arrow((1840, 3120), (880, 3295), "OCR blank; likely text", (-20, -20))
arrow((2080, 3175), (1620, 3295), "OCR text; likely blank", (0, -20))
arrow((2320, 3120), (2400, 3300), "No", (35, 0))

leaves = [
    (260, "substitution\nCheck common OCR confusion", "One character replaced"),
    (760, "insertion\nAdd missing character", "Character missing"),
    (1260, "deletion\nRemove OCR artefact", "Extra character"),
    (1760, "transposition\nCorrect character order", "Characters reversed"),
    (2260, "multiple_edits\nReview the full response", "Several differences"),
    (2760, "case_or_whitespace\nConfirm normalisation is acceptable", "Case or spacing only"),
]
for x, text, label in leaves:
    node(x, 4020, 440, 250, text)
    arrow((2450, 3540), (x, 3895))
    label_lines = wrapped(label, 400, font(23))
    label_y = 3790 - len(label_lines) * 17
    for label_line in label_lines:
        box = d.textbbox((0, 0), label_line, font=font(23))
        d.rectangle((x-(box[2]-box[0])/2-5, label_y-3,
                     x+(box[2]-box[0])/2+5, label_y+(box[3]-box[1])+3), fill=BG)
        d.text((x-(box[2]-box[0])/2, label_y), label_line, font=font(23), fill=INK)
        label_y += 31

OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT, "JPEG", quality=95, subsampling=0)
print(OUT.resolve())
