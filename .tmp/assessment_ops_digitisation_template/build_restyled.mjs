import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const ROOT = "/Users/ranshanzi/Documents/VSCode/data_analysis";
const TMP = `${ROOT}/.tmp/assessment_ops_digitisation_template`;
const OUT = `${ROOT}/output/hitl/Assessment_Operations_OCR_Monitoring_Deck_Digitisation_Template.pptx`;
const p = await PresentationFile.importPptx(await FileBlob.load(`${TMP}/template-starter.pptx`));

const targetSpecs = new Map([
  ["sh/bi18fqdk",[1,"shape","Text Placeholder 5"]],
  ["sh/n6xkbup0",[2,"shape","Title 1"]],["sh/wbqloz6h",[2,"shape","Content Placeholder 2"]],["sh/o76lkf6l",[2,"shape","Slide Number Placeholder 4"]],
  ["sh/g7ml0v6x",[3,"shape","Title 1"]],["sh/3udkba5o",[3,"shape","Content Placeholder 2"]],["sh/f6dk7q5s",[3,"shape","Slide Number Placeholder 4"]],
  ["sh/yxsjel43",[4,"shape","Title 1"]],["sh/orq18bm1",[4,"shape","Content Placeholder 2"]],["sh/9oj2t03q",[4,"shape","Slide Number Placeholder 4"]],
  ["sh/i90fy9g3",[5,"shape","Title 1"]],["sh/p8vy58zy",[5,"shape","Content Placeholder 2"]],["sh/5cbet4ze",[5,"shape","Slide Number Placeholder 4"]],
  ["sh/47ehwbm1",[6,"shape","Title 1"]],["sh/ip4fi98n",[6,"shape","Content Placeholder 2"]],["sh/wnmxgzqh",[6,"shape","Slide Number Placeholder 6"]],
  ["sh/8nipo7e9",[7,"shape","Title 1"]],["sh/itkru1wb",[7,"shape","Content Placeholder 2"]],["sh/zyp8r2xs",[7,"shape","Slide Number Placeholder 4"]],
  ["sh/72l8ny9g",[8,"shape","Title 1"]],["sh/87qp4zah",[8,"shape","Content Placeholder 2"]],["sh/s3upg3al",[8,"shape","Slide Number Placeholder 4"]],
  ["sh/43qtk3ql",[9,"shape","Title 1"]],["sh/m94ba987",[9,"shape","Content Placeholder 2"]],["sh/vyxs7y9o",[9,"shape","Slide Number Placeholder 4"]],
  ["sh/kfipkfuh",[10,"shape","Title 1"]],["sh/baponqd0",[10,"shape","Content Placeholder 2"]],["sh/ze9oradw",[10,"shape","Slide Number Placeholder 4"]],
  ["sh/4fqd4za1",[11,"shape","Title 1"]],["sh/elsva9s3",[11,"shape","Content Placeholder 2"]],["sh/ri1czutc",[11,"shape","Slide Number Placeholder 4"]],
  ["sh/n6dc7yhg",[12,"shape","Title 1"]],["sh/4b2tojih",[12,"shape","Content Placeholder 2"]],["sh/o7md03i1",[12,"shape","Slide Number Placeholder 4"]],
  ["sh/gjyd8jyd",[13,"shape","Title 1"]],["sh/369wjeh4",[13,"shape","Content Placeholder 2"]],["sh/fipwfehs",[13,"shape","Slide Number Placeholder 4"]],
  ["sh/210n2pwv",[14,"shape","Title 1"]],["sh/5s7m10f2",[14,"shape","Content Placeholder 2"]],["sh/por6d4f6",[14,"shape","Slide Number Placeholder 4"]],
  ["sh/dsfu9ori",[15,"shape","Title 1"]],["sh/214vupsv",[15,"shape","Content Placeholder 2"]],["sh/q5ovy9sr",[15,"shape","Slide Number Placeholder 4"]]
]);
function named(slideNumber, kind, name) {
  const slide = p.slides.items[slideNumber - 1];
  const collection = kind === "shape" ? slide.shapes : kind === "table" ? slide.tables : slide.images;
  const item = collection.items.find(x => x.name === name);
  if (!item) throw new Error(`Missing ${kind} ${name} on slide ${slideNumber}`);
  return item;
}
function resolveTarget(id) {
  const spec = targetSpecs.get(id);
  if (!spec) throw new Error(`Missing target spec for ${id}`);
  return named(...spec);
}

const textEdits = {
  "sh/bi18fqdk": "Operating the OCR review and answer-key monitoring pipeline",
  "sh/n6xkbup0": "The monitor protects against two different failure modes",
  "sh/wbqloz6h": "1 — OCR CAPTURE ERROR\nDoes the captured text need correction? Inspect the original script image and compare it with Captured.\n\n2 — ANSWER-KEY COVERAGE GAP\nCould a valid response be missing from the accepted variations? Review semantic and conflict evidence.\n\nKEEP THE DECISIONS SEPARATE\nA row may trigger OCR review, AK review, both, or neither.",
  "sh/o76lkf6l": "2",
  "sh/g7ml0v6x": "Human corrections train OCR review—not AK coverage",
  "sh/3udkba5o": "Published creates the correction label during training. It is never used as a scoring feature. AK suggestions remain rule-based until reviewed suggestion outcomes are available.",
  "sh/f6dk7q5s": "3",
  "sh/yxsjel43": "How reviewed corrections train the OCR model",
  "sh/orq18bm1": "Validation selects the probability threshold that meets the target correction recall. The final model is then refitted on all reviewed rows and saved with its threshold, features and learned confusion patterns.",
  "sh/9oj2t03q": "4",
  "sh/i90fy9g3": "The OCR model combines several weak signals",
  "sh/p8vy58zy": "SURFACE MATCH\nEdit similarity; character 3-grams; token overlap; exact AK match.\n\nANSWER STRUCTURE\nBlank and MCQ indicators; response lengths; variation count.\n\nOCR HISTORY AND SHAPE\nOCR confidence; character shares; learned confusion strength and coverage.\n\nSentence embeddings are used only for AK coverage.",
  "sh/5cbet4ze": "5",
  "sh/47ehwbm1": "The OCR threshold catches 98% of known corrections",
  "sh/ip4fi98n": "The saved threshold is 0.6814. Validation recall is 98.03%; average precision 61.27%; review precision 24.95%; review rate 8.84%.\n\nThirty corrected validation rows were missed and 1,494 were flagged. Higher recall increases the human-review queue.",
  "sh/wnmxgzqh": "6",
  "sh/8nipo7e9": "Scoring separates OCR review from AK coverage",
  "sh/itkru1wb": "Each row is parsed once, then evaluated by two decision paths. Combined routing de-duplicates rows and records why each item entered the review queue.",
  "sh/zyp8r2xs": "7",
  "sh/72l8ny9g": "Decision 1 prioritises likely OCR corrections",
  "sh/87qp4zah": "LOW RISK\nBelow threshold. No OCR review, but keep a quality-control sample.\n\nMEDIUM RISK\nAt or above the saved threshold. Add to the normal review queue.\n\nHIGH RISK\nAt or above max(0.80, threshold). Prioritise for review.\n\nThe predicted correction type is guidance only. The original image remains the source of truth.",
  "sh/s3upg3al": "8",
  "sh/43qtk3ql": "AK-gap suggestions need semantic evidence and no conflict",
  "sh/m94ba987": "A possible gap must be non-MCQ, non-blank and not an exact match. It needs a strong whole-string semantic score, low surface match and a semantic–surface gap. Conflict checks block related-but-not-equivalent answers.",
  "sh/vyxs7y9o": "9",
  "sh/kfipkfuh": "Semantic similarity alone is not enough",
  "sh/baponqd0": "A possible gap means review the answer key—not accept automatically. Number, negation, unit, date/time, polarity, key-term and single-word conflicts can block a suggestion.",
  "sh/ze9oradw": "10",
  "sh/4fqd4za1": "One routed queue, with evidence matched to the reason",
  "sh/elsva9s3": "Prioritise high OCR, then medium, then AK-only. Inspect the source image, review the relevant evidence, record a structured outcome, then close or escalate. Sample low-risk and conflict-blocked rows for quality control.",
  "sh/ri1czutc": "11",
  "sh/n6dc7yhg": "Structured reviewer outcomes create the next learning loop",
  "sh/4b2tojih": "EVERY REVIEWED ROW\nUID and scan_id; original Captured and image reference; reviewer-corrected text; decision reason; reviewer ID; timestamp; model hash and thresholds.\n\nAK CANDIDATE OUTCOME\nacceptable_new_variation; related_but_incorrect; incorrect_unrelated; ocr_error; uncertain.\n\nThese labels are required before AK-gap logic can be trained and validated rather than operated as rules.",
  "sh/o7md03i1": "12",
  "sh/gjyd8jyd": "Monitor quality, workload and drift",
  "sh/369wjeh4": "EACH RUN\nModel version; thresholds; rows scored; eligible AK rows; OCR, AK and combined flags; risk counts.\n\nWEEKLY OPERATIONS\nQueue size and ageing; reviews completed; reviewer time; escalations; blocked-row QC.\n\nVERIFIED QUALITY SAMPLES\nOCR recall; review precision; low-risk miss rate; AK acceptance rate; results by segment.",
  "sh/fipwfehs": "13",
  "sh/210n2pwv": "The latest run routes 3.96% of rows to human review",
  "sh/5s7m10f2": "Most workload is OCR review. AK review is smaller, but requires a distinct decision form and approval route. Combined routing removes duplicate rows that trigger both decisions.",
  "sh/por6d4f6": "14",
  "sh/dsfu9ori": "Agree the operating controls before routine monitoring begins",
  "sh/214vupsv": "Queue ownership and service level — who works each case, and how quickly?\nQuality control and change authority — what is sampled, and who approves changes?\nFeedback and cadence — where are outcomes stored, quality checked and discussed?",
  "sh/q5ovy9sr": "15"
};
for (const [id, value] of Object.entries(textEdits)) resolveTarget(id).text = value;

// Normalise inherited text runs to the source deck's Arial system. Imported
// placeholders can retain mixed paragraph-level formatting after replacement.
const titleIds = [
  "sh/n6xkbup0","sh/g7ml0v6x","sh/yxsjel43","sh/i90fy9g3","sh/47ehwbm1",
  "sh/8nipo7e9","sh/72l8ny9g","sh/43qtk3ql","sh/kfipkfuh","sh/4fqd4za1",
  "sh/n6dc7yhg","sh/gjyd8jyd","sh/210n2pwv","sh/dsfu9ori"
];
const bodyIds = [
  "sh/wbqloz6h","sh/3udkba5o","sh/orq18bm1","sh/p8vy58zy","sh/ip4fi98n",
  "sh/itkru1wb","sh/87qp4zah","sh/m94ba987","sh/baponqd0","sh/elsva9s3",
  "sh/4b2tojih","sh/369wjeh4","sh/5s7m10f2","sh/214vupsv"
];
const pageIds = [
  "sh/o76lkf6l","sh/f6dk7q5s","sh/9oj2t03q","sh/5cbet4ze","sh/wnmxgzqh",
  "sh/zyp8r2xs","sh/s3upg3al","sh/vyxs7y9o","sh/ze9oradw","sh/ri1czutc",
  "sh/o7md03i1","sh/fipwfehs","sh/por6d4f6","sh/q5ovy9sr"
];
for (const id of titleIds) {
  const titleShape = resolveTarget(id);
  titleShape.frame = { left: 35.17, top: 132.28, width: 1209.67, height: 65.61 };
  titleShape.text.style = { fontSize: 37.33, typeface: "Arial", color: "#000000", bold: false };
}
for (const id of bodyIds) resolveTarget(id).text.style = { fontSize: 22, typeface: "Arial", color: "#000000", bold: false, autoFit: "shrinkText" };
for (const id of pageIds) resolveTarget(id).text.style = { fontSize: 10.67, typeface: "Arial", color: "#000000", bold: false };

const tableEdits = {
  "tb/b6x83qxw": [
    ["Training evidence", "Current result", "Operational meaning"],
    ["Reviewed rows", "341,190", "Human-verified training population"],
    ["Corrections", "7,857 (2.3%)", "Rare class receives extra training weight"],
    ["Validation split", "67,712 rows", "Grouped by scan_id to reduce leakage"]
  ],
  "tb/re9o7a1g": [
    ["Captured", "Closest AK variation", "Decision"],
    ["cat", "dog", "BLOCK — different single words"],
    ["day after Wednesday", "Thursday", "POSSIBLE GAP — no conflict"],
    ["forty pounds", "fourteen pounds", "BLOCK — number mismatch"]
  ],
  "tb/tk3a5o36": [
    ["Measure", "Latest result", "Operational meaning"],
    ["Rows scored", "138,810", "Unreviewed scoring population"],
    ["OCR / AK flags", "5,477 / 193", "Two separate decision paths"],
    ["Combined queue", "5,501 (3.96%)", "169 both; 24 AK-only; 5,308 OCR-only"]
  ]
};
const tableSlides = {"tb/b6x83qxw":3,"tb/re9o7a1g":10,"tb/tk3a5o36":14};
for (const [id, rows] of Object.entries(tableEdits)) {
  const table = named(tableSlides[id], "table", "Table 2");
  rows.forEach((row, r) => row.forEach((value, c) => table.cells.set(r, c, value)));
}

async function replaceImage(slideNumber, path) {
  const slide = p.slides.items[slideNumber - 1];
  const oldImage = named(slideNumber, "image", slideNumber === 6 ? "Picture 60" : "Picture 12");
  const frame = oldImage.frame;
  const geometry = oldImage.geometry;
  oldImage.delete();
  const bytes = await fs.readFile(path);
  const blob = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  slide.images.add({ blob, contentType: path.endsWith(".jpg") ? "image/jpeg" : "image/png", alt: "Pipeline workflow", fit: "contain", position: frame, geometry });
}
await replaceImage(4, `${TMP}/assets/training.png`);
await replaceImage(6, `${TMP}/assets/validation.png`);
await replaceImage(7, `${TMP}/assets/scoring.png`);
await replaceImage(9, `${TMP}/assets/ak_conflicts.png`);
await replaceImage(11, `${TMP}/assets/ops_workflow.png`);

const talks = [
  "Timing: 1 minute. Frame this as an operating briefing for the teams who will work and monitor the queue.",
  "Timing: 1.5 minutes. Explain the difference between correcting OCR capture and reviewing answer-key coverage.",
  "Timing: 1.5 minutes. Explain what training truth exists and why AK-gap feedback is not yet a trained target.",
  "Timing: 1.5 minutes. Walk left to right through label creation, validation threshold selection and final refit.",
  "Timing: 1.5 minutes. Explain that weak signals are combined; sentence embeddings belong only to the AK decision.",
  "Timing: 2 minutes. Define recall in operational language and explain the workload trade-off.",
  "Timing: 1.5 minutes. Reinforce the separate decisions and the combined routing layer.",
  "Timing: 1.5 minutes. Explain priority order and that correction type never replaces image inspection.",
  "Timing: 2 minutes. Walk through eligibility, semantic evidence and the conflict checks.",
  "Timing: 1.5 minutes. Use the examples to show why semantic relatedness is not equivalence.",
  "Timing: 2 minutes. Walk the reviewer sequence and quality-control sampling requirement.",
  "Timing: 1.5 minutes. Agree the structured fields needed to create future training evidence.",
  "Timing: 1.5 minutes. Separate run reporting, weekly operations and verified quality measurement.",
  "Timing: 1.5 minutes. Make the latest workload concrete and explain overlap between decisions.",
  "Timing: 1.5 minutes plus discussion. Assign owners and dates for the operating controls."
];
for (let i = 0; i < p.slides.items.length; i++) {
  p.slides.items[i].speakerNotes.textFrame.setText(`${talks[i]}\n\n[Sources]\n- OCR_REVIEW.md\n- ocr_review.py\n- output/hitl/validation_report.json\n- output/hitl/no_human_flagged.scoring_report.json\n[/Sources]`);
  p.slides.items[i].speakerNotes.setVisible(true);
}

await fs.mkdir(`${ROOT}/output/hitl`, { recursive: true });
const out = await PresentationFile.exportPptx(p);
await out.save(OUT);
console.log(OUT);
