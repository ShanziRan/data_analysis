import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT = "/Users/ranshanzi/Documents/VSCode/data_analysis/output/hitl/Assessment_Operations_OCR_Monitoring_Deck.pptx";
const W = 1280, H = 720;
const C = { ink: "#000000", muted: "#50545B", panel: "#EDEDED", rule: "#B8BCC4", blue: "#3D8DFF", pale: "#D0EDFA", white: "#FFFFFF", red: "#D9534F", amber: "#E7A928", green: "#31936A" };
const FONT = "Helvetica Neue";

const p = Presentation.create({ slideSize: { width: W, height: H } });

function box(slide, name, left, top, width, height, fill=C.panel, line=C.rule, radius=false) {
  return slide.shapes.add({ geometry: radius ? "roundRect" : "rect", name,
    position: { left, top, width, height }, fill,
    line: { style: "solid", fill: line, width: 1 } });
}

function text(slide, name, value, left, top, width, height, size=22, opts={}) {
  const sh = slide.shapes.add({ geometry: "textbox", name,
    position: { left, top, width, height }, fill: "none",
    line: { style: "solid", fill: "none", width: 0 } });
  sh.text = value;
  sh.text.style = { fontSize: size, typeface: FONT, color: opts.color || C.ink,
    bold: !!opts.bold, alignment: opts.align || "left",
    verticalAlignment: opts.valign || "top", autoFit: opts.autoFit || "shrinkText",
    insets: opts.insets || { top: 0, right: 0, bottom: 0, left: 0 } };
  return sh;
}

function title(slide, value, num, kicker="ASSESSMENT OPERATIONS") {
  text(slide, `kicker-${num}`, kicker, 42, 26, 500, 28, 16, { bold: true, color: C.muted });
  text(slide, `title-${num}`, value, 42, 62, 1170, 76, 35, { bold: true, autoFit: "shrinkText" });
  slide.shapes.add({ geometry: "straightConnector1", name: `rule-${num}`,
    position: { left: 42, top: 149, width: 1196, height: 0 }, fill: "none",
    line: { style: "solid", fill: C.rule, width: 1 } });
  text(slide, `page-${num}`, String(num), 1158, 677, 80, 22, 13, { align: "right", color: C.muted });
}

function notes(slide, body, sources) {
  slide.speakerNotes.textFrame.setText(`${body}\n\n[Sources]\n${sources.map(s => `- ${s}`).join("\n")}`);
  slide.speakerNotes.setVisible(true);
}

function arrow(slide, name, left, top, width, height=0) {
  const reverse = width < 0 || height < 0;
  const x = width < 0 ? left + width : left;
  const y = height < 0 ? top + height : top;
  return slide.shapes.add({ geometry: "straightConnector1", name,
    position: { left: x, top: y, width: Math.abs(width), height: Math.abs(height) }, fill: "none",
    line: { style: "solid", fill: C.ink, width: 2 },
    ...(reverse
      ? { tail: { type: "arrow", width: "med", length: "med" } }
      : { head: { type: "arrow", width: "med", length: "med" } }) });
}

function bulletList(slide, name, items, left, top, width, height, size=22, gap="\n\n") {
  return text(slide, name, items.map(x => `• ${x}`).join(gap), left, top, width, height, size, { autoFit: "shrinkText" });
}

// 1 — cover
{
  const s = p.slides.add(); s.background.fill = C.white;
  text(s, "cover-kicker", "ASSESSMENT OPERATIONS", 42, 42, 600, 32, 18, { bold: true, color: C.muted });
  text(s, "cover-title", "Operating the OCR review and answer-key monitoring pipeline", 42, 180, 1070, 255, 62, { bold: true, valign: "bottom" });
  text(s, "cover-subtitle", "How to work the queues, interpret evidence, record decisions and monitor quality", 42, 500, 810, 100, 27);
  text(s, "cover-date", "September 2026", 42, 641, 300, 28, 16, { color: C.muted });
  notes(s, "Timing: 1 minute. Set the purpose: this is an operating briefing, not a model-development review. The team should leave knowing which queue to work, what each flag means, and what evidence to record.", ["OCR_REVIEW.md", "ocr_review.py"]);
}

// 2 — two failure modes
{
  const s = p.slides.add(); s.background.fill = C.white; title(s, "The monitor protects against two different failure modes", 2);
  box(s, "ocr-panel", 42, 205, 560, 350, C.panel, C.panel);
  box(s, "ak-panel", 636, 205, 602, 350, C.pale, C.pale);
  text(s, "ocr-head", "1  OCR capture error", 72, 238, 480, 46, 28, { bold: true });
  text(s, "ocr-example", "Written answer\n\"Thursday\"\n\nCaptured\n\"Thursdav\"", 72, 315, 210, 190, 23);
  text(s, "ocr-action", "Operational question\nDoes the captured text need correction?", 315, 315, 235, 150, 22, { bold: true });
  text(s, "ak-head", "2  Answer-key coverage gap", 670, 238, 520, 46, 28, { bold: true });
  text(s, "ak-example", "Captured\n\"the day after Wednesday\"\n\nCurrent AK\n\"Thursday\"", 670, 315, 280, 190, 23);
  text(s, "ak-action", "Operational question\nCould this be a valid alternative missing from the key?", 965, 315, 225, 160, 22, { bold: true });
  text(s, "takeaway", "Keep the decisions separate—even when the same row triggers both.", 42, 604, 1100, 48, 26, { bold: true });
  notes(s, "Timing: 1.5 minutes. Emphasise that OCR review changes Captured to what was written; AK review considers whether the verified wording should become an accepted answer variation.", ["ocr_review.py: score()", "OCR_REVIEW.md"]);
}

// 3 — training data
{
  const s = p.slides.add(); s.background.fill = C.white; title(s, "Human corrections teach the OCR decision—not the AK-gap decision", 3);
  const stats = [["341,190", "reviewed rows"], ["7,857", "human corrections"], ["2.3%", "correction rate"]];
  stats.forEach((d,i) => { const x=42+i*397; box(s, `stat-bg-${i}`, x, 205, 360, 190, C.panel, C.panel); text(s, `stat-${i}`, d[0], x+30, 235, 300, 80, 50, { bold:true }); text(s, `stat-l-${i}`, d[1], x+30, 325, 300, 42, 22); });
  bulletList(s, "train-bullets", [
    "Label = normalised Captured differs from human-verified Published.",
    "Corrected rows receive extra weight because they are rare.",
    "Scans are split by scan_id so related answers do not leak across validation.",
    "AK suggestions are rule-based until accepted/rejected suggestion feedback exists."
  ], 42, 440, 1140, 205, 21, "\n");
  notes(s, "Timing: 1.5 minutes. Explain why the OCR model can be validated against Published, while AK coverage currently cannot: the reviewed file records OCR corrections, not whether a new answer-key variation should be accepted.", ["output/hitl/validation_report.json", "ocr_review.py: train()"]);
}

// 4 — training workflow
{
  const s = p.slides.add(); s.background.fill = C.white; title(s, "Training turns reviewed corrections into a recall-controlled OCR model", 4);
  const xs=[42,218,394,570,746,922,1098];
  for(let i=0;i<6;i++) arrow(s,`train-arrow-${i}`,xs[i]+140,356,36,0);
  const labels=["Reviewed\nhuman.csv","Create OCR\nchange label","Learn character\nconfusions","Calculate 16\nfeatures","Split by\nscan_id","Train probability +\ntype models","Choose recall\nthreshold; refit"];
  xs.forEach((x,i)=>{box(s,`train-node-${i}`,x,285,140,145,i===6?C.pale:C.panel,C.rule);text(s,`train-text-${i}`,labels[i],x+12,315,116,82,19,{bold:i===6,align:"center",valign:"middle"});});
  text(s,"train-note","Published is used to create labels only. It is never used as a scoring feature.",42,500,1080,44,24,{bold:true});
  bulletList(s,"train-detail",["Validation selects the threshold that meets the target correction recall.","The final model is refitted on all reviewed rows after validation."],42,565,1100,90,20,"\n");
  notes(s, "Timing: 1.5 minutes. Walk left to right. Clarify that the saved artifact contains the probability model, correction-type model, learned confusions, feature names, variation limit and review threshold.", ["ocr_review.py: train()", "output/hitl/validation_report.json"]);
}

// 5 — feature families
{
  const s = p.slides.add(); s.background.fill = C.white; title(s, "The OCR score combines several weak signals", 5);
  const cols=[42,444,846]; const heads=["Surface match","Answer structure","OCR history and shape"];
  const bodies=[
    ["Edit similarity", "Character 3-gram similarity", "Exact token-set overlap", "Exact accepted-answer match"],
    ["Blank and MCQ indicators", "Captured and AK lengths", "Length difference", "Accepted-variation count"],
    ["OCR confidence", "Digit / letter / punctuation shares", "Known confusion strength", "Known confusion coverage"]
  ];
  cols.forEach((x,i)=>{text(s,`feat-head-${i}`,heads[i],x,210,350,45,26,{bold:true});box(s,`feat-rule-${i}`,x,265,350,4,C.blue,C.blue);bulletList(s,`feat-body-${i}`,bodies[i],x,300,350,270,21,"\n\n");});
  box(s,"feature-callout",42,605,1160,62,C.pale,C.pale);text(s,"feature-callout-text","Sentence embeddings are not part of the OCR probability. They are used only by the separate AK-coverage decision.",62,622,1120,30,22,{bold:true});
  notes(s, "Timing: 1.5 minutes. The important message is combination: no single similarity threshold determines OCR review. Mention that min_distance is logged as evidence but is not one of the 16 model features.", ["ocr_review.py: FEATURE_COLUMNS", "OCR_REVIEW.md"]);
}

// 6 — validation metrics
{
  const s = p.slides.add(); s.background.fill = C.white; title(s, "The OCR threshold catches 98% of known corrections", 6);
  s.charts.add("bar", { position:{left:42,top:205,width:720,height:390}, categories:["Recall","Average precision","Review precision","Review rate"],
    series:[{name:"Validation %",values:[98.03,61.27,24.95,8.84],fill:C.blue}], hasLegend:false,
    barOptions:{direction:"bar",grouping:"clustered",gapWidth:55}, dataLabels:{showValue:true,position:"outEnd",textStyle:{fontSize:18,fill:C.ink}},
    xAxis:{min:0,max:100,majorUnit:20,numberFormatCode:'0"%"',majorGridlines:{style:"solid",fill:C.rule,width:1}}, yAxis:{visible:true}, chartFill:C.white, chartLine:{fill:C.white,width:0}, plotAreaFill:C.white, plotAreaLine:{fill:C.white,width:0} });
  text(s,"val-threshold","0.6814",835,235,330,80,48,{bold:true}); text(s,"val-label","saved OCR threshold",835,318,330,38,22);
  bulletList(s,"val-meaning",["30 corrected validation rows were missed.","1,494 corrected validation rows were flagged.","Higher recall means a larger human-review queue."],835,400,350,180,21,"\n\n");
  text(s,"val-note","Validation estimates future performance; it does not guarantee the same recall on unreviewed data.",42,625,1160,35,20,{bold:true});
  notes(s, "Timing: 2 minutes. Define recall in operational language: among answers humans truly corrected, how many did the model flag? Explain the workload trade-off: the threshold is selected for recall, not maximum precision.", ["output/hitl/validation_report.json"]);
}

// 7 — scoring workflow
{
  const s = p.slides.add(); s.background.fill=C.white; title(s,"Scoring keeps OCR review and AK coverage as separate decisions",7);
  // connectors first
  arrow(s,"flow-a",610,226,0,52); arrow(s,"flow-b1",610,335,-360,66); arrow(s,"flow-b2",610,335,360,66);
  arrow(s,"flow-c1",250,475,0,65); arrow(s,"flow-c2",970,475,0,65); arrow(s,"flow-d1",370,600,200,0); arrow(s,"flow-d2",910,600,-200,0);
  box(s,"input",410,180,400,46,C.panel,C.rule); text(s,"input-t","Unreviewed OCR data",430,190,360,26,21,{bold:true,align:"center"});
  box(s,"prepare",410,278,400,57,C.panel,C.rule); text(s,"prepare-t","Parse AK independently + calculate evidence",430,292,360,28,20,{align:"center"});
  box(s,"ocr-decision",80,400,340,75,C.pale,C.pale); text(s,"ocr-decision-t","Decision 1: OCR review",100,422,300,30,25,{bold:true,align:"center"});
  box(s,"ak-decision",800,400,340,75,C.pale,C.pale); text(s,"ak-decision-t","Decision 2: AK coverage",820,422,300,30,25,{bold:true,align:"center"});
  box(s,"ocr-output",80,540,340,120,C.panel,C.rule); text(s,"ocr-output-t","Probability → threshold\nRisk + correction type\nrequires_ocr_review",100,557,300,80,20,{align:"center"});
  box(s,"ak-output",800,540,340,120,C.panel,C.rule); text(s,"ak-output-t","Semantic + surface gates\nConflict checks\npossible_gap_suggestion",820,557,300,80,20,{align:"center"});
  box(s,"combine",510,555,200,90,C.white,C.ink); text(s,"combine-t","Combined routing\n+ review reason",525,574,170,50,20,{bold:true,align:"center"});
  notes(s, "Timing: 2 minutes. Stress that requires_any_human_review is a routing convenience. The underlying evidence and reasons remain separate. A row may be OCR only, AK only, both, or neither.", ["ocr_review.py: score()", "output/hitl/no_human_flagged.scoring_report.json"]);
}

// 8 — OCR decision operations
{
  const s=p.slides.add();s.background.fill=C.white;title(s,"Decision 1 prioritises likely OCR corrections",8);
  const risks=[{x:42,w:350,c:C.panel,h:"LOW",b:"Below threshold\nNo OCR review\nKeep in QC sample"},{x:430,w:350,c:"#FFF0CC",h:"MEDIUM",b:"At or above threshold\nNormal review queue\nInspect evidence + scan"},{x:818,w:384,c:"#F5D6D4",h:"HIGH",b:"At or above max(0.8, threshold)\nPriority review\nInspect first"}];
  risks.forEach((r,i)=>{box(s,`risk-${i}`,r.x,210,r.w,215,r.c,r.c);text(s,`risk-h-${i}`,r.h,r.x+24,235,r.w-48,42,29,{bold:true});text(s,`risk-b-${i}`,r.b,r.x+24,302,r.w-48,90,21);});
  text(s,"type-head","Correction type is guidance—not an automatic edit",42,480,650,40,25,{bold:true});
  bulletList(s,"type-list",["substitution / insertion / deletion / transposition / multiple edits","blank_to_text / text_to_blank / case_or_whitespace","mcq_correction only when the answer key is MCQ"],42,540,720,115,19,"\n");
  box(s,"reviewer-rule",810,480,392,165,C.pale,C.pale);text(s,"reviewer-rule-t","Reviewer rule\n\nUse the original image as truth. Do not change Captured solely because the predicted type says so.",836,505,340,120,21,{bold:true});
  notes(s,"Timing: 1.5 minutes. Demonstrate the order reviewers should use: original image first, then Captured and AK, then model evidence. The correction type helps navigation but should never override the image.",["ocr_review.py: correction_type(), score()"]);
}

// 9 — AK gate
{
  const s=p.slides.add();s.background.fill=C.white;title(s,"Decision 2 requires semantic evidence and no material conflict",9);
  const xs=[42,286,530,774,1018]; const labels=["Eligible row\nnon-MCQ, non-blank, not exact","Best whole-string\nsemantic match","Low surface match\n≤ 0.60","Semantic–surface gap\n≥ 0.15","No conflict\npossible gap"];
  for(let i=0;i<4;i++) arrow(s,`ak-arrow-${i}`,xs[i]+190,302,54,0);
  xs.forEach((x,i)=>{box(s,`ak-gate-${i}`,x,245,190,115,i===4?C.pale:C.panel,C.rule);text(s,`ak-gate-t-${i}`,labels[i],x+14,272,162,65,18,{bold:i===4,align:"center"});});
  text(s,"conflict-head","Conflict checks block related-but-not-equivalent answers",42,425,800,40,25,{bold:true});
  const conf=["Different single words","Number mismatch","Date / time mismatch","Negation or polarity","Unit mismatch","Changed key term"];
  conf.forEach((v,i)=>{const x=42+(i%3)*395,y=490+Math.floor(i/3)*75;box(s,`conf-${i}`,x,y,350,52,C.white,C.rule);text(s,`conf-t-${i}`,v,x+16,y+14,318,24,19,{align:"center"});});
  text(s,"ak-note","Blocked rows keep the best variation, semantic score and conflict reasons for audit.",42,650,1100,30,21,{bold:true});
  notes(s,"Timing: 2 minutes. Explain that sentence embeddings measure relatedness as well as equivalence. The extra gates deliberately sacrifice some recall to reduce suggestions such as cat→dog.",["ocr_review.py: make_ak_coverage_decisions(), _ak_conflict_reasons()"]);
}

// 10 — examples table
{
  const s=p.slides.add();s.background.fill=C.white;title(s,"Examples show why semantic similarity alone is not enough",10);
  const values=[
    ["Captured","Closest AK variation","Decision","Reason"],
    ["cat","dog","BLOCK","Different single words"],
    ["the day after Wednesday","Thursday","POSSIBLE GAP","High semantic, low surface, no conflict"],
    ["forty pounds","fourteen pounds","BLOCK","Number and key-term mismatch"],
    ["the light is on","the light is not on","BLOCK","Negation mismatch"],
    ["Thursdav","Thursday","OCR REVIEW","Likely character-level capture error"]
  ];
  const t=s.tables.add({rows:6,columns:4,left:42,top:205,width:1196,height:365,columnWidths:[270,270,190,466],values});
  t.styleOptions={headerRow:true,bandedRows:true}; t.borders.assign({style:"solid",fill:C.rule,width:1});
  for(let c=0;c<4;c++){t.getCell(0,c).fill=C.ink;t.getCell(0,c).text.style={fontSize:18,bold:true,color:C.white};}
  for(let r=1;r<6;r++){for(let c=0;c<4;c++)t.getCell(r,c).text.style={fontSize:17,color:C.ink};}
  t.getCell(2,2).fill=C.pale; t.getCell(2,2).text.style={fontSize:17,bold:true,color:C.ink};
  text(s,"example-note","A possible gap still means “review the answer key,” not “accept automatically.”",42,610,1100,42,24,{bold:true});
  notes(s,"Timing: 2 minutes. Ask the team to say what they would inspect in each case. Reinforce the difference between blocked AK evidence and OCR review evidence.",["Real-model smoke checks performed against ocr_review.py", "ocr_review.py: _ak_conflict_reasons()"]);
}

// 11 — queue workflow
{
  const s=p.slides.add();s.background.fill=C.white;title(s,"Operations should work one routed queue with reason-specific views",11);
  const xs=[42,280,518,756,994]; for(let i=0;i<4;i++) arrow(s,`ops-arrow-${i}`,xs[i]+190,355,48,0);
  const steps=[
    ["1","Open next row","Prioritise high OCR, then medium, then AK-only."],
    ["2","Inspect source","Use the original scan as truth; compare Captured and AK."],
    ["3","Review evidence","OCR scores for capture; semantic/conflict evidence for AK."],
    ["4","Record outcome","Use structured fields. Never overwrite Captured."],
    ["5","Escalate or close","Route AK changes for approval; close normal corrections."]
  ];
  steps.forEach((d,i)=>{box(s,`ops-${i}`,xs[i],240,190,245,i===3?C.pale:C.panel,C.rule);text(s,`ops-num-${i}`,d[0],xs[i]+18,258,40,40,28,{bold:true,color:C.blue});text(s,`ops-head-${i}`,d[1],xs[i]+18,310,154,54,23,{bold:true});text(s,`ops-body-${i}`,d[2],xs[i]+18,382,154,75,17);});
  box(s,"ops-qc",42,555,1142,70,C.white,C.ink);text(s,"ops-qc-t","Quality control: sample low-risk rows, near-threshold rows and conflict-blocked AK rows—not only flagged rows.",64,575,1098,30,22,{bold:true});
  notes(s,"Timing: 2 minutes. This is the main operating slide. Confirm queue ordering, which screen fields are visible, who can approve an AK change, and how low-risk sampling will be assigned.",["OCR_REVIEW.md: Using the review queue", "output/hitl/OCR_Human_Review_Model_Leader_Brief.docx"]);
}

// 12 — record fields
{
  const s=p.slides.add();s.background.fill=C.white;title(s,"Structured reviewer outcomes create the next learning loop",12);
  text(s,"record-left-head","Every reviewed row",42,205,470,40,26,{bold:true});
  bulletList(s,"record-left",["UID and scan_id","Original Captured and scan image reference","Reviewer-corrected text","Decision reason and reviewer ID","Timestamp, model hash and thresholds"],42,270,500,280,21,"\n\n");
  box(s,"record-right-bg",610,205,592,350,C.pale,C.pale);
  text(s,"record-right-head","For AK candidates, add one outcome",642,238,520,40,26,{bold:true});
  bulletList(s,"record-right",["acceptable_new_variation","related_but_incorrect","incorrect_unrelated","ocr_error","uncertain"],642,305,500,220,22,"\n\n");
  text(s,"record-footer","These labels are required before AK-gap logic can be trained and validated rather than operated as rules.",42,610,1150,42,23,{bold:true});
  notes(s,"Timing: 1.5 minutes. Agree exact field names and where they will be stored. The most valuable new information is the accepted/rejected AK outcome, because it does not exist in the current training data.",["OCR_REVIEW.md", "ocr_review.py: build_scoring_report()"]);
}

// 13 — monitoring measures
{
  const s=p.slides.add();s.background.fill=C.white;title(s,"Monitoring must cover quality, workload and drift",13);
  const cols=[42,444,846], heads=["Each run","Weekly operations","After verified samples"], bodies=[
    ["Model hash + thresholds","Rows and eligible AK rows","OCR / AK / combined flags","Risk and reason counts","Score distributions"],
    ["Queue size and ageing","Reviews completed","Reviewer time","AK escalations and approvals","Blocked-row QC findings"],
    ["OCR correction recall","Review precision / yield","Low-risk miss rate","AK suggestion acceptance rate","Results by question / segment"]
  ];
  cols.forEach((x,i)=>{text(s,`mon-h-${i}`,heads[i],x,205,350,42,26,{bold:true});box(s,`mon-line-${i}`,x,260,350,4,i===0?C.blue:C.rule,i===0?C.blue:C.rule);bulletList(s,`mon-b-${i}`,bodies[i],x,295,350,290,20,"\n\n");});
  box(s,"mon-warning",42,620,1150,52,"#FFF0CC","#FFF0CC");text(s,"mon-warning-t","Unreviewed scoring data can report counts and rates—but not recall or precision without verified Published truth.",62,635,1110,24,20,{bold:true});
  notes(s,"Timing: 2 minutes. Separate what the pipeline can log automatically from what needs reviewed ground truth. Recommend a regular verified sample so recall, precision and AK acceptance can be estimated over time.",["output/hitl/no_human_flagged.scoring_report.json", "ocr_review.py: build_scoring_report()"]);
}

// 14 — current run
{
  const s=p.slides.add();s.background.fill=C.white;title(s,"The latest run routes 3.96% of rows to human review",14);
  const stats=[["138,810","rows scored"],["5,477","OCR flags"],["193","AK-gap flags"],["5,501","combined review rows"]];
  stats.forEach((d,i)=>{const x=42+i*298;box(s,`run-stat-bg-${i}`,x,205,260,175,i===3?C.pale:C.panel,i===3?C.pale:C.panel);text(s,`run-stat-${i}`,d[0],x+22,235,216,65,42,{bold:true});text(s,`run-l-${i}`,d[1],x+22,318,216,36,19);});
  text(s,"run-why","Why combined is not OCR + AK",42,445,450,38,26,{bold:true});
  bulletList(s,"run-detail",["169 rows triggered both decisions.","24 rows were AK-only suggestions.","5,308 rows were OCR-only flags.","11,230 semantic candidates were blocked by conflict rules."],42,505,540,140,20,"\n");
  box(s,"run-action",650,445,552,182,C.white,C.ink);text(s,"run-action-t","Operational implication\n\nMost workload is OCR review. AK review is smaller, but it needs a distinct decision form and approval route.",680,475,492,125,22,{bold:true});
  notes(s,"Timing: 1.5 minutes. Use these figures to make the workload concrete. Explain overlap: requires_any_human_review de-duplicates rows that trigger both decisions.",["output/hitl/no_human_flagged.scoring_report.json generated 2026-08-28"]);
}

// 15 — operating agreement
{
  const s=p.slides.add();s.background.fill=C.white;title(s,"Agree the operating controls before routine monitoring begins",15);
  const items=[
    ["Queue ownership","Who works OCR, AK-only and combined cases?"],
    ["Service level","How quickly must high, medium and AK escalations be reviewed?"],
    ["Quality control","What sample of low-risk and blocked rows is independently checked?"],
    ["Change authority","Who can change thresholds, approve model versions or update the AK?"],
    ["Feedback data","Where are reviewer outcomes stored and quality checked?"],
    ["Review cadence","When are workload, quality and drift measures discussed?"]
  ];
  items.forEach((d,i)=>{const col=i%2,row=Math.floor(i/2);const x=42+col*600,y=195+row*135;text(s,`close-h-${i}`,d[0],x,y,260,35,24,{bold:true});text(s,`close-b-${i}`,d[1],x+285,y,280,70,18);s.shapes.add({geometry:"straightConnector1",name:`close-r-${i}`,position:{left:x,top:y+90,width:560,height:0},fill:"none",line:{style:"solid",fill:C.rule,width:1}});});
  box(s,"close-final",42,610,1160,65,C.pale,C.pale);text(s,"close-final-t","Success means fewer missed OCR errors, a manageable queue and traceable answer-key improvements—not automatic marking.",65,629,1110,30,22,{bold:true,align:"center"});
  notes(s,"Timing: 1.5 minutes plus discussion. Close by assigning owners and dates for the six controls. Invite the team to identify gaps in the proposed reviewer screen and outcome fields.",["OCR_REVIEW.md", "output/hitl/OCR_Human_Review_Model_Leader_Brief.docx"]);
}

await fs.mkdir("/Users/ranshanzi/Documents/VSCode/data_analysis/output/hitl", { recursive: true });
const file = await PresentationFile.exportPptx(p);
await file.save(OUT);
console.log(OUT);
