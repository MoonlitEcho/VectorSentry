const pptxgen = require("pptxgenjs");

// ---------- Palette: same as Review I deck, for visual continuity ----------
const NAVY = "1B2A56";
const NAVY_DARK = "111B38";
const ICE = "CFE0FA";
const TEAL = "0E7C7B";
const TEAL_LIGHT = "E4F3F2";
const WHITE = "FFFFFF";
const INK = "20242E";
const MUTED = "5A6472";
const CARD = "F4F7FC";
const AMBER = "C97A22";
const GOOD = "1E824C";
const BAD = "B3261E";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5 in
const PGW = 13.333, PGH = 7.5;

const RES = "../phase2_proposed/results/"; // relative to this scratch dir

// ---------- helpers (same design system as Review I) ----------
function darkBg(slide) { slide.background = { color: NAVY }; }
function lightBg(slide) { slide.background = { color: WHITE }; }

function footer(slide, pageNo, dark) {
  slide.addText("VectorSentry  |  Review II", {
    x: 0.5, y: PGH - 0.42, w: 7, h: 0.3, fontSize: 9, color: dark ? "8CA0C8" : MUTED,
    fontFace: "Calibri", isTextBox: true, margin: 0,
  });
  slide.addText(String(pageNo), {
    x: PGW - 0.9, y: PGH - 0.42, w: 0.5, h: 0.3, fontSize: 9, color: dark ? "8CA0C8" : MUTED,
    fontFace: "Calibri", align: "right", isTextBox: true, margin: 0,
  });
}

function sectionTag(slide, text, dark) {
  slide.addText(text.toUpperCase(), {
    x: 0.6, y: 0.42, w: 11, h: 0.35, fontSize: 12.5, bold: true, color: dark ? ICE : TEAL,
    fontFace: "Calibri", charSpacing: 2, isTextBox: true, margin: 0,
  });
}

function title(slide, text, dark) {
  slide.addText(text, {
    x: 0.6, y: 0.75, w: 11.6, h: 0.75, fontSize: 32, bold: true, color: dark ? WHITE : NAVY,
    fontFace: "Cambria", isTextBox: true, margin: 0,
  });
}

function pillNumber(slide, x, y, n, big) {
  const d = big ? 0.55 : 0.42;
  slide.addShape("ellipse", { x, y, w: d, h: d, fill: { color: TEAL }, line: { type: "none" } });
  slide.addText(String(n), {
    x, y, w: d, h: d, align: "center", valign: "middle", fontSize: big ? 20 : 15, bold: true,
    color: WHITE, fontFace: "Calibri", isTextBox: true, margin: 0,
  });
}

function card(slide, x, y, w, h, opts) {
  opts = opts || {};
  slide.addShape("roundRect", {
    x, y, w, h, rectRadius: 0.08,
    fill: { color: opts.fill || CARD },
    line: { type: "none" },
    shadow: opts.noShadow ? undefined : { type: "outer", color: "1B2A56", opacity: 0.12, blur: 8, offset: 3, angle: 90 },
  });
}

function statusPill(slide, x, y, text, color) {
  const w = 0.24 + text.length * 0.082;
  slide.addShape("roundRect", { x, y, w, h: 0.32, rectRadius: 0.16, fill: { color }, line: { type: "none" } });
  slide.addText(text, { x, y, w, h: 0.32, align: "center", valign: "middle", fontSize: 9, bold: true, color: WHITE, fontFace: "Calibri", isTextBox: true, margin: 0 });
}

// =====================================================================
// SLIDE 1 — TITLE
// =====================================================================
{
  const s = pres.addSlide();
  darkBg(s);
  s.addShape("rect", { x: 0, y: 0, w: PGW, h: PGH, fill: { color: NAVY }, line: { type: "none" } });
  s.addShape("ellipse", { x: 9.6, y: -2.2, w: 6.5, h: 6.5, fill: { color: NAVY_DARK }, line: { type: "none" } });
  s.addShape("ellipse", { x: 10.6, y: 4.6, w: 4.2, h: 4.2, fill: { color: TEAL }, line: { type: "none" } });

  s.addText("KUMARAGURU COLLEGE OF TECHNOLOGY  •  KSI", {
    x: 0.7, y: 0.55, w: 8, h: 0.35, fontSize: 12, color: "8CA0C8", fontFace: "Calibri",
    charSpacing: 2, isTextBox: true, margin: 0,
  });

  s.addText("VectorSentry", {
    x: 0.7, y: 2.25, w: 10.5, h: 1.1, fontSize: 54, bold: true, color: WHITE, fontFace: "Cambria",
    isTextBox: true, margin: 0,
  });
  s.addText("Budget-Aware Prioritization for Mixed-Generation Embedding\nContamination in Production Vector Databases", {
    x: 0.72, y: 3.15, w: 10.3, h: 0.85, fontSize: 19, color: ICE, fontFace: "Calibri",
    isTextBox: true, margin: 0, lineSpacingMultiple: 1.15,
  });
  s.addText("Review II — Built, Tested, and Evaluated", {
    x: 0.72, y: 3.98, w: 10.3, h: 0.4, fontSize: 17, bold: true, italic: true, color: TEAL_LIGHT, fontFace: "Calibri",
    isTextBox: true, margin: 0,
  });

  card(s, 0.7, 4.6, 8.6, 1.55, { fill: "16224A", noShadow: true });
  s.addText([
    { text: "Team Member(s): ", options: { bold: true, color: ICE } },
    { text: "[Full Name — Register No. — Department]  (add one line per member)\n", options: { color: WHITE } },
    { text: "Guide: ", options: { bold: true, color: ICE } },
    { text: "[Guide Name, Designation]\n", options: { color: WHITE } },
    { text: "Review Date: ", options: { bold: true, color: ICE } },
    { text: "[Confirm exact review date]   |   UG Capstone Projects (2026-27) — Phase I, Review II", options: { color: WHITE } },
  ], { x: 0.95, y: 4.8, w: 8.1, h: 1.2, fontSize: 13.5, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.35 });

  s.addText("Fields in brackets, and the guide-approval proof required by the review circular, must be completed before presenting.", {
    x: 0.7, y: 6.3, w: 8.7, h: 0.4, fontSize: 10.5, italic: true, color: AMBER, fontFace: "Calibri", isTextBox: true, margin: 0,
  });

  s.addNotes(
    "Good [morning/afternoon]. I'm presenting VectorSentry's Review II — since Review I, we've moved from a proposal to a built, tested, and evaluated system. " +
    "Today I'll briefly respond to the feedback from Review I, then walk through the literature, the built architecture, real experimental results, how we tested it, and a live demonstration of the system actually running."
  );
}

// =====================================================================
// SLIDE 2 — RESPONSE TO REVIEW I FEEDBACK
// =====================================================================
{
  const s = pres.addSlide();
  lightBg(s);
  sectionTag(s, "Response to Review I Feedback", false);
  title(s, "Addressing the Panel's Questions", false);
  s.addText("Three questions from Review I, answered directly — with evidence, not just argument.", {
    x: 0.6, y: 1.42, w: 12, h: 0.3, fontSize: 11, italic: true, color: MUTED, fontFace: "Calibri", isTextBox: true, margin: 0,
  });

  const qs = [
    ["Q1", "“Why is there a paper from the 1900s?”",
     "Sinha & Zoltners (1979) and Breslau et al. (1999) are foundational theory, not stale trivia — the way citing Dijkstra (1959) for shortest-path algorithms wouldn't be “old.” We don't just cite the 1979 knapsack formulation — our scheduler is a direct implementation of it, and we measured it: 0.00% mean optimality gap (max 0.14%) against the exact solver, on the exact instance where a naive approach was 88% wrong. The 45-year-old math holds up empirically, today, on our own data."],
    ["Q2", "“Doesn't this need a lot of overhead?”",
     "Measured, not assumed: Detection scans the full 5,000-document benchmark in under 0.6 seconds (real measured runtime, all three detectors). Migration only shadow-builds the documents actually selected for a fix — not the whole corpus. Verification re-scores a small curated golden-query set, not the full query load. Full breakdown on the Complexity slide next."],
    ["Q3", "“What's the space and time complexity?”",
     "Detection: O(N log N) with an ANN index. Prioritization: O(N log N) for production (frontier-greedy); the exact ILP solver is intentionally kept off the live path, used only for small-scale validation. Migration/Verification: bounded by k (flagged documents), not N (corpus size). Full table, with real measured wall-clock numbers, coming up shortly."],
  ];
  let y = 1.85;
  qs.forEach((q) => {
    card(s, 0.6, y, 12.15, 1.55, { fill: NAVY, noShadow: true });
    pillNumber(s, 0.85, y + 0.2, q[0].slice(1), true);
    s.addText(q[1], { x: 1.65, y: y + 0.15, w: 10.9, h: 0.4, fontSize: 13.5, bold: true, color: TEAL_LIGHT, fontFace: "Calibri", isTextBox: true, margin: 0 });
    s.addText(q[2], { x: 1.65, y: y + 0.55, w: 10.9, h: 0.92, fontSize: 10.5, color: WHITE, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.18 });
    y += 1.68;
  });

  footer(s, 2, false);
  s.addNotes(
    "Before anything else, I want to directly answer the three questions the panel raised at Review I, because we took them seriously and they shaped what we prioritized building. " +
    "On the old papers: these are foundational theory, and we didn't just cite the 1979 knapsack paper — we validated it. Our scheduler, built directly on that formulation, measures a 0.00% mean optimality gap against the exact solver, with a max of 0.14%, on exactly the instance where a naive approach was 88% wrong. " +
    "On overhead: we now have real numbers. Detection scans the full five-thousand-document benchmark in under 0.6 seconds. Migration only touches the documents actually selected for a fix, not the whole corpus. " +
    "On complexity: full breakdown is next, with real measured wall-clock times alongside the theoretical bounds."
  );
}

// =====================================================================
// SLIDE 3 — PROBLEM STATEMENT & MOTIVATION (condensed recap)
// =====================================================================
{
  const s = pres.addSlide();
  lightBg(s);
  sectionTag(s, "Recap — Problem Statement", false);
  title(s, "The Problem, Briefly Revisited", false);

  s.addText(
    "Production vector databases silently degrade in retrieval quality when a subset of documents is embedded by a different generation of the embedding model than the rest of the corpus. Cosine similarity is not comparable across generations, so the database keeps returning results — just quietly worse ones, with no exception ever thrown.",
    { x: 0.6, y: 1.65, w: 12.15, h: 1.2, fontSize: 14, color: INK, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.3 }
  );

  const cols = [
    ["Real", "OpenAI and Cohere have both shipped embedding updates that change internal geometry; teams re-embedding piecemeal create exactly this mix."],
    ["Silent", "No exceptions, no alerts, no crash logs — the system looks healthy while quietly serving worse results."],
    ["Growing", "20+ commercial vector-database systems have launched in the past five years (Pan et al., 2024) — every one exposed to this failure mode."],
  ];
  let cx = 0.6;
  const cw = 3.98;
  cols.forEach((c) => {
    card(s, cx, 3.1, cw, 2.0);
    s.addText(c[0], { x: cx + 0.25, y: 3.3, w: cw - 0.5, h: 0.4, fontSize: 15, bold: true, color: TEAL, fontFace: "Calibri", isTextBox: true, margin: 0 });
    s.addText(c[1], { x: cx + 0.25, y: 3.7, w: cw - 0.5, h: 1.25, fontSize: 10.5, color: INK, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.2 });
    cx += cw + 0.19;
  });

  card(s, 0.6, 5.35, 12.15, 1.3, { fill: NAVY, noShadow: true });
  s.addText([
    { text: "Since Review I:  ", options: { bold: true, color: TEAL_LIGHT } },
    { text: "we've moved from arguing this problem to measuring it. Our own 5,000-document benchmark (detailed ahead) now quantifies exactly how retrieval quality degrades as contamination rises — real data, not projection.", options: { color: WHITE } },
  ], { x: 0.9, y: 5.5, w: 11.5, h: 1.0, fontSize: 12, fontFace: "Calibri", valign: "middle", isTextBox: true, margin: 0, lineSpacingMultiple: 1.25 });

  footer(s, 3, false);
  s.addNotes(
    "Quick recap for context, since you saw this in depth at Review I: mixed-generation embedding contamination silently degrades retrieval quality, because cosine similarity isn't comparable across two different embedding-model generations. " +
    "It's real — documented in OpenAI and Cohere's own update history. It's silent — no exceptions, no alerts. And it's growing — over twenty commercial vector-database systems have launched in the last five years, all exposed to this. " +
    "The key difference from Review I: we've now built the benchmark and measured this directly, which is what the rest of today's presentation is about."
  );
}

// =====================================================================
// SLIDE 4 — RESEARCH GAP IDENTIFICATION (recap)
// =====================================================================
{
  const s = pres.addSlide();
  lightBg(s);
  sectionTag(s, "Research Gap Identification", false);
  title(s, "From Literature to Research Gap", false);

  const chain = [
    ["Existing research", "Two mature, separate bodies of work: OOD/anomaly detection and cheap embedding realignment."],
    ["Common approach", "Detect anomalous vectors → apply ONE fix (full re-embed or one uniform patch) to everyone flagged."],
    ["Limitation", "Every reviewed fix mechanism is corpus-wide and uniform — none reasons about WHICH document to fix first under a real budget."],
    ["Research gap", "No budget-aware, traffic-informed, per-document prioritization layer exists between 'detect' and 'fix' — and no safety gate ensures a chosen plan doesn't regress production quality before it ships."],
  ];
  let y = 1.7;
  const rh = 0.95;
  chain.forEach((c, i) => {
    const isGap = i === chain.length - 1;
    card(s, 0.6, y, 8.5, rh, { fill: isGap ? NAVY : CARD, noShadow: isGap });
    pillNumber(s, 0.85, y + (rh - 0.42) / 2, i + 1, false);
    s.addText(c[0], { x: 1.5, y: y + 0.1, w: 2.1, h: rh - 0.2, fontSize: 12.5, bold: true, color: isGap ? WHITE : NAVY, fontFace: "Calibri", valign: "middle", isTextBox: true, margin: 0 });
    s.addText(c[1], { x: 3.7, y: y + 0.08, w: 5.3, h: rh - 0.16, fontSize: 11, color: isGap ? ICE : INK, fontFace: "Calibri", valign: "middle", isTextBox: true, margin: 0, lineSpacingMultiple: 1.15 });
    if (i < chain.length - 1) {
      s.addText("↓", { x: 0.6, y: y + rh - 0.05, w: 0.5, h: 0.25, fontSize: 14, color: TEAL, fontFace: "Calibri", align: "center", isTextBox: true, margin: 0 });
    }
    y += rh + 0.16;
  });

  card(s, 9.3, 1.7, 3.45, 4.85, { fill: TEAL_LIGHT });
  s.addText("OUR CONTRIBUTION", { x: 9.55, y: 1.9, w: 3, h: 0.3, fontSize: 11, bold: true, color: TEAL, fontFace: "Calibri", charSpacing: 1.5, isTextBox: true, margin: 0 });
  s.addText(
    "A Multiple-Choice Knapsack formulation — {Full Re-embed, Patch, Defer} per document, one shared budget — solved exactly (ILP) and approximately (greedy), gated by automated shadow-build + verification.",
    { x: 9.55, y: 2.3, w: 3, h: 1.9, fontSize: 11, color: NAVY, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.25 }
  );
  statusPill(s, 9.55, 4.35, "NOW BUILT & MEASURED", GOOD);
  s.addText("Every claim on this slide is now backed by a real, measured experiment, not just a design argument — see the numbers cited throughout this deck.", {
    x: 9.55, y: 4.85, w: 3, h: 1.6, fontSize: 9.5, italic: true, color: MUTED, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.2,
  });

  footer(s, 4, false);
  s.addNotes(
    "This is the same research-gap logic from Review I, kept because it's still the spine of the project: detection literature finds anomalies but doesn't say what to do next; patching literature fixes cheaply but uniformly, with no per-document decision under a budget. " +
    "That gap is exactly a Multiple-Choice Knapsack Problem — which is what we've now actually built and measured, not just proposed."
  );
}

// =====================================================================
// SLIDE 5 — DETAILED LITERATURE REVIEW: DETECTION & PATCHING
// =====================================================================
{
  const s = pres.addSlide();
  lightBg(s);
  sectionTag(s, "Detailed Literature Review — 1 of 2", false);
  title(s, "Literature Review — Detection & Patching", false);
  s.addText("7 of 15 sources — the papers directly targeting detection or repair of embedding contamination.", {
    x: 0.6, y: 1.4, w: 12, h: 0.3, fontSize: 10, italic: true, color: MUTED, fontFace: "Calibri", isTextBox: true, margin: 0,
  });

  const head1 = ["Source", "Core Concept", "Gap", "How We Address It"];
  const data1 = [
    ["Sun et al., 2022", "Non-parametric OOD detection via k-NN distance in embedding space", "Detects anomalies only — doesn't decide what to do next", "Adapted as our Detection module's core scoring method — now built and measured (see Complexity slide)"],
    ["Lu et al., 2024", "Taxonomy of OOD-detection methods across ML domains", "No treatment of embedding-generation-drift-specific detection", "Confirms KNN-OOD as a mature, well-grounded detector choice"],
    ["Liu et al., 2008", "Isolation Forest — isolates anomalies via random feature-space splits", "General-purpose; untested on embedding-space anisotropy", "Evaluated as an alternative detector; F1 0.11–0.45, weaker than KNN-OOD (real, measured)"],
    ["Breunig et al., 2000", "LOF — density-relative local outlier scoring", "Same generality gap; heavier compute than KNN-OOD", "Second alternative detector; F1 0.20–0.51, weaker than KNN-OOD (real, measured)"],
    ["Vejendla, 2025", "Drift-Adapter — lightweight learned adapter cheaply realigns embeddings", "One uniform fix for the whole corpus — no per-document choice", "Adapted as our Patch action, gated by a budget decision"],
    ["Hu et al., 2022", "Trains new models to stay backward-compatible with old embeddings", "Needs control of model training — unusable with third-party APIs", "Discussed as a complementary, prevention-side alternative"],
    ["Lample et al., 2018", "MUSE — orthogonal Procrustes alignment between spaces, no parallel data", "Built for cross-lingual spaces; assumes no budget constraint", "Theoretical foundation for our Patch action's alignment transform"],
  ];

  s.addTable(
    [head1.map((h) => ({ text: h, options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 9.5 } }))].concat(
      data1.map((row) => row.map((c) => ({ text: c, options: { fontSize: 8.1, color: INK } })))
    ),
    {
      x: 0.6, y: 1.8, w: 12.15, h: 5.15,
      colW: [1.85, 3.05, 3.05, 4.2],
      border: { type: "solid", color: "D8DEEA", pt: 0.75 },
      autoPage: false,
      valign: "middle",
      margin: [3, 6, 3, 6],
    }
  );

  footer(s, 5, false);
  s.addNotes(
    "These seven sources directly target detecting or repairing embedding contamination. KNN-OOD, Isolation Forest, and LOF detect anomalies but never say what to do once you've found one — we've now built and measured all three as candidate detectors; KNN-OOD's F1 of 0.51 to 0.57 is why it's our production choice, as noted in the table. " +
    "Drift-Adapter and the Backward-Compatible-Embeddings paper both show cheap realignment or avoidance is possible, but each applies one uniform strategy — our Patch action adapts Drift-Adapter's mechanism, gated by our own budget decision. MUSE gives us the underlying alignment math."
  );
}

// =====================================================================
// SLIDE 6 — DETAILED LITERATURE REVIEW: FOUNDATIONAL WORK
// =====================================================================
{
  const s = pres.addSlide();
  lightBg(s);
  sectionTag(s, "Detailed Literature Review — 2 of 2", false);
  title(s, "Literature Review — Foundational Work", false);
  s.addText("8 of 15 sources — systems, benchmarking, and Operations-Research foundations our design is built on.", {
    x: 0.6, y: 1.4, w: 12, h: 0.3, fontSize: 10, italic: true, color: MUTED, fontFace: "Calibri", isTextBox: true, margin: 0,
  });

  const head2 = ["Source", "Core Concept", "Gap", "How We Address It"];
  const data2 = [
    ["Malkov & Yashunin, 2020", "HNSW — graph index for fast approximate nearest-neighbor search", "Assumes a geometrically consistent index — silently broken by contamination", "Names exactly why contamination is dangerous for the indexes we target"],
    ["Pan et al., 2024", "Survey of 20+ vector-database system architectures", "No treatment of post-deployment embedding integrity", "Evidences real-world scale and relevance of the problem"],
    ["Muennighoff et al., 2023", "MTEB — unified benchmark for comparing embedding models", "Static snapshot; doesn't test mixed-generation drift scenarios", "Explains why models churn often — the practical trigger for our problem"],
    ["Wu et al., 2024", "Quantifies how retrieval quality affects RAG faithfulness", "Doesn't address the retrieval-quality-degradation cause itself", "Motivates why contamination matters beyond raw metrics"],
    ["Fuster Baggetto & Fresno, 2022", "Tests whether embedding anisotropy actually harms retrieval", "Studied within one generation, not across mixed generations", "Motivated our own mixed-generation anisotropy stress test (built — see results)"],
    ["Sinha & Zoltners, 1979", "Exact formulation for one-choice-per-group allocation under budget", "Purely theoretical — never applied to embedding remediation", "Mathematical core of Prioritization — now measured at 0.00% mean gap (max 0.14%)"],
    ["Breslau et al., 1999", "Real web traffic follows a Zipf (highly skewed) distribution", "About web objects, not vector-DB query traffic specifically", "Grounds the traffic-weighted benefit function used in our built scheduler"],
    ["Karval & Singh, 2023", "Names 'silent failure' as a distinct, under-monitored ML risk category", "General MLOps treatment — not specific to vector databases", "Provides the framing our Problem Statement applies directly"],
  ];

  s.addTable(
    [head2.map((h) => ({ text: h, options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 9.5 } }))].concat(
      data2.map((row) => row.map((c) => ({ text: c, options: { fontSize: 7.7, color: INK } })))
    ),
    {
      x: 0.6, y: 1.8, w: 12.15, h: 4.55,
      colW: [1.85, 3.05, 3.05, 4.2],
      border: { type: "solid", color: "D8DEEA", pt: 0.75 },
      autoPage: false,
      valign: "middle",
      margin: [3, 5, 3, 5],
    }
  );

  card(s, 0.6, 6.5, 12.15, 0.65, { fill: NAVY, noShadow: true });
  s.addText([
    { text: "On the two highlighted rows above (1979, 1999): ", options: { bold: true, color: TEAL_LIGHT } },
    { text: "foundational theorems don't expire. We cite them because we implement them — and Sinha & Zoltners' 45-year-old formulation now measures at 0.00% mean optimality gap in our own built scheduler.", options: { color: WHITE } },
  ], { x: 0.85, y: 6.5, w: 11.65, h: 0.65, fontSize: 10, fontFace: "Calibri", valign: "middle", isTextBox: true, margin: 0, lineSpacingMultiple: 1.15 });

  footer(s, 6, false);
  s.addNotes(
    "These eight are our foundation layer. HNSW explains exactly why contamination is dangerous — it assumes a consistent index geometry that contamination silently breaks. " +
    "The two rows the Review I panel specifically asked about are Sinha and Zoltners, 1979, and Breslau and colleagues, 1999. We're not citing them as historical curiosities — our scheduler is a direct implementation of the 1979 formulation, and we've now measured it: 0.00% mean optimality gap, max 0.14%, against the exact solver. The math holds up, today, on our own data."
  );
}

// =====================================================================
// SLIDE 7 — DATASET DETAILS
// =====================================================================
{
  const s = pres.addSlide();
  lightBg(s);
  sectionTag(s, "Dataset Details", false);
  title(s, "Dataset & Benchmark Construction", false);

  const left = [
    ["Corpus", "BEIR SciFact — a real, peer-reviewed IR benchmark with genuine relevance judgments (qrels), not synthetic labels."],
    ["Scale", "5,000 documents — a deliberate subsample that keeps the exact-ILP baseline solvable for direct optimality comparison, while remaining a meaningful evaluation."],
    ["Embedding generations", "Generation A = all-MiniLM-L6-v2, Generation B = all-MiniLM-L12-v2 — two real, open sentence-transformer models, standing in for 'old' and 'new' provider versions."],
    ["Reproducibility", "Fixed seed (42) across every experiment; fully scripted corpus construction, contamination injection, and evaluation — deterministic and re-runnable."],
  ];
  let ly = 1.7;
  left.forEach((r) => {
    card(s, 0.6, ly, 6.6, 1.28);
    s.addText(r[0], { x: 0.85, y: ly + 0.12, w: 6.1, h: 0.3, fontSize: 12.5, bold: true, color: TEAL, fontFace: "Calibri", isTextBox: true, margin: 0 });
    s.addText(r[1], { x: 0.85, y: ly + 0.46, w: 6.1, h: 0.75, fontSize: 10, color: INK, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.18 });
    ly += 1.42;
  });

  card(s, 7.55, 1.7, 5.2, 5.68, { fill: NAVY, noShadow: true });
  s.addText("EXPERIMENTAL SWEEP", { x: 7.85, y: 1.9, w: 4.6, h: 0.3, fontSize: 11, bold: true, color: ICE, fontFace: "Calibri", charSpacing: 1.5, isTextBox: true, margin: 0 });
  const sweep = [
    ["Contamination ratios tested", "5%, 10%, 20%, 35%, 50%"],
    ["Remediation budget levels tested", "10%, 20%, 30%, 50%, 75%, 100%"],
    ["Query traffic model", "Zipf-distributed synthetic traffic (Breslau et al., 1999) — a small fraction of documents receive most simulated query volume"],
    ["Evaluation metrics", "nDCG@10 and Recall@10, both unweighted and traffic-weighted"],
    ["Total evaluated combinations", "5 ratios × 6 budget levels = 30 scenarios per method, run to completion"],
  ];
  let sy = 2.3;
  sweep.forEach((r) => {
    s.addText(r[0], { x: 7.85, y: sy, w: 4.6, h: 0.28, fontSize: 11, bold: true, color: WHITE, fontFace: "Calibri", isTextBox: true, margin: 0 });
    s.addText(r[1], { x: 7.85, y: sy + 0.3, w: 4.6, h: 0.62, fontSize: 9.5, color: ICE, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.15 });
    sy += 0.98;
  });

  footer(s, 7, false);
  s.addNotes(
    "Dataset details, as required. We use BEIR SciFact — a real, peer-reviewed IR benchmark with genuine relevance judgments, not something we invented. Five thousand documents keeps the exact-ILP baseline solvable for direct comparison while remaining meaningful. " +
    "Contamination is introduced by re-embedding a controlled subset with a second real embedding model — all-MiniLM-L12-v2 standing in for a 'new' provider generation against all-MiniLM-L6-v2 as the 'old' one. " +
    "We swept five contamination ratios against six budget levels — thirty scenarios per method, all run to completion, with a fixed seed for full reproducibility. Query traffic is modeled as Zipf-distributed, directly grounded in the Breslau et al. 1999 paper from our literature review — not an arbitrary choice."
  );
}

// =====================================================================
// SLIDE 8 — SYSTEM ARCHITECTURE AND WORKFLOW
// =====================================================================
{
  const s = pres.addSlide();
  lightBg(s);
  sectionTag(s, "System Architecture and Workflow", false);
  title(s, "System Architecture & Workflow", false);
  statusPill(s, 11.0, 0.8, "BUILT & RUNNING", GOOD);

  const stages = [
    ["Live Vector\nDatabase", CARD, INK],
    ["Detect", TEAL, WHITE],
    ["Prioritize", NAVY, WHITE],
    ["Migrate", AMBER, WHITE],
    ["Verify", BAD, WHITE],
  ];
  const y0 = 1.85, bw = 1.95, bh = 1.0, gap = 0.42;
  let x0 = 0.6;
  stages.forEach((st, i) => {
    s.addShape("roundRect", { x: x0, y: y0, w: bw, h: bh, rectRadius: 0.09, fill: { color: st[1] }, line: { type: "none" } });
    s.addText(st[0], { x: x0, y: y0, w: bw, h: bh, align: "center", valign: "middle", fontSize: 13, bold: true, color: st[2], fontFace: "Calibri", isTextBox: true, margin: 0 });
    if (i < stages.length - 1) {
      s.addText("→", { x: x0 + bw, y: y0, w: gap, h: bh, align: "center", valign: "middle", fontSize: 20, bold: true, color: NAVY, fontFace: "Calibri", isTextBox: true, margin: 0 });
    }
    x0 += bw + gap;
  });

  s.addShape("roundRect", { x: x0, y: y0, w: 2.0, h: bh, rectRadius: 0.09, fill: { color: GOOD }, line: { type: "none" } });
  s.addText("Cutover\n(live)", { x: x0, y: y0, w: 2.0, h: bh, align: "center", valign: "middle", fontSize: 12.5, bold: true, color: WHITE, fontFace: "Calibri", isTextBox: true, margin: 0 });
  s.addText("pass", { x: x0 - 0.2, y: y0 - 0.32, w: 1, h: 0.28, fontSize: 9.5, italic: true, color: GOOD, fontFace: "Calibri", isTextBox: true, margin: 0 });
  s.addShape("line", { x: 5.42, y: y0 + bh + 0.3, w: 0, h: 0.55, line: { color: BAD, width: 1.25, dashType: "dash" } });
  s.addShape("roundRect", { x: 4.45, y: y0 + bh + 0.85, w: 1.95, h: 0.7, rectRadius: 0.09, fill: { color: "F6E3E1" }, line: { color: BAD, width: 1 } });
  s.addText("Rollback\n(unchanged)", { x: 4.45, y: y0 + bh + 0.85, w: 1.95, h: 0.7, align: "center", valign: "middle", fontSize: 11, bold: true, color: BAD, fontFace: "Calibri", isTextBox: true, margin: 0 });
  s.addText("fail — retried\nor deferred", { x: 6.5, y: y0 + bh + 0.85, w: 1.7, h: 0.7, fontSize: 9, italic: true, color: BAD, fontFace: "Calibri", valign: "middle", isTextBox: true, margin: 0, lineSpacingMultiple: 1.1 });

  s.addShape("bentConnector3", { x: 1.5, y: y0 + bh + 0.1, w: x0 + 1.0 - 1.5, h: 1.4, line: { color: NAVY, width: 1.25, dashType: "dash" }, flipV: true });
  s.addText("continuous autonomous monitoring — \"Live Sentry\" module — built and validated end-to-end on the real benchmark", {
    x: 0.6, y: y0 + bh + 1.45, w: 9.5, h: 0.3, fontSize: 10, italic: true, color: NAVY, fontFace: "Calibri", isTextBox: true, margin: 0,
  });

  const desc = [
    ["Detect", "KNN-OOD score vs. a stable-baseline-calibrated threshold — no oracle labels used. Built."],
    ["Prioritize", "MCKP scheduler: severity × traffic × recovery, one shared budget, exact ILP + frontier greedy. Built."],
    ["Migrate", "Shadow-build the selected fix — the live index is never mutated in place. Built."],
    ["Verify", "Golden-query canary check against the shadow, with automatic rollback on failure. Built."],
  ];
  let dy = 4.15;
  desc.forEach((d) => {
    card(s, 0.6, dy, 12.15, 0.58);
    s.addText([{ text: d[0] + ":  ", options: { bold: true, color: TEAL } }, { text: d[1], options: { color: INK } }], {
      x: 0.85, y: dy, w: 11.6, h: 0.58, fontSize: 11, fontFace: "Calibri", valign: "middle", isTextBox: true, margin: 0,
    });
    dy += 0.68;
  });

  footer(s, 8, false);
  s.addNotes(
    "This is the architecture as built and currently running — not a proposal anymore. A live vector database feeds Detect, which scores documents against a baseline-calibrated threshold. Prioritize is our knapsack scheduler, deciding full re-embed, patch, or defer for every flagged document. " +
    "Migrate builds the fix as a shadow copy — the live index is never touched directly. Verify is the safety gate: golden-query canary check, automatic rollback on failure, retry or defer as appropriate. " +
    "And the whole loop runs autonomously and continuously in our 'Live Sentry' module — built, integrated, and validated end-to-end on the real 5,000-document benchmark."
  );
}

// =====================================================================
// SLIDE 9 — PROPOSED METHODOLOGY AND SOLUTION
// =====================================================================
{
  const s = pres.addSlide();
  lightBg(s);
  sectionTag(s, "Proposed Methodology and Solution", false);
  title(s, "Methodology — Module Breakdown", false);
  s.addText("Four modules, each with a defined input/output contract — every one implemented, unit-tested, and integrated.", {
    x: 0.6, y: 1.42, w: 11.8, h: 0.32, fontSize: 11, italic: true, color: MUTED, fontFace: "Calibri", isTextBox: true, margin: 0,
  });

  const mods = [
    ["M1", "Detection", "KNN-OOD scoring vs. a stable-baseline percentile threshold. Evaluated against LOF and Isolation Forest as alternatives.", "F1 = 0.51–0.57 across contamination ratios (real, measured)"],
    ["M2", "Prioritization", "MCKP formulation — exact ILP for validation, frontier-greedy for production scale.", "0.00% mean optimality gap, max 0.14% (real, measured)"],
    ["M3", "Migration", "Shadow-builds only the selected documents; live index never mutated in place.", "10/30 scenarios correctly aborted by the safety gate (real, measured)"],
    ["M4", "Verification & Cutover", "Golden-query canary evaluation; atomic cutover on pass, automatic rollback/retry/defer on fail.", "Beats best Phase-1 baseline in 18/30 scenarios (real, measured)"],
  ];
  let mx = 0.6;
  const mw = 2.9;
  mods.forEach((m) => {
    card(s, mx, 2.0, mw, 4.65, { fill: mx === 0.6 ? NAVY : CARD, noShadow: mx === 0.6 });
    s.addShape("roundRect", { x: mx + 0.22, y: 2.22, w: 0.6, h: 0.6, rectRadius: 0.3, fill: { color: TEAL }, line: { type: "none" } });
    s.addText(m[0], { x: mx + 0.22, y: 2.22, w: 0.6, h: 0.6, align: "center", valign: "middle", fontSize: 15, bold: true, color: WHITE, fontFace: "Calibri", isTextBox: true, margin: 0 });
    s.addText(m[1], { x: mx + 0.22, y: 2.92, w: mw - 0.44, h: 0.5, fontSize: 13.5, bold: true, color: mx === 0.6 ? WHITE : NAVY, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.05 });
    s.addText(m[2], { x: mx + 0.22, y: 3.48, w: mw - 0.44, h: 1.55, fontSize: 9.5, color: mx === 0.6 ? ICE : INK, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.18 });
    s.addShape("line", { x: mx + 0.22, y: 5.15, w: mw - 0.44, h: 0, line: { color: mx === 0.6 ? "33447A" : "D8DEEA", width: 1 } });
    s.addText(m[3], { x: mx + 0.22, y: 5.3, w: mw - 0.44, h: 1.2, fontSize: 9.3, bold: true, italic: true, color: mx === 0.6 ? TEAL_LIGHT : TEAL, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.2 });
    mx += mw + 0.19;
  });

  footer(s, 9, false);
  s.addNotes(
    "Same four-module design from Review I, now with a status update: every module is implemented, unit-tested, and integrated — this isn't a plan anymore. " +
    "Detection measures F1 between 0.51 and 0.57 across contamination ratios. Prioritization's frontier-greedy approximation measures a 0.00% mean optimality gap, max 0.14%, against the exact solver. Migration and Verification together correctly aborted 10 of 30 test scenarios where a plan would have regressed production quality. And head-to-head, the full system beats the best Phase-1 baseline in 18 of 30 scenarios. " +
    "Each of those numbers comes from a completed, real experiment — full methodology and evaluation detail available in our supporting documentation."
  );
}

// =====================================================================
// SLIDE 10 — COMPLEXITY & OVERHEAD ANALYSIS
// =====================================================================
{
  const s = pres.addSlide();
  lightBg(s);
  sectionTag(s, "Complexity & Overhead Analysis", false);
  title(s, "Space & Time Complexity", false);
  s.addText("N = corpus size (5,000). k = flagged/selected documents (k ≪ N typically). d = embedding dimension. M = baseline set size. Q = golden-query count (small, fixed).", {
    x: 0.6, y: 1.4, w: 12.15, h: 0.32, fontSize: 9.5, italic: true, color: MUTED, fontFace: "Calibri", isTextBox: true, margin: 0,
  });

  const head = ["Module", "Time Complexity", "Space Complexity", "Measured Overhead (real)"];
  const data = [
    ["Detection (M1)", "O(N log M) with an ANN index", "O(N·d) + O(M·d) baseline index", "0.43–0.58s to scan the full 5,000-doc benchmark"],
    ["Prioritization (M2) — production", "O(N log N) — sort-dominated (frontier-greedy)", "O(N)", "Near-linear; 0.00% mean gap, max 0.14% vs. exact"],
    ["Prioritization (M2) — exact (offline only)", "NP-hard in general; pseudo-polynomial DP / ILP branch-and-bound", "O(N·B)", "Used only on ≤300-doc subsamples — kept off the live path"],
    ["Migration (M3)", "O(k·d) — scales with fixes selected, not corpus size", "O(k·d) shadow copy — not a full corpus duplicate", "Overhead bounded by contamination scope, not N"],
    ["Verification (M4)", "O(Q·k) — small fixed query set", "O(Q)", "Cheap — re-scores a curated golden set, not full query load"],
  ];

  s.addTable(
    [head.map((h) => ({ text: h, options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 10 } }))].concat(
      data.map((row) => row.map((c) => ({ text: c, options: { fontSize: 9.2, color: INK } })))
    ),
    {
      x: 0.6, y: 1.85, w: 12.15, h: 3.7,
      colW: [2.5, 3.35, 2.8, 3.5],
      border: { type: "solid", color: "D8DEEA", pt: 0.75 },
      autoPage: false,
      valign: "middle",
      margin: [4, 6, 4, 6],
    }
  );

  card(s, 0.6, 5.7, 12.15, 1.25, { fill: NAVY, noShadow: true });
  s.addText([
    { text: "The overall system is near-linear in corpus size, by design.  ", options: { bold: true, color: TEAL_LIGHT } },
    { text: "The only super-linear step (exact ILP) is deliberately kept off the production path — used solely for offline validation of the greedy approximation, which is what actually runs live. Space overhead is bounded by k (documents actually being fixed), never a full second copy of the corpus.", options: { color: WHITE } },
  ], { x: 0.9, y: 5.85, w: 11.5, h: 0.95, fontSize: 11.5, fontFace: "Calibri", valign: "middle", isTextBox: true, margin: 0, lineSpacingMultiple: 1.25 });

  footer(s, 10, false);
  s.addNotes(
    "This slide is our direct, complete answer to the complexity and overhead questions from Review I. Detection is near-linear with an ANN index — measured at under 0.6 seconds for the full five-thousand-document benchmark, across all three detectors we tested. " +
    "Prioritization's production path, frontier-greedy, is also near-linear — dominated by a sort. The exact ILP solver IS expensive — NP-hard in general — but we deliberately never run it live; it's an offline validation tool we used on small subsamples to prove the greedy approximation is trustworthy. " +
    "Migration and Verification both scale with k, the number of documents actually being touched, not N, the corpus size — so the overhead is proportional to how contaminated the corpus actually is, not a fixed tax on every query. " +
    "Bottom line: the only expensive step is intentionally kept off the live system."
  );
}

// =====================================================================
// SLIDE 15 — TESTING, VALIDATION & RISK ASSESSMENT
// =====================================================================
{
  const s = pres.addSlide();
  lightBg(s);
  sectionTag(s, "Testing, Validation & Risk Assessment", false);
  title(s, "Testing, Validation & Risk Assessment", false);
  s.addText("Real reliability issues found through destructive testing at full scale (5,000 docs) — and fixed before this review.", {
    x: 0.6, y: 1.42, w: 12, h: 0.3, fontSize: 10.5, italic: true, color: MUTED, fontFace: "Calibri", isTextBox: true, margin: 0,
  });

  const bugs = [
    ["Shrinking-budget-pool stall", "Budget was computed from the shrinking remaining-candidate count each tick, so the absolute budget eventually fell below the cost of even one fix — stalling the incident just short of resolution.", "Budget is now sized off a stable reference: the incident's initial flagged count, not the shrinking pool."],
    ["Repeated re-scheduling of fixed documents", "A document already successfully remediated could be handed to the scheduler again if still flagged by residual noise, winning the zero-benefit tie-break every time and starving genuinely-contaminated documents of budget.", "Remediated documents are now tracked with a monotonic flag and excluded from all future scheduling rounds."],
    ["Infinite verification-retry loop", "Escalating budget on verification failure only helps when scarcity IS the cause. When a fix shifted ranking for an unrelated query, escalating forever stalled the incident permanently at maximum budget.", "A MAX_CONSECUTIVE_FAILURES cap now defers the candidate for manual review instead of retrying forever."],
  ];
  let by = 1.85;
  bugs.forEach((b, i) => {
    card(s, 0.6, by, 12.15, 1.28);
    pillNumber(s, 0.85, by + 0.16, i + 1, false);
    s.addText(b[0], { x: 1.5, y: by + 0.08, w: 3.3, h: 1.12, fontSize: 11.5, bold: true, color: BAD, fontFace: "Calibri", valign: "middle", isTextBox: true, margin: 0, lineSpacingMultiple: 1.1 });
    s.addText(b[1], { x: 4.95, y: by + 0.08, w: 4.05, h: 1.12, fontSize: 8.8, color: INK, fontFace: "Calibri", valign: "middle", isTextBox: true, margin: 0, lineSpacingMultiple: 1.15 });
    s.addText([{ text: "Fix:  ", options: { bold: true, color: GOOD } }, { text: b[2], options: { color: INK } }], { x: 9.15, y: by + 0.08, w: 3.5, h: 1.12, fontSize: 8.8, fontFace: "Calibri", valign: "middle", isTextBox: true, margin: 0, lineSpacingMultiple: 1.15 });
    by += 1.42;
  });

  s.addText("These are regression tests in our own test suite today — each bug can never silently reappear.", {
    x: 0.6, y: by + 0.05, w: 12.15, h: 0.3, fontSize: 9.5, italic: true, color: MUTED, fontFace: "Calibri", isTextBox: true, margin: 0,
  });

  footer(s, 11, false);
  s.addNotes(
    "This is genuine destructive testing, not a checkbox. Running the full system at five-thousand-document scale surfaced three real reliability bugs, none of which showed up in smaller smoke tests. " +
    "First, a shrinking-budget-pool bug that stalled incidents just short of resolution — fixed by sizing budget off a stable reference count. Second, already-fixed documents could be repeatedly rescheduled, starving genuinely contaminated ones — fixed with a monotonic remediated flag. Third, an infinite retry loop when escalating budget couldn't actually fix the real problem — fixed with a hard cap that defers to manual review instead of looping forever. " +
    "All three are now permanent regression tests, so they can never silently come back — this is exactly the kind of proactive, layered defense the rubric's Risk Assessment and Testing criteria ask for."
  );
}

// =====================================================================
// SLIDE 17 — FEASIBILITY ANALYSIS
// =====================================================================
{
  const s = pres.addSlide();
  lightBg(s);
  sectionTag(s, "Feasibility Analysis", false);
  title(s, "Feasibility Analysis — Now Evidence-Based", false);
  s.addText("At Review I this was an argument. Today it's a measurement.", {
    x: 0.6, y: 1.42, w: 12, h: 0.3, fontSize: 11.5, italic: true, color: MUTED, fontFace: "Calibri", isTextBox: true, margin: 0,
  });

  const feas = [
    ["Standard, proven components", "sentence-transformers, scikit-learn, numpy, an off-the-shelf ILP solver, Streamlit — every component is a well-documented tool. No custom infrastructure was needed to reach these results."],
    ["Bounded compute cost — proven, not just argued", "Detection under 0.6s for the full benchmark; frontier-greedy scheduling near-linear and measured within 0.14% of optimal; migration/verification bounded by contamination scope. Full breakdown on the Complexity slide."],
    ["Built end-to-end on a real, non-trivial benchmark", "Five thousand documents from BEIR SciFact, thirty ratio-by-budget scenarios run to completion — not a toy example, not a partial prototype."],
    ["Autonomous operation validated end-to-end", "The full detect-alert-schedule-migrate-verify-clear loop runs unattended after a single manual trigger, confirmed on the real 5,000-document benchmark."],
    ["A build order that was actually validated by testing", "Three real reliability bugs were found and fixed during full-scale testing — evidence the modular, dependency-ordered build plan actually works in practice, not just on paper."],
  ];
  let y = 1.85;
  feas.forEach((f) => {
    card(s, 0.6, y, 12.15, 0.92);
    s.addText([{ text: f[0] + " — ", options: { bold: true, color: NAVY } }, { text: f[1], options: { color: INK } }], {
      x: 0.85, y, w: 11.6, h: 0.92, fontSize: 10.3, fontFace: "Calibri", valign: "middle", isTextBox: true, margin: 0, lineSpacingMultiple: 1.2,
    });
    y += 1.02;
  });

  footer(s, 12, false);
  s.addNotes(
    "At Review I, feasibility was an argument — standard tools, provable complexity bounds, a de-risked build order. Today, every one of those claims has evidence behind it. " +
    "The components are still standard, but now we can point to real measured runtimes. The build order wasn't just theoretically de-risked — it was actually tested at full scale, and the three bugs we found and fixed are direct proof the process works. " +
    "And autonomous operation isn't a diagram anymore — it's a built, running module, validated end-to-end on the real benchmark."
  );
}

// =====================================================================
// SLIDE 18 — NOVELTY OF THE PROPOSED WORK
// =====================================================================
{
  const s = pres.addSlide();
  lightBg(s);
  sectionTag(s, "Novelty of the Proposed Work", false);
  title(s, "Novelty — Now Validated, Not Just Argued", false);

  const rows = [
    ["Novelty", TEAL, [
      "First application (to our literature) of a Multiple-Choice Knapsack formulation to embedding-contamination remediation — now measured at 0.00% mean optimality gap (max 0.14%) against the exact solver.",
      "A formal 3-way action space {Full Re-embed, Patch, Defer} where every reviewed prior work offers only a binary or single-fix choice — implemented and exercised across all 30 test scenarios.",
      "A closed detect → prioritize → migrate → verify loop with an automated safety gate — proven by 10 real, correct aborts in testing, not a theoretical claim.",
    ]],
    ["Creativity", AMBER, [
      "Reframing 'which documents to fix' as a combinatorial-optimization problem rather than a heuristic ranking rule — validated: naive heuristic sorting was 88% wrong; our knapsack-based scheduler was within 0.14%.",
      "Traffic-weighting severity via a Zipf-distributed traffic model (Breslau et al., 1999) — built into every one of our 30 evaluated scenarios.",
      "The Defer action as a first-class, mathematically justified choice — actually exercised in the built scheduler's real output across every tested ratio.",
    ]],
  ];
  let ry = 1.75;
  rows.forEach((r) => {
    card(s, 0.6, ry, 12.15, 2.55);
    s.addShape("roundRect", { x: 0.85, y: ry + 0.22, w: 2.0, h: 0.5, rectRadius: 0.09, fill: { color: r[1] }, line: { type: "none" } });
    s.addText(r[0], { x: 0.85, y: ry + 0.22, w: 2.0, h: 0.5, align: "center", valign: "middle", fontSize: 13.5, bold: true, color: WHITE, fontFace: "Calibri", isTextBox: true, margin: 0 });
    s.addText(r[2].map((t) => ({ text: t, options: { bullet: { code: "2022" }, breakLine: true } })), {
      x: 3.15, y: ry + 0.15, w: 9.35, h: 2.3, fontSize: 9.8, color: INK, fontFace: "Calibri", isTextBox: true, margin: 0, paraSpaceAfter: 7, lineSpacingMultiple: 1.13,
    });
    ry += 2.75;
  });

  footer(s, 13, false);
  s.addNotes(
    "The novelty and creativity claims from Review I are unchanged in substance, but every one now has empirical evidence behind it, not just design reasoning. " +
    "The knapsack formulation isn't just 'first to our literature' — we measured it beating a naive alternative by 88 percentage points. The safety gate isn't just a design feature — it fired correctly ten real times in testing. The Defer action isn't just philosophically justified — it's in the real scheduler output across every one of our thirty evaluated scenarios."
  );
}

// =====================================================================
// SLIDE 20 — REFERENCES
// =====================================================================
{
  const s = pres.addSlide();
  lightBg(s);
  sectionTag(s, "References", false);
  title(s, "References (APA)", false);
  s.addText("All 15 sources — abbreviated per guideline space constraints; full annotated list in the supporting document.", {
    x: 0.6, y: 1.4, w: 12, h: 0.3, fontSize: 10, italic: true, color: MUTED, fontFace: "Calibri", isTextBox: true, margin: 0,
  });

  const refs = [
    "Breslau, L., Cao, P., Fan, L., Phillips, G., & Shenker, S. (1999). Web caching and Zipf-like distributions: Evidence and implications. IEEE INFOCOM.",
    "Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). LOF: Identifying density-based local outliers. ACM SIGMOD.",
    "Fuster Baggetto, A., & Fresno, V. (2022). Is anisotropy truly harmful for dense retrieval? Findings of ACL: EMNLP 2022.",
    "Hu, Y., Bansal, S., Cao, K., Rao, R., Subbian, K., & Leskovec, J. (2022). Learning backward compatible embeddings. ACM KDD.",
    "Karval, R., & Singh, A. (2023). Silent failures in machine learning production systems. IEEE Conference Publication.",
    "Lample, G., Conneau, A., Ranzato, M., Denoyer, L., & Jégou, H. (2018). Word translation without parallel data (MUSE). ICLR.",
    "Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest. IEEE ICDM.",
    "Lu, J., Yin, W., & Xu, Y. (2024). Recent advances in OOD detection: Problems and approaches. ACM Computing Surveys.",
    "Malkov, Y. A., & Yashunin, D. A. (2020). Efficient and robust approximate nearest neighbor search using HNSW. IEEE TPAMI.",
    "Muennighoff, N., Tazi, N., Magne, L., & Reimers, N. (2023). MTEB: Massive text embedding benchmark. ACL.",
    "Pan, J. J., Wang, J., & Li, G. (2024). Survey of vector database management systems. The VLDB Journal, Springer.",
    "Sinha, P., & Zoltners, A. A. (1979). The multiple-choice knapsack problem. Operations Research, 27(3).",
    "Sun, Y., Ming, Y., Zhu, X., & Li, Y. (2022). Out-of-distribution detection with deep nearest neighbors. ICML.",
    "Vejendla, H. (2025). Drift-Adapter: A lightweight framework for continuous dense retrieval adaptation. EMNLP.",
    "Wu, K., Wu, E., & Zou, J. (2024). How faithful are RAG models? Quantifying the tug-of-war. arXiv preprint / survey.",
  ];
  s.addText(refs.map((r, i) => ({ text: r, options: { breakLine: i < refs.length - 1 } })), {
    x: 0.6, y: 1.85, w: 12.15, h: 5.15, fontSize: 8.6, color: INK, fontFace: "Calibri", isTextBox: true, margin: 0, paraSpaceAfter: 4.5, lineSpacingMultiple: 1.05,
  });

  footer(s, 14, false);
  s.addNotes(
    "Our full reference list — unchanged from Review I, still fifteen individually verified sources in APA format. Thank you — that's VectorSentry's Review II. I'm happy to take any questions."
  );
}

pres.writeFile({ fileName: "VectorSentry_Review2_v2.pptx" }).then(() => console.log("Full 14-slide Review II deck written: VectorSentry_Review2_v2.pptx"));
