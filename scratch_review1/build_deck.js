const pptxgen = require("pptxgenjs");

// ---------- Palette: Midnight Executive (navy) + KSI teal, matching institutional branding ----------
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

// ---------- helpers ----------
function darkBg(slide) { slide.background = { color: NAVY }; }
function lightBg(slide) { slide.background = { color: WHITE }; }

function footer(slide, pageNo, dark) {
  slide.addText("VectorSentry  |  Review I", {
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
    x: 0.6, y: 0.42, w: 10.5, h: 0.35, fontSize: 12.5, bold: true, color: dark ? ICE : TEAL,
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
    x: 0.7, y: 2.55, w: 10.5, h: 1.1, fontSize: 54, bold: true, color: WHITE, fontFace: "Cambria",
    isTextBox: true, margin: 0,
  });
  s.addText("Budget-Aware Prioritization for Mixed-Generation Embedding\nContamination in Production Vector Databases", {
    x: 0.72, y: 3.55, w: 10.3, h: 0.95, fontSize: 19, color: ICE, fontFace: "Calibri",
    isTextBox: true, margin: 0, lineSpacingMultiple: 1.15,
  });

  card(s, 0.7, 4.85, 8.6, 1.55, { fill: "16224A", noShadow: true });
  s.addText([
    { text: "Team Member(s): ", options: { bold: true, color: ICE } },
    { text: "[Full Name — Register No. — Department]  (add one line per member)\n", options: { color: WHITE } },
    { text: "Guide: ", options: { bold: true, color: ICE } },
    { text: "[Guide Name, Designation]\n", options: { color: WHITE } },
    { text: "Review Date: ", options: { bold: true, color: ICE } },
    { text: "[Confirm exact review date]   |   UG Capstone Projects (2026-27) — Phase I, Review I", options: { color: WHITE } },
  ], { x: 0.95, y: 5.05, w: 8.1, h: 1.2, fontSize: 13.5, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.35 });

  s.addText("Fields in brackets must be completed before presenting — see final checklist.", {
    x: 0.7, y: 6.55, w: 8.5, h: 0.3, fontSize: 10.5, italic: true, color: AMBER, fontFace: "Calibri", isTextBox: true, margin: 0,
  });

  s.addNotes(
    "Good [morning/afternoon]. I'm presenting VectorSentry — a proposed system to detect and safely fix a silent failure mode in production vector databases: mixed-generation embedding contamination. " +
    "This is Review I — our first pitch of the idea. VectorSentry is scoped as a full-year capstone: today I'll walk you through the problem, the literature grounding it, our proposed design, and our year-long execution plan — not finished results."
  );
}

// =====================================================================
// SLIDE 2 — PROBLEM STATEMENT
// =====================================================================
{
  const s = pres.addSlide();
  lightBg(s);
  sectionTag(s, "Problem Identification", false);
  title(s, "Problem Statement", false);

  s.addText(
    "Production vector databases silently degrade in retrieval quality when a subset of documents is re-embedded by a different generation of the embedding model than the rest of the corpus — after a provider update, a model migration, or an incomplete backfill. Cosine similarity is not comparable across generations, so the database keeps returning results — just quietly worse ones, with nothing signaling anything is wrong.",
    { x: 0.6, y: 1.65, w: 6.6, h: 2.3, fontSize: 14.5, color: INK, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.3 }
  );

  const rows = [
    ["Root cause", "Each embedding-model generation's outputs occupy a different geometry — comparing across generations is mathematically meaningless."],
    ["Why it's silent", "No exception is thrown. The index still returns k results; only relevance quietly worsens."],
  ];
  let y = 4.1;
  rows.forEach((r) => {
    card(s, 0.6, y, 6.6, 1.05);
    s.addText(r[0], { x: 0.8, y: y + 0.1, w: 6.2, h: 0.3, fontSize: 12.5, bold: true, color: TEAL, fontFace: "Calibri", isTextBox: true, margin: 0 });
    s.addText(r[1], { x: 0.8, y: y + 0.42, w: 6.2, h: 0.58, fontSize: 11.5, color: INK, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.15 });
    y += 1.18;
  });

  card(s, 7.55, 1.65, 5.2, 4.9, { fill: NAVY, noShadow: true });
  s.addText("WHY THIS MATTERS", { x: 7.85, y: 1.9, w: 4.6, h: 0.3, fontSize: 11, bold: true, color: ICE, fontFace: "Calibri", charSpacing: 1.5, isTextBox: true, margin: 0 });

  const why = [
    ["It's already happening", "OpenAI updated text-embedding-ada-002 multiple times through 2023; Cohere's embed-v3 differs internally from embed-v2. Teams re-embed piecemeal, creating exactly this mix."],
    ["It's invisible to normal monitoring", "No exceptions, no alerts, no crash logs — the system looks healthy while quietly serving worse results."],
    ["The exposure is growing", "Over 20 commercial vector-database systems have launched in the past five years (Pan et al., 2024), and every one is exposed to this exact failure mode."],
  ];
  let wy = 2.35;
  why.forEach((v) => {
    s.addText(v[0], { x: 7.85, y: wy, w: 4.6, h: 0.3, fontSize: 12.5, bold: true, color: WHITE, fontFace: "Calibri", isTextBox: true, margin: 0 });
    s.addText(v[1], { x: 7.85, y: wy + 0.32, w: 4.6, h: 0.7, fontSize: 10, color: ICE, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.18 });
    wy += 1.12;
  });

  s.addShape("line", { x: 7.85, y: wy + 0.05, w: 4.6, h: 0, line: { color: "33447A", width: 1 } });
  s.addText("Real, silent, and growing — that combination is why this is worth solving now, before it causes a visible incident, not after.", {
    x: 7.85, y: wy + 0.2, w: 4.6, h: 0.7, fontSize: 10.5, italic: true, color: "8CA0C8", fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.2,
  });

  footer(s, 2, false);
  s.addNotes(
    "The core problem: a vector database can end up with documents embedded by two different generations of a model — say, half by an old provider version, half by a new one — after an update. " +
    "Cosine similarity across those two generations is meaningless, but nothing errors out. It just quietly retrieves worse results. " +
    "Why does this matter enough to spend a year on? Three reasons. First, it's already happening — OpenAI and Cohere have both shipped embedding updates that silently changed their models' internal geometry. Second, it's invisible to normal monitoring — no exceptions, no alerts. Third, the exposure is growing — over twenty commercial vector-database systems have launched in the last five years, and every one faces this same risk. " +
    "This is Review I, so we haven't run a quantified benchmark yet — that's planned Phase II work, described later in the plan."
  );
}

// =====================================================================
// SLIDE 3 — OBJECTIVES & SCOPE
// =====================================================================
{
  const s = pres.addSlide();
  lightBg(s);
  sectionTag(s, "Objectives", false);
  title(s, "Objectives, Scope & Target Users", false);

  card(s, 0.6, 1.45, 12.15, 0.68, { fill: NAVY, noShadow: true });
  s.addText([
    { text: "Main aim:  ", options: { bold: true, color: TEAL_LIGHT } },
    { text: "Design and build a budget-aware system that detects mixed-generation embedding contamination and safely prioritizes its remediation — recovering the most retrieval quality per unit of cost, without ever silently shipping a regression.", options: { color: WHITE } },
  ], { x: 0.9, y: 1.45, w: 11.55, h: 0.68, fontSize: 11.5, fontFace: "Calibri", valign: "middle", isTextBox: true, margin: 0, lineSpacingMultiple: 1.15 });

  const objs = [
    ["Detect", "reliably", "Flag mixed-generation contamination with an unsupervised, oracle-free detector.", "Target: F1 ≥ 0.50, with no ground-truth labels used"],
    ["Prioritize & recover", "efficiently", "Formulate remediation as a budget-constrained optimization, not a uniform sweep, recovering more quality per unit budget than current (FIFO) practice.", "Target: optimality gap ≤ 5% vs. exact solver; ≥ 40% quality headroom recovered at ≤ 30% budget"],
    ["Deploy", "safely", "Guarantee no remediation ships if it would regress a protected query set.", "Target: zero silent regressions ever ship to the live index"],
  ];

  let y = 2.25;
  objs.forEach((o, i) => {
    card(s, 0.6, y, 12.15, 0.88);
    pillNumber(s, 0.85, y + 0.23, i + 1, false);
    s.addText([
      { text: o[0] + " ", options: { bold: true, color: NAVY, fontSize: 13.5 } },
      { text: o[1], options: { italic: true, color: TEAL, fontSize: 11.5 } },
    ], { x: 1.5, y: y + 0.08, w: 4.7, h: 0.3, fontFace: "Calibri", isTextBox: true, margin: 0 });
    s.addText(o[2], { x: 1.5, y: y + 0.38, w: 4.7, h: 0.46, fontSize: 9.3, color: MUTED, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.08 });

    s.addShape("line", { x: 6.35, y: y + 0.1, w: 0, h: 0.68, line: { color: "D8DEEA", width: 1 } });

    s.addText("Target / success criterion", { x: 6.6, y: y + 0.08, w: 5.9, h: 0.2, fontSize: 8.5, color: MUTED, fontFace: "Calibri", isTextBox: true, margin: 0 });
    s.addText(o[3], { x: 6.6, y: y + 0.3, w: 5.9, h: 0.55, fontSize: 10.8, bold: true, color: NAVY, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.1 });

    y += 0.98;
  });

  card(s, 0.6, y + 0.06, 12.15, 0.85, { fill: TEAL_LIGHT });
  s.addText("TARGET CONSUMERS", { x: 0.85, y: y + 0.17, w: 11.6, h: 0.22, fontSize: 9.5, bold: true, color: TEAL, fontFace: "Calibri", charSpacing: 1.2, isTextBox: true, margin: 0 });
  s.addText(
    "Engineering and MLOps teams operating production vector databases and RAG / semantic-search systems — especially teams relying on third-party embedding APIs (OpenAI, Cohere) whose model updates are outside their control, and teams running large-scale stacks (Pinecone, Weaviate, Milvus, FAISS) where a full re-embed on every update is too costly.",
    { x: 0.85, y: y + 0.39, w: 11.6, h: 0.48, fontSize: 9.3, color: NAVY, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.12 }
  );
  y += 0.99;

  s.addText([
    { text: "Scope: ", options: { bold: true, color: NAVY } },
    { text: "BEIR SciFact corpus (5,000 docs), open sentence-transformer models, laptop-scale reproducible benchmark.   ", options: { color: INK } },
    { text: "Out of scope (this phase): ", options: { bold: true, color: NAVY } },
    { text: "live production traffic replay; multi-corpus generalization — see Future Work.", options: { color: INK } },
  ], { x: 0.6, y: y + 0.06, w: 12.1, h: 0.4, fontSize: 9.5, fontFace: "Calibri", isTextBox: true, margin: 0 });

  footer(s, 3, false);
  s.addNotes(
    "Our main aim, in one sentence: design and build a budget-aware system that detects mixed-generation embedding contamination and safely prioritizes its remediation, without ever silently shipping a regression. " +
    "That breaks into three objectives, each with a measurable target we've set for ourselves before writing any code. Detect: we're targeting F1 of at least 0.5 for an unsupervised detector with no ground-truth labels. Prioritize and recover efficiently: we want our fast greedy heuristic within 5% of the true optimum, recovering at least 40% of the achievable quality headroom at just 30% of the budget. Deploy safely: a guarantee, not a hope, that we never silently ship a worse index. " +
    "Our target consumers are engineering and MLOps teams running production vector databases — especially anyone dependent on a third-party embedding API they don't control, or running at a scale where a full re-embed on every model update just isn't affordable. " +
    "Scope is bounded deliberately — a laptop-reproducible benchmark, not a live production deployment — which keeps the whole project achievable within the year."
  );
}

// =====================================================================
// SLIDE 4 — DOMAIN / COHORT KNOWLEDGE
// =====================================================================
{
  const s = pres.addSlide();
  lightBg(s);
  sectionTag(s, "Domain / Cohort Knowledge", false);
  title(s, "Domain & Cohort Knowledge", false);

  card(s, 0.6, 1.65, 3.75, 4.85, { fill: NAVY, noShadow: true });
  s.addText("DOMAIN", { x: 0.85, y: 1.85, w: 3.3, h: 0.3, fontSize: 11, bold: true, color: ICE, fontFace: "Calibri", charSpacing: 1.5, isTextBox: true, margin: 0 });
  s.addText("Artificial Intelligence & Machine Learning", { x: 0.85, y: 2.15, w: 3.3, h: 0.5, fontSize: 17, bold: true, color: WHITE, fontFace: "Cambria", isTextBox: true, margin: 0, lineSpacingMultiple: 1.1 });
  s.addText("Sub-domain: Information Retrieval & MLOps Systems Engineering", { x: 0.85, y: 2.75, w: 3.3, h: 0.55, fontSize: 11.5, italic: true, color: ICE, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.15 });
  s.addText("[Confirm exact enrolled cohort/track name with coordinator]", { x: 0.85, y: 3.3, w: 3.3, h: 0.4, fontSize: 9.5, italic: true, color: AMBER, fontFace: "Calibri", isTextBox: true, margin: 0 });

  s.addShape("line", { x: 0.85, y: 3.85, w: 3.25, h: 0, line: { color: "33447A", width: 1 } });
  s.addText("KEY TERMS", { x: 0.85, y: 4.0, w: 3.3, h: 0.3, fontSize: 11, bold: true, color: ICE, fontFace: "Calibri", charSpacing: 1.5, isTextBox: true, margin: 0 });
  s.addText(
    "• Embedding — a dense numeric vector encoding meaning\n• Cosine similarity — angle-based closeness between vectors\n• ANN index (e.g. HNSW/FAISS) — fast approximate vector search\n• Embedding drift — silent semantic shift across model versions",
    { x: 0.85, y: 4.32, w: 3.3, h: 2.0, fontSize: 10.5, color: ICE, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.35 }
  );

  const right = [
    ["Current Trends & Technologies", "Retrieval-Augmented Generation (RAG) has made vector search a core production dependency; providers now ship frequent embedding-model updates (OpenAI, Cohere, open sentence-transformers). Over 20 commercial vector-database systems have launched in the past five years (Pan et al., 2024, VLDB Journal)."],
    ["Real-World Applications", "Pinecone, Weaviate, Milvus, and FAISS/HNSW-based search stacks power semantic search and RAG assistants at companies ranging from startups to major AI labs — every one of them is exposed to this exact silent-drift failure mode."],
    ["Relevance to This Project", "VectorSentry sits precisely at the intersection of information-retrieval evaluation, anomaly detection, and MLOps deployment safety — three established sub-fields this project deliberately combines rather than treats in isolation."],
  ];
  let ry = 1.65;
  right.forEach((r) => {
    card(s, 4.55, ry, 8.15, 1.53);
    s.addText(r[0], { x: 4.8, y: ry + 0.12, w: 7.6, h: 0.3, fontSize: 13, bold: true, color: TEAL, fontFace: "Calibri", isTextBox: true, margin: 0 });
    s.addText(r[1], { x: 4.8, y: ry + 0.46, w: 7.6, h: 1.0, fontSize: 11, color: INK, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.2 });
    ry += 1.68;
  });

  footer(s, 4, false);
  s.addNotes(
    "Our domain is AI/ML applied to information retrieval and MLOps — specifically, the operational side of running vector search systems in production, not just training embedding models. " +
    "The field is moving fast: RAG has made vector databases a default dependency, and there are now over twenty commercial vector-database products, per a 2024 VLDB Journal survey we cite. " +
    "Every one of these systems — Pinecone, Weaviate, Milvus — is exposed to the exact contamination problem we're solving, because they all face the same embedding-model-update reality."
  );
}

// =====================================================================
// SLIDE 5 — LITERATURE SURVEY: DETECTION & PATCHING (1 of 2)
// =====================================================================
{
  const s = pres.addSlide();
  lightBg(s);
  sectionTag(s, "Literature Survey — 1 of 2", false);
  title(s, "Literature Survey — Detection & Patching", false);
  s.addText("7 of 15 sources — the papers directly targeting detection or repair of embedding contamination. Full APA list on the References slide.", {
    x: 0.6, y: 1.4, w: 12, h: 0.3, fontSize: 10, italic: true, color: MUTED, fontFace: "Calibri", isTextBox: true, margin: 0,
  });

  const head1 = ["Source", "Core Concept", "Gap", "How We Address It"];
  const data1 = [
    ["Sun et al., 2022", "Non-parametric OOD detection via k-NN distance in embedding space", "Detects anomalies only — doesn't decide what to do next", "Adapted as our Detection module's core scoring method"],
    ["Lu et al., 2024", "Taxonomy of OOD-detection methods across ML domains", "No treatment of embedding-generation-drift-specific detection", "Confirms KNN-OOD as a mature, well-grounded detector choice"],
    ["Liu et al., 2008", "Isolation Forest — isolates anomalies via random feature-space splits", "General-purpose; untested on embedding-space anisotropy", "Evaluated as an alternative detector in our design"],
    ["Breunig et al., 2000", "LOF — density-relative local outlier scoring", "Same generality gap; heavier compute than KNN-OOD", "Second alternative detector considered in our design"],
    ["Vejendla, 2025", "Drift-Adapter — lightweight learned adapter cheaply realigns embeddings", "One uniform fix for the whole corpus — no per-document choice", "Adapted as our Patch action, gated by a budget decision"],
    ["Hu et al., 2022", "Trains new models to stay backward-compatible with old embeddings", "Needs control of model training — unusable with third-party APIs", "Discussed as a complementary, prevention-side alternative"],
    ["Lample et al., 2018", "MUSE — orthogonal Procrustes alignment between spaces, no parallel data", "Built for cross-lingual spaces; assumes no budget constraint", "Theoretical foundation for our Patch action's alignment transform"],
  ];

  s.addTable(
    [head1.map((h) => ({ text: h, options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 9.5 } }))].concat(
      data1.map((row) => row.map((c) => ({ text: c, options: { fontSize: 8.3, color: INK } })))
    ),
    {
      x: 0.6, y: 1.8, w: 12.15, h: 5.15,
      colW: [1.9, 3.15, 3.15, 3.95],
      border: { type: "solid", color: "D8DEEA", pt: 0.75 },
      autoPage: false,
      valign: "middle",
      margin: [3, 6, 3, 6],
    }
  );

  footer(s, 5, false);
  s.addNotes(
    "These seven sources directly target detecting or repairing embedding contamination. Reading across each row: KNN-OOD, Isolation Forest, and LOF detect anomalies but never say what to do once you've found one. " +
    "Drift-Adapter and the Backward-Compatible-Embeddings paper both show you can cheaply realign or avoid re-embedding entirely — but every one applies its fix as ONE uniform action, with no per-document decision. MUSE gives us the underlying alignment math. " +
    "The last column is the point of this slide: for every single one of these, we can name exactly how VectorSentry either adapts, extends, or is grounded in that specific paper — not just 'reads' it."
  );
}

// =====================================================================
// SLIDE 6 — LITERATURE SURVEY: FOUNDATIONAL WORK (2 of 2)
// =====================================================================
{
  const s = pres.addSlide();
  lightBg(s);
  sectionTag(s, "Literature Survey — 2 of 2", false);
  title(s, "Literature Survey — Foundational Work", false);
  s.addText("8 of 15 sources — systems, benchmarking, and Operations-Research foundations our design is built on.", {
    x: 0.6, y: 1.4, w: 12, h: 0.3, fontSize: 10, italic: true, color: MUTED, fontFace: "Calibri", isTextBox: true, margin: 0,
  });

  const head2 = ["Source", "Core Concept", "Gap", "How We Address It"];
  const data2 = [
    ["Malkov & Yashunin, 2020", "HNSW — graph index for fast approximate nearest-neighbor search", "Assumes a geometrically consistent index — silently broken by contamination", "Names exactly why contamination is dangerous for the indexes we target"],
    ["Pan et al., 2024", "Survey of 20+ vector-database system architectures", "No treatment of post-deployment embedding integrity", "Evidences real-world scale and relevance of the problem"],
    ["Muennighoff et al., 2023", "MTEB — unified benchmark for comparing embedding models", "Static snapshot; doesn't test mixed-generation drift scenarios", "Explains why models churn often — the practical trigger for our problem"],
    ["Wu et al., 2024", "Quantifies how retrieval quality affects RAG faithfulness", "Doesn't address the retrieval-quality-degradation cause itself", "Motivates why contamination matters beyond raw metrics"],
    ["Fuster Baggetto & Fresno, 2022", "Tests whether embedding anisotropy actually harms retrieval", "Studied within one generation, not across mixed generations", "Motivates a planned mixed-generation anisotropy stress-test"],
    ["Sinha & Zoltners, 1979", "Exact formulation for one-choice-per-group allocation under budget", "Purely theoretical — never applied to embedding remediation", "The mathematical core of our Prioritization module"],
    ["Breslau et al., 1999", "Real web traffic follows a Zipf (highly skewed) distribution", "About web objects, not vector-DB query traffic specifically", "Grounds our traffic-weighted benefit function in the scheduler"],
    ["Karval & Singh, 2023", "Names 'silent failure' as a distinct, under-monitored ML risk category", "General MLOps treatment — not specific to vector databases", "Provides the framing our Problem Statement slide applies directly"],
  ];

  s.addTable(
    [head2.map((h) => ({ text: h, options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 9.5 } }))].concat(
      data2.map((row) => row.map((c) => ({ text: c, options: { fontSize: 8, color: INK } })))
    ),
    {
      x: 0.6, y: 1.8, w: 12.15, h: 5.15,
      colW: [1.9, 3.15, 3.15, 3.95],
      border: { type: "solid", color: "D8DEEA", pt: 0.75 },
      autoPage: false,
      valign: "middle",
      margin: [3, 5, 3, 5],
    }
  );

  footer(s, 6, false);
  s.addNotes(
    "These eight sources are our foundation layer — none of them is directly 'about' contamination remediation, but each one is load-bearing. " +
    "HNSW explains exactly why contamination is dangerous — it assumes a consistent index geometry that contamination silently breaks. The vector-DB survey and MTEB establish real-world scale and explain why embedding models churn so often in the first place. " +
    "The two most important rows for our own methodology are the last four: Sinha and Zoltners' 1979 knapsack paper is the literal mathematical core of our scheduler, and Breslau and colleagues' 1999 Zipf-traffic paper is why our benefit function weights by real query traffic, not just severity. " +
    "Together with the previous slide, that's how we go from fifteen separate papers to one coherent research gap — which is exactly what's next."
  );
}

// =====================================================================
// SLIDE 7 — RESEARCH GAP
// =====================================================================
{
  const s = pres.addSlide();
  lightBg(s);
  sectionTag(s, "Literature → Research Gap", false);
  title(s, "From Literature to Research Gap", false);

  const chain = [
    ["Existing research", "Two mature, separate bodies of work: OOD/anomaly detection (Cluster A) and cheap embedding realignment (Cluster B)."],
    ["Common approach", "Detect anomalous vectors → apply ONE fix (full re-embed or one uniform patch transform) to everyone flagged."],
    ["Strengths", "Detection is well-validated (KNN-OOD F1 competitive); Drift-Adapter-style patching recovers 95–99% recall at ~1% the compute cost."],
    ["Limitations", "Every reviewed fix mechanism is corpus-wide and uniform — none reasons about WHICH document to fix first when the budget can't cover everyone."],
    ["Unresolved problem", "Given N contaminated documents with different severity AND different real query traffic, and 3 possible actions of different cost/quality, how should ONE bounded budget be allocated system-wide?"],
    ["Research gap", "No budget-aware, traffic-informed, per-document PRIORITIZATION layer exists between 'detect' and 'fix' — and no safety gate ensures a chosen plan doesn't regress production quality before it ships."],
  ];

  let y = 1.65;
  const rh = 0.78;
  chain.forEach((c, i) => {
    const isGap = i === chain.length - 1;
    card(s, 0.6, y, 8.5, rh, { fill: isGap ? NAVY : CARD, noShadow: isGap });
    pillNumber(s, 0.85, y + (rh - 0.42) / 2, i + 1, false);
    s.addText(c[0], { x: 1.5, y: y + 0.08, w: 2.1, h: rh - 0.16, fontSize: 12, bold: true, color: isGap ? WHITE : NAVY, fontFace: "Calibri", valign: "middle", isTextBox: true, margin: 0 });
    s.addText(c[1], { x: 3.7, y: y + 0.06, w: 5.3, h: rh - 0.12, fontSize: 10, color: isGap ? ICE : INK, fontFace: "Calibri", valign: "middle", isTextBox: true, margin: 0, lineSpacingMultiple: 1.12 });
    if (i < chain.length - 1) {
      s.addText("↓", { x: 0.6, y: y + rh - 0.05, w: 0.5, h: 0.25, fontSize: 14, color: TEAL, fontFace: "Calibri", align: "center", isTextBox: true, margin: 0 });
    }
    y += rh + 0.14;
  });

  card(s, 9.3, 1.65, 3.45, 5.0, { fill: TEAL_LIGHT });
  s.addText("OUR CONTRIBUTION", { x: 9.55, y: 1.85, w: 3, h: 0.3, fontSize: 11, bold: true, color: TEAL, fontFace: "Calibri", charSpacing: 1.5, isTextBox: true, margin: 0 });
  s.addText(
    "Formulate remediation as a Multiple-Choice Knapsack Problem — {Full Re-embed, Patch, Defer} per document, benefit = severity × traffic × recovery, one shared budget — solved exactly (ILP) and approximately (greedy), gated by automated shadow-build + verification before anything ships.",
    { x: 9.55, y: 2.25, w: 3, h: 2.5, fontSize: 11, color: NAVY, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.25 }
  );
  s.addText("Directly grounded in Sinha & Zoltners (1979) for the optimization core and Vejendla (2025) for the patch mechanism — not invented from nothing.", {
    x: 9.55, y: 5.0, w: 3, h: 1.5, fontSize: 9.5, italic: true, color: MUTED, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.2,
  });

  footer(s, 7, false);
  s.addNotes(
    "This slide is the logical spine of the whole project. Starting from what's already known — mature detection theory and mature-but-uniform repair mechanisms — I walk down to exactly where the literature stops: nobody decides WHICH document to fix first under a real budget. " +
    "That's not a vague gap — I can point to it in the comparison table on the previous slide, in the 'gap identified' column, for every single Cluster A and B paper. " +
    "Our contribution isn't invented from nothing either — the optimization core is a 1979 Operations Research formulation, and the patch mechanism itself is grounded directly in this year's Drift-Adapter paper. What's new is combining them into one budget-aware decision layer with a safety gate."
  );
}

// =====================================================================
// SLIDE 8 — PROPOSED CONCEPT
// =====================================================================
{
  const s = pres.addSlide();
  lightBg(s);
  sectionTag(s, "Proposed Concept", false);
  title(s, "Proposed Concept — VectorSentry", false);

  card(s, 0.6, 1.65, 12.15, 1.15, { fill: NAVY, noShadow: true });
  s.addText(
    "Prioritize contamination remediation the way an incident-response team would — not by fixing everything, and not by fixing arbitrarily, but by ranking WHAT to fix first under a real budget, using real traffic and real severity signals, verified safe before every change goes live.",
    { x: 0.9, y: 1.78, w: 11.5, h: 0.9, fontSize: 14.5, color: WHITE, fontFace: "Calibri", valign: "middle", isTextBox: true, margin: 0, lineSpacingMultiple: 1.2 }
  );

  s.addText("For every flagged document, the scheduler chooses exactly one of three actions:", {
    x: 0.6, y: 3.05, w: 11, h: 0.35, fontSize: 13, bold: true, color: NAVY, fontFace: "Calibri", isTextBox: true, margin: 0,
  });

  const actions = [
    ["Full Re-embed", "High cost", "Full quality recovery", GOOD, "Re-run the document through the current embedding model — the ground-truth fix."],
    ["Patch", "Low cost", "Partial recovery", AMBER, "A lightweight learned Procrustes/linear transform (Drift-Adapter-style) realigns the old embedding — cheap, imperfect."],
    ["Defer", "Zero cost", "Zero recovery", MUTED, "Leave it contaminated for now — the mathematically correct choice when severity × traffic is low relative to cost."],
  ];
  let ax = 0.6;
  actions.forEach((a) => {
    card(s, ax, 3.55, 3.92, 2.75);
    s.addShape("roundRect", { x: ax + 0.25, y: 3.8, w: 0.55, h: 0.55, rectRadius: 0.28, fill: { color: a[3] }, line: { type: "none" } });
    s.addText(a[0], { x: ax + 0.25, y: 4.5, w: 3.4, h: 0.35, fontSize: 15, bold: true, color: NAVY, fontFace: "Calibri", isTextBox: true, margin: 0 });
    s.addText([
      { text: a[1], options: { bold: true, color: a[3] } },
      { text: "   •   ", options: { color: MUTED } },
      { text: a[2], options: { bold: true, color: a[3] } },
    ], { x: ax + 0.25, y: 4.88, w: 3.4, h: 0.3, fontSize: 10.5, fontFace: "Calibri", isTextBox: true, margin: 0 });
    s.addText(a[3] === undefined ? "" : "", { x: 0, y: 0, w: 0.1, h: 0.1, isTextBox: true }); // no-op spacer
    s.addText(a[a.length - 1], { x: ax + 0.25, y: 5.25, w: 3.4, h: 0.95, fontSize: 9.8, color: INK, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.15 });
    ax += 4.12;
  });

  footer(s, 8, false);
  s.addNotes(
    "The concept, in one sentence: treat contamination remediation as an incident-response prioritization problem, not a batch job. " +
    "Concretely, for every document our detector flags, VectorSentry's scheduler picks exactly one of three actions — full re-embed for the documents that matter most, a cheap patch for the ones where partial recovery is good enough, or defer for the ones not worth spending budget on at all. " +
    "That third option, Defer, is deliberately part of the formal action set, not an afterthought — sometimes the mathematically correct answer really is 'not yet,' and pretending otherwise is how uniform strategies waste budget."
  );
}

// =====================================================================
// SLIDE 9 — ENGINEERING METHODOLOGY & MODULES
// =====================================================================
{
  const s = pres.addSlide();
  lightBg(s);
  sectionTag(s, "Engineering Applications & Methodology", false);
  title(s, "Engineering Methodology & Modules", false);
  s.addText("Four independently testable modules — each with a defined input/output contract, so any one can be swapped or re-validated in isolation.", {
    x: 0.6, y: 1.42, w: 11.8, h: 0.32, fontSize: 11, italic: true, color: MUTED, fontFace: "Calibri", isTextBox: true, margin: 0,
  });

  const mods = [
    ["M1", "Detection", "KNN-OOD scoring, calibrated against a stable-baseline percentile threshold (no oracle contamination labels used).", "In: Gen-A/Gen-B embeddings  →  Out: per-doc anomaly score + flag"],
    ["M2", "Prioritization", "MCKP formulation — 3 actions x N documents, 1 shared budget; solved exactly via ILP (small scale) and via a frontier-greedy approximation (production scale).", "In: flagged docs + severity + traffic  →  Out: per-doc action plan"],
    ["M3", "Migration", "Shadow-build the planned fixes as a separate copy; the live index is never mutated in place, matching real zero-downtime deployment practice.", "In: action plan  →  Out: shadow embedding set"],
    ["M4", "Verification & Cutover", "Golden-query canary evaluation against the shadow; atomic cutover on pass, automatic rollback + retry/defer on fail.", "In: shadow set  →  Out: live index update, or abort + reason"],
  ];
  let mx = 0.6;
  const mw = 2.9;
  mods.forEach((m) => {
    card(s, mx, 2.05, mw, 4.55, { fill: mx === 0.6 ? NAVY : CARD, noShadow: mx === 0.6 });
    // color each module card distinctly using teal header regardless
    s.addShape("roundRect", { x: mx + 0.22, y: 2.28, w: 0.6, h: 0.6, rectRadius: 0.3, fill: { color: TEAL }, line: { type: "none" } });
    s.addText(m[0], { x: mx + 0.22, y: 2.28, w: 0.6, h: 0.6, align: "center", valign: "middle", fontSize: 15, bold: true, color: WHITE, fontFace: "Calibri", isTextBox: true, margin: 0 });
    s.addText(m[1], { x: mx + 0.22, y: 3.0, w: mw - 0.44, h: 0.55, fontSize: 14, bold: true, color: mx === 0.6 ? WHITE : NAVY, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.05 });
    s.addText(m[2], { x: mx + 0.22, y: 3.6, w: mw - 0.44, h: 2.0, fontSize: 9.8, color: mx === 0.6 ? ICE : INK, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.18 });
    s.addShape("line", { x: mx + 0.22, y: 5.7, w: mw - 0.44, h: 0, line: { color: mx === 0.6 ? "33447A" : "D8DEEA", width: 1 } });
    s.addText(m[3], { x: mx + 0.22, y: 5.82, w: mw - 0.44, h: 0.7, fontSize: 8.8, italic: true, color: mx === 0.6 ? "8CA0C8" : MUTED, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.15 });
    mx += mw + 0.19;
  });

  footer(s, 9, false);
  s.addNotes(
    "These four modules ARE the engineering application of this project — they chain together end to end: Detection feeds flagged documents to Prioritization, which produces an action plan; Migration shadow-builds that plan; Verification either cuts it over live or rolls it back. " +
    "Our methodology follows standard software-engineering modular decomposition — each module has a strict input/output contract, so each can be unit-tested and even swapped independently. " +
    "Detection and Migration/Verification are adapted from established practice — KNN-OOD from Sun et al. 2022, shadow-build-and-cutover from real zero-downtime deployment patterns. " +
    "Prioritization, Module 2, is our own contribution — the knapsack formulation grounded in Sinha and Zoltners' 1979 Operations Research paper, applied here for the first time to embedding remediation. " +
    "We deliberately chose this framework — a well-understood exact optimization technique — over something like reinforcement learning, because it's provably optimal at small scale, has a fast well-understood approximation for production scale, and every decision it makes is explainable, which matters a great deal for a system that's allowed to modify a production database."
  );
}

// =====================================================================
// SLIDE 11 — NOVELTY & CREATIVITY
// =====================================================================
{
  const s = pres.addSlide();
  lightBg(s);
  sectionTag(s, "Novelty & Creativity", false);
  title(s, "Novelty & Creativity", false);

  const rows = [
    ["Novelty", TEAL, [
      "First application (to our literature) of a Multiple-Choice Knapsack formulation to embedding-contamination remediation — prior OR knapsack work targets resource allocation, not vector-DB integrity.",
      "A formal 3-way action space {Full Re-embed, Patch, Defer} where every reviewed prior work offers only a binary or single-fix choice.",
      "A closed detect → prioritize → migrate → verify loop with an automated safety gate — prior work stops at detection or at a one-shot patch, with no verified-safe deployment step.",
    ]],
    ["Creativity", AMBER, [
      "Reframing 'which documents to fix' as a classic combinatorial-optimization problem (knapsack) rather than a heuristic ranking rule — brings 45 years of provably-optimal OR theory to a brand-new problem domain.",
      "Traffic-weighting severity: a rarely-queried contaminated document and a heavily-queried one are NOT equally urgent — this benefit function borrows directly from Zipf-distributed real-world query traffic (Breslau et al., 1999).",
      "The Defer action is a first-class, mathematically justified choice — not a fallback — deliberately spending zero budget when severity x traffic doesn't clear the cost bar.",
    ]],
  ];

  let ry = 1.75;
  rows.forEach((r) => {
    card(s, 0.6, ry, 12.15, 2.55);
    s.addShape("roundRect", { x: 0.85, y: ry + 0.22, w: 2.0, h: 0.5, rectRadius: 0.09, fill: { color: r[1] }, line: { type: "none" } });
    s.addText(r[0], { x: 0.85, y: ry + 0.22, w: 2.0, h: 0.5, align: "center", valign: "middle", fontSize: 13.5, bold: true, color: WHITE, fontFace: "Calibri", isTextBox: true, margin: 0 });
    s.addText(r[2].map((t) => ({ text: t, options: { bullet: { code: "2022" }, breakLine: true } })), {
      x: 3.15, y: ry + 0.18, w: 9.35, h: 2.2, fontSize: 10.3, color: INK, fontFace: "Calibri", isTextBox: true, margin: 0, paraSpaceAfter: 8, lineSpacingMultiple: 1.15,
    });
    ry += 2.75;
  });

  footer(s, 10, false);
  s.addNotes(
    "On novelty: to the fifteen papers we reviewed, nobody has applied a knapsack formulation to vector-database contamination specifically — knapsack optimization is well-established in Operations Research generally, but not in this domain. " +
    "Nobody offers a three-way action choice either — it's always 'detect it' or 'here's one fix.' And nobody closes the loop with an automated, verified-safe deployment gate. " +
    "On creativity — the traffic-weighting idea specifically came from recognizing that real query logs are Zipf-distributed, a fact established in 1999 networking literature, and realizing that fact should directly shape which documents get budget first. That's borrowing an insight from a completely different subfield — caching and CDN traffic — and applying it here."
  );
}

// =====================================================================
// SLIDE 12 — REAL-WORLD RELEVANCE & TECHNICAL FEASIBILITY
// =====================================================================
{
  const s = pres.addSlide();
  lightBg(s);
  sectionTag(s, "Real-World Impact & Feasibility", false);
  title(s, "Real-World Impact & Feasibility", false);

  card(s, 0.6, 1.7, 5.9, 4.9, { fill: NAVY, noShadow: true });
  s.addText("REAL-WORLD RELEVANCE", { x: 0.9, y: 1.95, w: 5.3, h: 0.3, fontSize: 11.5, bold: true, color: TEAL_LIGHT, fontFace: "Calibri", charSpacing: 1.2, isTextBox: true, margin: 0 });
  s.addText([
    { text: "Every RAG / semantic-search system faces this the moment its embedding provider ships a model update.", options: { breakLine: true } },
    { text: "20+ production vector-DB systems now in commercial use (Pinecone, Weaviate, Milvus, Qdrant, pgvector) — Pan et al., 2024 VLDBJ survey.", options: { breakLine: true } },
    { text: "OpenAI, Cohere, and open-source embedding models are updated multiple times per year — every update is a contamination event at scale for any system that can't afford a full re-embed.", options: { breakLine: true } },
    { text: "Karval & Singh (2023) document 'silent' ML production failures as a distinct, under-addressed category — this project targets exactly that category for vector search.", options: {} },
  ].map((it, i, arr) => ({ text: it.text, options: { ...it.options, bullet: { code: "2022" }, breakLine: i < arr.length - 1 } })), {
    x: 0.9, y: 2.35, w: 5.3, h: 3.9, fontSize: 11, color: WHITE, fontFace: "Calibri", isTextBox: true, margin: 0, paraSpaceAfter: 10, lineSpacingMultiple: 1.2,
  });

  card(s, 6.7, 1.7, 6.05, 4.9);
  s.addText("TECHNICAL FEASIBILITY", { x: 7.0, y: 1.95, w: 5.4, h: 0.3, fontSize: 11.5, bold: true, color: TEAL, fontFace: "Calibri", charSpacing: 1.2, isTextBox: true, margin: 0 });
  const feas = [
    ["Standard, proven components", "sentence-transformers, scikit-learn, numpy, an off-the-shelf ILP solver, Streamlit — every planned component is an established, well-documented tool, not a research prototype."],
    ["Bounded compute cost by design", "Greedy heuristics for knapsack-structured problems are near-linear time with well-understood approximation bounds relative to the exact ILP solution — the algorithmic basis is sound before we run a single experiment."],
    ["Achievable benchmark scope", "A 5,000-document corpus keeps the exact-ILP baseline solvable for direct comparison, while staying large enough to be a meaningful evaluation — deliberately scoped to fit within the year."],
    ["A concrete, de-risked build order", "Four independently testable modules (Detection, Prioritization, Migration, Verification) give a clear implementation sequence — each can be built and validated before the next depends on it."],
  ];
  let fy = 2.35;
  feas.forEach((f) => {
    s.addText([{ text: f[0] + " — ", options: { bold: true, color: NAVY } }, { text: f[1], options: { color: INK } }], {
      x: 7.0, y: fy, w: 5.4, h: 0.85, fontSize: 10.3, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.2,
    });
    fy += 0.98;
  });

  footer(s, 11, false);
  s.addNotes(
    "This isn't a hypothetical scenario — every production RAG or semantic search system depends on an embedding model that WILL be updated eventually, and Pan et al.'s 2024 survey counts over twenty commercial vector-database systems now deployed. " +
    "On feasibility, we're not asking you to take it on faith that this can be built — every component is a standard, well-documented tool, the algorithm we're proposing has provable complexity bounds independent of our own implementation, and we've deliberately scoped the benchmark size so the whole build-and-evaluate cycle fits within the year. " +
    "The four-module breakdown gives us a concrete, de-risked build order, which is exactly what we'll walk through next on the project plan."
  );
}

// =====================================================================
// SLIDE 13 — PROJECT PLAN
// =====================================================================
{
  const s = pres.addSlide();
  lightBg(s);
  sectionTag(s, "Project Plan — One-Year Roadmap", false);
  title(s, "Project Plan & Timeline", false);

  const phases = [
    ["Literature Survey & Problem Formulation", "Systematic review of 15 sources; root-cause analysis and research-gap identification (what you've seen today).", "Months 1–2"],
    ["System Design & Architecture Finalization", "Detailed design of the 4-module pipeline (Detect / Prioritize / Migrate / Verify); benchmark and evaluation-metric design.", "Months 2–3"],
    ["Detection Module (M1)", "Implement and tune the KNN-OOD-based contamination detector against a calibrated clean-baseline threshold.", "Months 3–5"],
    ["Prioritization Scheduler (M2)", "Implement the MCKP formulation — exact ILP solver plus a frontier-greedy approximation for production scale.", "Months 5–7"],
    ["Migration & Verification Pipeline (M3–M4)", "Build the shadow-build, golden-query verification, and atomic cutover/rollback mechanism.", "Months 7–9"],
    ["Integration & Benchmark Evaluation", "Run the full pipeline end-to-end on the BEIR SciFact benchmark; head-to-head comparison against naive baselines.", "Months 9–11"],
    ["Live Demo, Final Report & Review II", "Autonomous \"Live Sentry\" proof-of-concept; final report writing and defense preparation.", "Months 11–12"],
  ];
  let py = 1.45;
  const rowH = 0.62, rowGap = 0.74;
  phases.forEach((p, i) => {
    card(s, 0.6, py, 12.15, rowH);
    pillNumber(s, 0.82, py + 0.1, i + 1, false);
    s.addText(p[0], { x: 1.42, y: py + 0.05, w: 7.6, h: 0.28, fontSize: 12, bold: true, color: NAVY, fontFace: "Calibri", isTextBox: true, margin: 0 });
    s.addText(p[1], { x: 1.42, y: py + 0.32, w: 7.6, h: 0.28, fontSize: 8.8, color: MUTED, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.05 });
    s.addShape("roundRect", { x: 10.55, y: py + 0.12, w: 1.95, h: 0.36, rectRadius: 0.18, fill: { color: TEAL_LIGHT }, line: { type: "none" } });
    s.addText(p[2], { x: 10.55, y: py + 0.12, w: 1.95, h: 0.36, align: "center", valign: "middle", fontSize: 9.5, bold: true, color: TEAL, fontFace: "Calibri", isTextBox: true, margin: 0 });
    py += rowGap;
  });

  s.addText("A full-year plan by design — each module builds on a validated predecessor, so the build order itself de-risks the schedule.", {
    x: 0.6, y: py + 0.06, w: 12.15, h: 0.3, fontSize: 9.5, italic: true, color: MUTED, fontFace: "Calibri", isTextBox: true, margin: 0,
  });

  footer(s, 12, false);
  s.addNotes(
    "This is our roadmap for the full year, not a status report — nothing here is done yet beyond the literature survey and problem formulation you've already seen today. " +
    "We've deliberately ordered the build so each module only starts once its predecessor is validated: detection first, since prioritization needs a working detector's output to schedule against; migration and verification last, since they depend on both. " +
    "The final two months are reserved for integration, evaluation, the live autonomous demo, and writing up the final report — building in slack rather than assuming everything goes right on the first attempt."
  );
}

// =====================================================================
// SLIDE 14 — EXPECTED OUTCOMES
// =====================================================================
{
  const s = pres.addSlide();
  lightBg(s);
  sectionTag(s, "Expected Outcomes", false);
  title(s, "Expected Outcomes & Success Criteria", false);
  s.addText("These are the outcomes VectorSentry is designed to achieve — targets we will validate empirically over the course of the year, not results we already have.", {
    x: 0.6, y: 1.45, w: 12, h: 0.35, fontSize: 11.5, italic: true, color: MUTED, fontFace: "Calibri", isTextBox: true, margin: 0,
  });

  const outs = [
    ["A reliable, oracle-free detector", "Flags mixed-generation contamination without requiring labeled ground truth, at accuracy competitive with established OOD-detection baselines."],
    ["A provably near-optimal scheduler", "Frontier-greedy prioritization tracks the exact ILP solution closely, so budget is spent close to optimally even at production scale."],
    ["Meaningfully better recovery per budget unit", "Outperforms naive full-re-embed-everything practice under the same budget, especially at tight budgets where prioritization matters most."],
    ["A safety gate that never ships a regression", "No remediation plan is deployed if it would regress a protected, high-value query set — verified automatically before every cutover."],
  ];
  let oy = 1.95;
  outs.forEach((o, i) => {
    card(s, 0.6, oy, 12.15, 0.78);
    pillNumber(s, 0.85, oy + 0.18, i + 1, false);
    s.addText(o[0], { x: 1.5, y: oy + 0.08, w: 10.8, h: 0.32, fontSize: 13, bold: true, color: NAVY, fontFace: "Calibri", isTextBox: true, margin: 0 });
    s.addText(o[1], { x: 1.5, y: oy + 0.4, w: 10.8, h: 0.36, fontSize: 10, color: INK, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.1 });
    oy += 0.88;
  });

  card(s, 0.6, oy + 0.08, 12.15, 1.3, { fill: NAVY, noShadow: true });
  s.addText("HOW WE'LL KNOW IF WE SUCCEEDED", { x: 0.9, y: oy + 0.24, w: 11.5, h: 0.28, fontSize: 10.5, bold: true, color: TEAL_LIGHT, fontFace: "Calibri", charSpacing: 1, isTextBox: true, margin: 0 });
  s.addText(
    "Each objective from the Objectives & Scope slide has a specific, pre-committed target metric. Review II will report the actual measured numbers against these targets — including where we fall short, not just where we succeed.",
    { x: 0.9, y: oy + 0.56, w: 11.5, h: 0.75, fontSize: 11.5, color: WHITE, fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacingMultiple: 1.3 }
  );

  footer(s, 13, false);
  s.addNotes(
    "These are expected outcomes and success criteria, not results — we're deliberately not presenting numbers today because we haven't run the evaluation yet; that's exactly what the year of Phase II work is for. " +
    "Each one maps directly to an objective from the Objectives slide: a reliable detector, a near-optimal scheduler, better recovery than naive practice, and a safety gate that never silently regresses production. " +
    "What we're committing to here is accountability — Review II will show the actual measured numbers against these exact targets, including any that we miss, not a filtered highlight reel."
  );
}

// =====================================================================
// SLIDE 15 — REFERENCES
// =====================================================================
{
  const s = pres.addSlide();
  lightBg(s);
  sectionTag(s, "References", false);
  title(s, "References (APA)", false);
  s.addText("All 15 sources — abbreviated per guideline space constraints; full annotated list with problem/method/dataset/findings/limitations in the companion document.", {
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
    "The full fifteen-source reference list, in APA format. Every one of these was individually verified against the actual publication — author list, venue, and year — before being included; none were taken from memory without checking. " +
    "The companion document has the complete annotated version with each paper's problem addressed, methodology, dataset, key findings, and limitations, exactly as the literature-survey guideline requires."
  );
}

pres.writeFile({ fileName: "VectorSentry_Review1.pptx" }).then(() => console.log("Full 14-slide deck written: VectorSentry_Review1.pptx"));
