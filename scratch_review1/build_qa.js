const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Header, Footer, PageNumber,
} = require("docx");
const fs = require("fs");

const NAVY = "1B2A56";
const TEAL = "0E7C7B";
const MUTED = "5A6472";

const PAGE_W = 12240, PAGE_H = 15840; // US Letter

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 160 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 260, after: 100 } });
}
function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 140 } });
}

const body = [];

body.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 80 },
  children: [new TextRun({ text: "VectorSentry — Review I", bold: true, size: 40, color: NAVY })],
}));
body.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 40 },
  children: [new TextRun({ text: "Q&A Preparation — Questions the Panel May Ask", size: 26, color: TEAL })],
}));
body.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 300 },
  children: [new TextRun({ text: "Matches the current 14-slide deck. Answers are written to be spoken in 15-45 seconds — practice them out loud, don't read them verbatim in the review.", italics: true, size: 18, color: MUTED })],
}));

const sections = [
  ["1. Problem & Motivation", [
    ["Why does mixed-generation contamination happen in practice, and why doesn't the system just error out?",
     "Because cosine similarity between two different embedding-model generations is mathematically well-defined — it always returns a number between -1 and 1 — it's just not meaningful. The index has no way to know the geometry changed, so it keeps returning results with quietly worse relevance, never an exception."],
    ["You haven't measured this yet — how do you know the quality drop is real and not just a theoretical concern?",
     "It's grounded in documented industry practice, not just theory: OpenAI and Cohere have both shipped embedding updates that change internal geometry, and the anisotropy/embedding-drift literature we reviewed (Fuster Baggetto & Fresno, 2022) treats this as an active, measurable phenomenon. Quantifying the exact magnitude on our own benchmark is precisely the first deliverable of Phase II — we're deliberately not claiming a number today because we haven't earned it yet."],
    ["Isn't this just a rare edge case?",
     "No — Pan et al.'s 2024 VLDB Journal survey counts 20+ commercial vector-database systems now deployed, and every major embedding provider ships model updates multiple times a year. Any team that can't afford to re-embed its entire corpus on every update hits this."],
  ]],
  ["2. Objectives, Scope & Target Users", [
    ["Why only three objectives instead of four?",
     "We merged what were originally two separate objectives — 'prioritize optimally' and 'recover efficiently' — into one. Efficient recovery is really the direct consequence of optimal prioritization, not an independent capability; measuring it separately would double-count the same thing. The combined objective still carries both original target metrics — the 5% optimality-gap target and the 40%-headroom-at-30%-budget target."],
    ["Who exactly are your target consumers, and why does that shape your design?",
     "Engineering and MLOps teams running production vector databases, especially those depending on third-party embedding APIs like OpenAI or Cohere. That specific choice shapes the design in two ways: it rules out prevention-side fixes like training your own backward-compatible model, since you don't control the embedding provider — and it's exactly why we assume a real, limited remediation budget, which is what motivates the knapsack formulation in the first place."],
    ["Why exclude live production traffic replay and multi-corpus generalization from scope?",
     "To keep the project laptop-reproducible and genuinely achievable within one year. Both are named explicitly as future work rather than silently dropped — we'd rather scope tightly and deliver than promise broadly and not finish."],
    ["What does the 'main aim' statement add that the three objectives don't already say?",
     "It's the one-sentence version — what someone should remember if they only remember one thing. The three objectives underneath are how we'll actually know whether we achieved it."],
  ]],
  ["3. Literature & Research Gap", [
    ["Why split the literature into two tables — Detection & Patching, and Foundational Work — instead of one comparison table?",
     "So the 'gap' column stays meaningful for every row. A detection or patching paper has a gap in the normal research sense — something it doesn't solve. A foundational paper like HNSW doesn't have a research gap in that sense; it has an assumption our project specifically challenges. Splitting by role keeps every 'how we address it' cell honest instead of forcing every citation into the same gap-shaped box."],
    ["How is 'how we address it' different from just restating your own project fifteen times?",
     "Each entry is specific to that paper, not a repeated template. KNN-OOD is literally reused as our scoring method. MUSE is the literal alignment math our Patch action's transform uses. HNSW explains the exact mechanism by which contamination becomes dangerous. Each relationship is traceable and different — some papers we adapt directly, some we position as alternatives, some we build our motivation on."],
    ["What stops someone from saying 'this is just Drift-Adapter with extra steps'?",
     "Drift-Adapter, and every other patching paper we reviewed, applies its fix uniformly to the whole corpus — no notion of a budget or a per-document choice. Our contribution is the decision layer above that: which documents get which action under a shared budget, a distinct Operations-Research problem (Sinha & Zoltners, 1979) none of the patching papers address."],
    ["Walk me through your research-gap logic in one breath.",
     "Detection literature finds anomalies but doesn't say what to do next; patching literature fixes cheaply but applies one uniform fix to everyone. Nobody decides which document gets which treatment under a real, limited budget. That's the gap — and it's exactly a Multiple-Choice Knapsack Problem, which is what we formulate it as."],
  ]],
  ["4. Methodology & Design Choices", [
    ["Why is there no system architecture diagram in this deck?",
     "The four-module breakdown, each with an explicit input/output contract, already fully specifies the data flow — Detection's output is literally Prioritization's input, Prioritization's output is Migration's input, and so on through Verification. A separate diagram would repeat the same information visually rather than add anything new, so we chose to spend that slide budget on the rigor of the contracts instead of a redundant picture."],
    ["Why formulate this as a knapsack problem instead of a simpler heuristic, like just sorting by severity?",
     "A pure severity sort ignores that documents have very different fix costs and very different query traffic. Knapsack is the right formulation because it's built exactly for 'maximize value under one shared budget, with mutually exclusive choices per item' — and it comes with provable optimality guarantees a heuristic sort simply doesn't have."],
    ["Why not use reinforcement learning or a learned policy instead?",
     "Explainability and safety. Every decision our scheduler makes traces back to a specific benefit/cost calculation — important for a system allowed to modify a production database. An RL policy is a black box we'd have far less confidence deploying without validation that's out of scope for a one-year capstone."],
    ["What exactly does 'Defer' mean, and why is it a real action rather than just doing nothing?",
     "Defer is a formal, budget-zero action the optimizer can choose when a document's severity times traffic doesn't justify its cost relative to better uses of the budget. It's mathematically identical to any other action in the knapsack — proven correct, not just defaulted to."],
    ["How does the safety gate actually decide pass or fail?",
     "It re-evaluates a curated set of 'golden queries' against the shadow-built candidate index. If retrieval quality on that golden set would regress beyond a tolerance threshold, the gate aborts the cutover and either retries with more budget or defers the affected documents."],
  ]],
  ["5. Feasibility, Plan & Risk", [
    ["Why should we believe this is achievable in one year?",
     "We scoped it deliberately: a 5,000-document benchmark keeps the exact-ILP baseline tractable, every component is a standard off-the-shelf tool, and the four-module decomposition gives a de-risked build order where each module only starts once its predecessor is validated. The plan reserves the last two months purely for integration and write-up."],
    ["What's your biggest risk, and what happens if a module doesn't hit its target?",
     "The detector is the biggest risk — if KNN-OOD doesn't reach F1 ≥ 0.50, every downstream module schedules against noisy signal. Our mitigation: KNN-OOD is already validated in the literature on other domains, so we're applying a known-good method, not inventing detection from scratch. LOF and Isolation Forest are documented fallbacks from the same review."],
    ["Why BEIR SciFact specifically, and only 5,000 documents?",
     "It's a real, standard, peer-reviewed IR benchmark with real relevance judgments, letting us compute real nDCG/recall. 5,000 documents keeps the exact-ILP baseline solvable for a direct optimality comparison — production-scale validation is explicitly named as future work, not claimed as already proven."],
    ["Your project plan mentions 'System Design & Architecture Finalization' — but you just said there's no architecture slide. Isn't that a contradiction?",
     "No — the module-level design is already fixed, that's what today's Methodology slide shows. That project-plan phase is about finalizing implementation-level detail underneath it — exact interfaces, data schemas, class boundaries — the kind of detail that belongs in engineering documentation, not a pitch deck."],
  ]],
  ["6. Novelty & Future Scope", [
    ["How confident are you that nobody has done this before?",
     "We phrase every novelty claim as 'to our literature' or 'in the fifteen papers we reviewed' — never as an unqualified absolute. Within that scope, no paper we found combines a formal budget-constrained action space with an automated safety gate for embedding remediation specifically."],
    ["What's explicitly out of scope this year?",
     "Production-scale validation beyond 5,000 documents, integration against a real vector-DB API instead of our own benchmark harness, and a second corpus to corroborate our findings — all named as future work, not hidden omissions."],
    ["Is the knapsack solver going to scale to millions of documents?",
     "The exact ILP solver won't — that's why the frontier-greedy approximation is part of the design from day one. Greedy heuristics for this problem structure have well-understood, provable approximation bounds in the Operations Research literature (Sinha & Zoltners, 1979), independent of our own implementation. Validating that bound at larger scale is future work beyond this capstone year."],
  ]],
];

sections.forEach(([theme, qs]) => {
  body.push(h1(theme));
  qs.forEach(([q, a]) => {
    body.push(new Paragraph({ spacing: { before: 140, after: 40 }, children: [new TextRun({ text: "Q: ", bold: true, color: TEAL, size: 21 }), new TextRun({ text: q, bold: true, size: 21 })] }));
    body.push(new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "A: ", bold: true, color: NAVY, size: 20 }), new TextRun({ text: a, size: 20 })] }));
  });
});

const doc = new Document({
  sections: [
    {
      properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 } } },
      headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "VectorSentry — Review I Q&A Prep", size: 16, color: MUTED, italics: true })] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 16, color: MUTED })] })] }) },
      children: body,
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("VectorSentry_Review1_QandA.docx", buf);
  console.log("Q&A document written: VectorSentry_Review1_QandA.docx");
});
