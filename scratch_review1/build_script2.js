const {
  Document, Packer, Paragraph, TextRun, AlignmentType,
  Header, Footer, PageNumber,
} = require("docx");
const fs = require("fs");

const NAVY = "1B2A56";
const TEAL = "0E7C7B";
const MUTED = "5A6472";

const PAGE_W = 12240, PAGE_H = 15840; // US Letter

function slideHeading(n, title, time) {
  return new Paragraph({
    spacing: { before: 420, after: 120 },
    children: [
      new TextRun({ text: `SLIDE ${n}`, bold: true, size: 22, color: TEAL, characterSpacing: 20 }),
      new TextRun({ text: `  —  ${title}`, bold: true, size: 26, color: NAVY }),
      new TextRun({ text: `   (~${time})`, italics: true, size: 18, color: MUTED }),
    ],
    border: { bottom: { color: "D8DEEA", space: 6, style: "single", size: 6 } },
  });
}
function script(text) {
  return text.split("\n\n").map((para) => new Paragraph({
    spacing: { after: 160 },
    children: [new TextRun({ text: para.trim(), size: 24 })],
  }));
}
function cue(text) {
  return new Paragraph({
    spacing: { before: 40, after: 200 },
    children: [new TextRun({ text, italics: true, size: 18, color: MUTED })],
  });
}

const body = [];

body.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 80 },
  children: [new TextRun({ text: "VectorSentry — Review II", bold: true, size: 40, color: NAVY })],
}));
body.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 40 },
  children: [new TextRun({ text: "Full Speaker Script (14-slide deck)", size: 26, color: TEAL })],
}));
body.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 300 },
  children: [new TextRun({ text: "Read directly from this during your presentation. Estimated total runtime: ~16 minutes across 14 slides — practice once against a timer and trim to your actual slot.", italics: true, size: 18, color: MUTED })],
}));

// SLIDE 1
body.push(slideHeading(1, "Title", "30 sec"));
body.push(...script(
`Good [morning/afternoon], everyone. I'm presenting VectorSentry's Review Two — Budget-Aware Prioritization for Mixed-Generation Embedding Contamination in Production Vector Databases.

Since Review One, we've moved from a proposal to a built, tested, and evaluated system. Today I'll briefly respond to the feedback from Review One, then walk through the literature, our built architecture, the methodology and feasibility evidence, how we tested the system, and where the project stands.`
));
body.push(cue("[Advance to Response to Review I Feedback]"));

// SLIDE 2
body.push(slideHeading(2, "Response to Review I Feedback", "90 sec"));
body.push(...script(
`Before anything else, I want to directly answer the three questions the panel raised at Review One, because we took them seriously and they shaped what we prioritized building this term.

First: why is there a paper from the 1900s? Sinha and Zoltners, 1979, and Breslau and colleagues, 1999, are foundational theory, not stale trivia — the same way citing Dijkstra's 1959 shortest-path algorithm wouldn't be considered 'old.' We didn't just cite the 1979 knapsack formulation, we implemented it directly in our scheduler, and we measured it: a 0.00 percent mean optimality gap, with a maximum of 0.14 percent, against the exact solver, on exactly the instance where a naive approach was 88 percent wrong. The forty-five-year-old math holds up empirically, today, on our own data.

Second: doesn't this need a lot of overhead? We now have real numbers, not assumptions. Detection scans the full five-thousand-document benchmark in under 0.6 seconds, measured across all three detectors we tested. Migration only shadow-builds the documents actually selected for a fix, not the whole corpus. Verification re-scores a small curated golden-query set, not the full query load.

Third: what's the space and time complexity? Detection is O of N log N with an approximate nearest-neighbor index. Prioritization's production path is also O of N log N — the exact ILP solver is deliberately kept off the live path, used only for small-scale validation. Migration and verification are both bounded by k, the number of flagged documents, not N, the full corpus size. The full complexity table, with real measured wall-clock numbers alongside the theoretical bounds, is coming up shortly.`
));
body.push(cue("[Advance to Problem Statement recap]"));

// SLIDE 3
body.push(slideHeading(3, "The Problem, Briefly Revisited", "55 sec"));
body.push(...script(
`Quick recap for context, since you saw this in depth at Review One. Production vector databases silently degrade in retrieval quality when a subset of documents is embedded by a different generation of the embedding model than the rest of the corpus. Cosine similarity isn't comparable across two different embedding-model generations, so the database keeps returning results — just quietly worse ones, with no exception ever thrown.

It's real — documented in OpenAI and Cohere's own update history. It's silent — no exceptions, no alerts, no crash logs. And it's growing — over twenty commercial vector-database systems have launched in the last five years, all exposed to this same risk.

The key difference from Review One: we've now built the benchmark and measured this directly, which is what the rest of today's presentation is about — we've moved from arguing this problem to measuring it.`
));
body.push(cue("[Advance to Research Gap Identification]"));

// SLIDE 4
body.push(slideHeading(4, "Research Gap Identification", "70 sec"));
body.push(...script(
`This is the same research-gap logic from Review One, kept because it's still the spine of the project. Existing research gives us two mature, separate bodies of work: out-of-distribution and anomaly detection, and cheap embedding realignment. The common approach across both: detect anomalous vectors, then apply one fix — full re-embed or one uniform patch — to everyone flagged.

But the limitation is consistent: every reviewed fix mechanism is corpus-wide and uniform. None of them reason about which document to fix first when the budget can't cover everyone.

That's our research gap: no budget-aware, traffic-informed, per-document prioritization layer exists between 'detect' and 'fix' — and no safety gate ensures a chosen plan doesn't regress production quality before it ships.

Our contribution — a Multiple-Choice Knapsack formulation, three actions per document, one shared budget, solved exactly and approximately, gated by automated verification — is what we've now actually built and measured, not just proposed. Every claim on this slide is backed by a real experiment, and you'll see those numbers cited throughout the rest of this deck.`
));
body.push(cue("[Advance to Literature Review — Detection & Patching]"));

// SLIDE 5
body.push(slideHeading(5, "Literature Review — Detection & Patching", "75 sec"));
body.push(...script(
`These seven sources directly target detecting or repairing embedding contamination.

Three detection papers: Sun and colleagues' 2022 KNN-based out-of-distribution detector, which we've adapted directly as our Detection module's core scoring method — its F1 of 0.51 to 0.57 is exactly why it's our production choice, as noted in the table. Isolation Forest and Local Outlier Factor were evaluated as alternatives — F1 0.11 to 0.45 and 0.20 to 0.51 respectively, both weaker than KNN-OOD. And a 2024 survey confirms KNN-OOD as a mature, well-grounded choice.

Three patching papers: Vejendla's Drift-Adapter, published at EMNLP this year, shows cheap realignment is possible — we've adapted its mechanism directly as our Patch action, but gated by our own budget decision. Hu and colleagues' Backward-Compatible-Embeddings paper takes the opposite, prevention-side approach — training new models to stay compatible, which we discuss as complementary. And Lample and colleagues' 2018 MUSE paper gives us the underlying Procrustes-alignment math our Patch action's transform is built on.`
));
body.push(cue("[Advance to Literature Review — Foundational Work]"));

// SLIDE 6
body.push(slideHeading(6, "Literature Review — Foundational Work", "75 sec"));
body.push(...script(
`The remaining eight sources are our foundation layer. HNSW — the graph-indexing algorithm nearly every production vector database is built on — assumes a geometrically consistent index, an assumption our exact problem silently violates; that's precisely why contamination is dangerous. The vector-database survey and MTEB establish real-world scale and explain why embedding models churn so often, which is the practical trigger for our problem.

The two rows the Review One panel specifically asked about are Sinha and Zoltners, 1979, and Breslau and colleagues, 1999. We're not citing them as historical curiosities. Sinha and Zoltners' knapsack formulation is the mathematical core of our Prioritization module, now measured at a 0.00 percent mean optimality gap. And Breslau and colleagues' Zipf-traffic result directly grounds the traffic-weighted benefit function in our built scheduler. Foundational theorems don't expire — we cite them because we implement them.`
));
body.push(cue("[Advance to Dataset Details]"));

// SLIDE 7
body.push(slideHeading(7, "Dataset & Benchmark Construction", "70 sec"));
body.push(...script(
`Dataset details, as required. We use BEIR SciFact — a real, peer-reviewed information-retrieval benchmark with genuine relevance judgments, not synthetic labels we invented. Five thousand documents keeps the exact-ILP baseline solvable for direct optimality comparison, while remaining a meaningful evaluation.

Contamination is introduced by re-embedding a controlled subset with a second real embedding model — all-MiniLM-L12-v2 standing in for a 'new' provider generation, against all-MiniLM-L6-v2 as the 'old' one.

We swept five contamination ratios — 5, 10, 20, 35, and 50 percent — against six budget levels, from 10 to 100 percent — thirty scenarios per method, all run to completion, with a fixed seed of 42 for full reproducibility. Query traffic is modeled as Zipf-distributed, directly grounded in the Breslau et al. 1999 paper from our literature review, not an arbitrary choice. And we evaluate with both nDCG at 10 and Recall at 10, unweighted and traffic-weighted.`
));
body.push(cue("[Advance to System Architecture & Workflow]"));

// SLIDE 8
body.push(slideHeading(8, "System Architecture & Workflow", "70 sec"));
body.push(...script(
`This is the architecture as built and currently running — not a proposal anymore. A live vector database feeds Detect, which scores documents against a baseline-calibrated threshold. Prioritize is our knapsack scheduler, deciding full re-embed, patch, or defer for every flagged document. Migrate builds the fix as a shadow copy — the live index is never touched directly. Verify is the safety gate: a golden-query canary check, with automatic rollback on failure, and retry or defer as appropriate.

And the whole loop runs autonomously and continuously in our 'Live Sentry' module — built, integrated, and validated end to end on the real five-thousand-document benchmark.`
));
body.push(cue("[Advance to Methodology — Module Breakdown]"));

// SLIDE 9
body.push(slideHeading(9, "Methodology — Module Breakdown", "80 sec"));
body.push(...script(
`Same four-module design from Review One, now with a status update: every module is implemented, unit-tested, and integrated — this isn't a plan anymore.

Module 1, Detection, measures F1 between 0.51 and 0.57 across contamination ratios. Module 2, Prioritization's frontier-greedy approximation, measures a 0.00 percent mean optimality gap, with a maximum of 0.14 percent, against the exact solver. Modules 3 and 4, Migration and Verification together, correctly aborted ten of thirty test scenarios where a plan would have regressed production quality. And head-to-head, the full system beats the best Phase-1 baseline in eighteen of thirty scenarios.

Each of those numbers comes from a completed, real experiment — full methodology and evaluation detail is available in our supporting documentation.`
));
body.push(cue("[Advance to Complexity & Overhead Analysis]"));

// SLIDE 10
body.push(slideHeading(10, "Space & Time Complexity", "85 sec"));
body.push(...script(
`This slide is our direct, complete answer to the complexity and overhead questions from Review One. Detection is near-linear with an approximate nearest-neighbor index — measured at under 0.6 seconds for the full five-thousand-document benchmark, across all three detectors we tested.

Prioritization's production path, frontier-greedy, is also near-linear, dominated by a sort. The exact ILP solver IS expensive — NP-hard in general — but we deliberately never run it live; it's an offline validation tool we used on small subsamples to prove the greedy approximation is trustworthy.

Migration and Verification both scale with k, the number of documents actually being touched, not N, the corpus size — so the overhead is proportional to how contaminated the corpus actually is, not a fixed tax on every query.

Bottom line: the only expensive step in this whole system is intentionally kept off the live path.`
));
body.push(cue("[Advance to Testing, Validation & Risk Assessment]"));

// SLIDE 11
body.push(slideHeading(11, "Testing, Validation & Risk Assessment", "90 sec"));
body.push(...script(
`This is genuine destructive testing, not a checkbox exercise. Running the full system at five-thousand-document scale surfaced three real reliability bugs, none of which showed up in smaller smoke tests.

First, a shrinking-budget-pool bug: budget was computed from the shrinking remaining-candidate count each tick, so it eventually fell below the cost of even one fix, stalling the incident just short of resolution. Fixed by sizing budget off a stable reference — the incident's initial flagged count.

Second, already-fixed documents could be repeatedly rescheduled, winning the zero-benefit tie-break every time and starving genuinely contaminated documents of budget. Fixed with a monotonic remediated flag that excludes them from all future scheduling rounds.

Third, an infinite verification-retry loop: escalating budget on failure only helps when scarcity is the actual cause. When a fix shifted ranking for an unrelated query, escalating forever stalled the incident permanently. Fixed with a hard cap — MAX_CONSECUTIVE_FAILURES — that defers the candidate for manual review instead of retrying forever.

All three are now permanent regression tests in our own suite, so they can never silently reappear. This is exactly the kind of proactive, layered defense the rubric's Risk Assessment and Testing criteria are looking for.`
));
body.push(cue("[Advance to Feasibility Analysis]"));

// SLIDE 12
body.push(slideHeading(12, "Feasibility Analysis — Now Evidence-Based", "75 sec"));
body.push(...script(
`At Review One, feasibility was an argument. Today, every claim has evidence behind it.

The components are still standard — sentence-transformers, scikit-learn, numpy, an off-the-shelf ILP solver, Streamlit — no custom infrastructure was needed. Bounded compute cost isn't just argued anymore either — detection under 0.6 seconds, scheduling measured within 0.14 percent of optimal, migration and verification bounded by contamination scope, not corpus size.

We built end-to-end on a real, non-trivial benchmark — five thousand documents, thirty scenarios run to completion, not a toy example. Autonomous operation is validated end-to-end, not just diagrammed. And the build order wasn't just theoretically de-risked — it was actually tested at full scale, and the three bugs we found and fixed are direct proof the process works in practice.`
));
body.push(cue("[Advance to Novelty of the Proposed Work]"));

// SLIDE 13
body.push(slideHeading(13, "Novelty — Now Validated, Not Just Argued", "75 sec"));
body.push(...script(
`The novelty and creativity claims from Review One are unchanged in substance, but every one now has empirical evidence behind it, not just design reasoning.

On novelty: the knapsack formulation isn't just 'first, to our literature' anymore — we measured it beating a naive alternative by 88 percentage points. The three-way action space isn't just a design choice — it's implemented and exercised across all thirty test scenarios. And the safety gate isn't just a feature on a diagram — it fired correctly ten real times in testing.

On creativity: reframing this as a combinatorial-optimization problem is validated — naive heuristic sorting was 88 percent wrong, our knapsack-based scheduler was within 0.14 percent. The Zipf-distributed traffic-weighting is built into every one of our thirty evaluated scenarios. And the Defer action isn't just philosophically justified — it's in the real scheduler's output across every tested ratio.`
));
body.push(cue("[Advance to References]"));

// SLIDE 14
body.push(slideHeading(14, "References", "25 sec"));
body.push(...script(
`Our full reference list — unchanged from Review One, still fifteen individually verified sources in APA format.

That's VectorSentry's Review Two. Thank you — I'm happy to take any questions.`
));

const doc = new Document({
  sections: [
    {
      properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 } } },
      headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "VectorSentry — Review II Speaker Script", size: 16, color: MUTED, italics: true })] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 16, color: MUTED })] })] }) },
      children: body,
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("VectorSentry_Review2_SpeakerScript.docx", buf);
  console.log("Speaker script written: VectorSentry_Review2_SpeakerScript.docx");
});
