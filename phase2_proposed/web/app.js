/* VectorSentry Live Sentry -- front end. Polls /api/state once a second and
   redraws seven ECharts views; every number comes from the engine. */
"use strict";

const C = {
  blue: "#3987e5", blueSoft: "rgba(57,135,229,.18)", orange: "#d95926", slate: "#3a4558", later: "#5b6474",
  ink: "#e8eaee", ink2: "#b4bac5", muted: "#7d8696", grid: "#1f2633", axis: "#2b3342",
  good: "#0ca30c", critical: "#d03b3b", affected: "#7a2e33", warning: "#fab219",
};
// Index-map status codes (see web_app.py).
const STATUS = [
  { key: "healthy", label: "Healthy", color: "#2b3a52" },
  { key: "affected", label: "Hit by update, not yet caught", color: C.affected },
  { key: "flagged", label: "Flagged", color: "#ff5a5f" },
  { key: "reembed", label: "Re-embedded", color: C.blue },
  { key: "patch", label: "Quick patch", color: C.orange },
];
const DECISION = {
  full_reembed: { label: "Re-embed", color: C.blue },
  patch: { label: "Quick patch", color: C.orange },
  defer: { label: "Later", color: C.later },
};
const FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif';
const $ = (id) => document.getElementById(id);
const fmt = (n) => (n == null ? "–" : Math.round(n).toLocaleString());
const pct = (v, d = 0) => (v == null ? "–" : `${(v * 100).toFixed(d)}%`);

const charts = {};
function chart(id) {
  if (!charts[id]) charts[id] = echarts.init($(id), null, { renderer: "canvas" });
  return charts[id];
}
window.addEventListener("resize", () => Object.values(charts).forEach((c) => c.resize()));

const baseText = { color: C.ink2, fontFamily: FONT };
const tooltip = (extra = {}) => ({
  backgroundColor: "rgba(10,14,22,.95)", borderColor: "rgba(148,163,184,.25)", borderWidth: 1,
  textStyle: { color: C.ink, fontFamily: FONT, fontSize: 12 }, extraCssText: "border-radius:10px;box-shadow:0 8px 30px rgba(0,0,0,.5);",
  ...extra,
});
const axisCommon = {
  axisLine: { lineStyle: { color: C.axis } }, axisTick: { show: false },
  axisLabel: { color: C.muted, fontFamily: FONT, fontSize: 11 },
  splitLine: { lineStyle: { color: C.grid } }, nameTextStyle: { color: C.ink2, fontFamily: FONT, fontSize: 12 },
};

/* ---------- number tweening ---------- */
const shown = {};
function tween(id, target, render) {
  const el = $(id);
  if (target == null || Number.isNaN(target)) { el.textContent = "–"; shown[id] = null; return; }
  const from = shown[id] ?? target;
  shown[id] = target;
  if (from === target) { el.textContent = render(target); return; }
  const t0 = performance.now(), dur = 700;
  const step = (now) => {
    const k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 3);
    el.textContent = render(from + (target - from) * e);
    if (k < 1 && shown[id] === target) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/* ---------- controls ---------- */
let S = null; // latest state
let layout = null;
let selectedRound = 1;
let budget = 0.3, safety = "balanced";

function setSeg(segId, value) {
  document.querySelectorAll(`#${segId} button`).forEach((b) => b.classList.toggle("on", b.dataset.v === String(value)));
}
async function post(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  return r.json();
}
document.querySelectorAll("#budgetSeg button").forEach((b) => b.addEventListener("click", async () => {
  budget = parseFloat(b.dataset.v); setSeg("budgetSeg", b.dataset.v); await post("/api/config", { budget, safety });
}));
document.querySelectorAll("#safetySeg button").forEach((b) => b.addEventListener("click", async () => {
  safety = b.dataset.v; setSeg("safetySeg", safety); await post("/api/config", { budget, safety });
}));
$("simulateBtn").addEventListener("click", async () => {
  $("simulateBtn").disabled = true; selectedRound = 1; lastCurveKey = ""; await post("/api/simulate"); poll();
});
$("resetBtn").addEventListener("click", async () => { selectedRound = 1; lastCurveKey = ""; await post("/api/reset"); poll(); });
setSeg("budgetSeg", "0.3"); setSeg("safetySeg", "balanced");

/* ---------- status, banner, pipeline ---------- */
function phase(s) {
  if (s.status === "needs_review") return "review";
  if (s.incident_active && s.status === "remediating") return "fixing";
  if (s.incident_active) return "detecting";
  if (s.status === "resolved" && s.n_patched > 0) return "polishing";
  if (s.status === "resolved") return "resolved";
  return "watching";
}
function renderStatus(s) {
  const ph = phase(s), excess = Math.max(0, s.flagged_live - s.flagged_ref);
  const pill = {
    watching: ["", "All clear · monitoring"], detecting: ["alert", "Incident detected"], fixing: ["work", "Repairing automatically"],
    polishing: ["work", "Resolved · polishing"], resolved: ["", "Back to normal"], review: ["warn", "Paused for review"],
  }[ph];
  $("statusPill").className = `status-pill ${pill[0]}`;
  $("statusText").textContent = pill[1];
  $("simulateBtn").disabled = s.incident_active;

  const banners = {
    watching: ["good", "Healthy", "The live index matches the reference. Press <b>⚡ Simulate provider update</b> to break it."],
    detecting: ["alert", "Provider update detected", `<b>${fmt(s.flagged_live)}</b> documents look like they were written by a different model.`],
    fixing: ["work", "Repairing, busiest documents first", `<b>${fmt(excess)}</b> suspicious documents left. Every batch is tested before it goes live.`],
    polishing: ["work", "Incident resolved, polishing", `Replacing <b>${fmt(s.n_patched)}</b> quick patches with exact re-embeds in the background.`],
    resolved: ["good", "Back to normal", `<b>${fmt(s.n_remediated)}</b> documents repaired. Every change passed the safety check.`],
    review: ["warn", "Paused for a human", `<b>${fmt(excess)}</b> documents remain whose fixes failed the safety check, so nothing was forced through.`],
  }[ph];
  $("banner").className = `banner ${banners[0]}`;
  $("banner").innerHTML = `<div class="bh">${banners[1]}</div>${banners[2]}`;

  const steps = [
    ["Watch", "Compare live index with the reference"],
    ["Detect", "Fingerprint vectors from the wrong model"],
    ["Prioritize & fix", "Knapsack plan: busiest documents first"],
    ["Verify", `Replay ${s.canary_size} logged searches per batch`],
    ["Polish", "Swap quick patches for exact re-embeds"],
  ];
  const active = { watching: 0, detecting: 1, fixing: 2, review: 3, polishing: 4, resolved: 5 }[ph];
  $("pipeline").innerHTML = '<div class="kicker" style="margin-bottom:6px">Autonomous pipeline</div>' + steps.map(([t, sub], i) => {
    let cls = i < active ? "done" : i === active ? "active" : "";
    if (ph === "fixing" && i === 3) cls = "active";
    if (ph === "review" && i === 3) cls = "blocked";
    if (ph === "watching" && i === 0) cls = "active";
    const icon = cls === "done" ? "✓" : cls === "blocked" ? "!" : i + 1;
    return `<div class="step ${cls}"><div class="ic">${icon}</div><div><div class="t">${t}</div><div class="s">${sub}</div></div></div>`;
  }).join("");
}

function renderKpis(s) {
  tween("kpiFlagged", s.flagged_live, fmt);
  $("kpiFlaggedD").textContent = s.flagged_live > s.flagged_ref ? `+${fmt(s.flagged_live - s.flagged_ref)} above normal (${s.flagged_ref})` : `normal background: ${s.flagged_ref}`;
  tween("kpiRepaired", s.n_remediated, fmt);
  $("kpiRepairedD").textContent = s.n_patched ? `${fmt(s.n_patched)} still on a quick patch` : s.n_deferred ? `${fmt(s.n_deferred)} waiting for review` : "";
  tween("kpiBudget", s.budget_spent, (v) => v.toFixed(0));
}

/* ---------- index map ---------- */
let lastStatusKey = "";
function renderMap(s) {
  if (!layout) return;
  const status = s.doc_status;
  const counts = [0, 0, 0, 0, 0];
  status.forEach((v) => counts[v]++);
  $("mapLegend").innerHTML = STATUS.map((st, i) => (i === 0 || counts[i] ? `<span><i class="sw" style="background:${st.color}"></i>${st.label}<b>${fmt(counts[i])}</b></span>` : "")).join("");

  const key = status.join("");
  if (key === lastStatusKey) return;
  lastStatusKey = key;
  // Draw healthy documents first so every changed document sits on top.
  const order = status.map((v, i) => i).sort((a, b) => (status[a] === 0) - (status[b] === 0) || layout.size[a] - layout.size[b]).reverse();
  const data = order.map((i) => [layout.xy[i][0], layout.xy[i][1], layout.size[i], status[i], i]);
  const pulsing = status.map((v, i) => [v, i]).filter(([v]) => v === 2).map(([, i]) => i)
    .sort((a, b) => layout.size[b] - layout.size[a]).slice(0, 18)
    .map((i) => [layout.xy[i][0], layout.xy[i][1], layout.size[i], 2, i]);

  chart("mapChart").setOption({
    animationDurationUpdate: 650,
    grid: { left: 4, right: 4, top: 4, bottom: 4 },
    xAxis: { show: false, min: -0.02, max: 1.02 }, yAxis: { show: false, min: -0.02, max: 1.02 },
    tooltip: tooltip({
      formatter: (p) => {
        const i = p.value[4];
        return `<b>Document ${layout.doc_ids[i]}</b><br>${STATUS[p.value[3]].label}<br>Search traffic ${layout.traffic_vs_average[i].toFixed(2)}× average`;
      },
    }),
    series: [
      {
        type: "scatter", data, symbolSize: (v) => v[2] * (v[3] === 0 ? 1 : 1.25),
        itemStyle: {
          color: (p) => STATUS[p.value[3]].color,
          opacity: 1, borderColor: "rgba(6,9,15,.6)", borderWidth: 0.6,
          shadowBlur: 0,
        },
        emphasis: { scale: 1.6 },
        progressive: 0,
      },
      {
        type: "effectScatter", data: pulsing, symbolSize: (v) => v[2] * 1.25, rippleEffect: { brushType: "stroke", scale: 3.2, period: 2.2 },
        itemStyle: { color: STATUS[2].color, shadowBlur: 12, shadowColor: "rgba(255,90,95,.7)" }, silent: true, zlevel: 1,
      },
    ],
  });
}

/* ---------- health gauge ---------- */
function renderGauge(s) {
  const v = s.health == null ? 100 : s.health * 100;
  const color = v >= 99.5 ? C.good : C.blue;
  chart("gaugeChart").setOption({
    series: [{
      type: "gauge", startAngle: 205, endAngle: -25, min: 70, max: 100, splitNumber: 6, radius: "100%", center: ["50%", "62%"],
      progress: { show: true, width: 14, roundCap: true, itemStyle: { color, shadowBlur: 18, shadowColor: color } },
      axisLine: { roundCap: true, lineStyle: { width: 14, color: [[1, "#1b2230"]] } },
      axisTick: { show: false }, splitLine: { show: false },
      axisLabel: { distance: -38, color: C.muted, fontSize: 10, fontFamily: FONT, formatter: (x) => (x % 10 === 0 ? `${x}%` : "") },
      pointer: { show: false }, anchor: { show: false }, title: { show: false },
      detail: {
        valueAnimation: true, offsetCenter: [0, "-4%"], fontSize: 40, fontWeight: 800, fontFamily: FONT, color: C.ink,
        formatter: (x) => `${x.toFixed(1)}%`,
      },
      data: [{ value: Math.round(v * 10) / 10 }],
      animationDurationUpdate: 800,
    }],
  });
  $("healthNote").textContent = s.health == null ? "" :
    v >= 99.95 ? "Popular searches return exactly what they returned before the incident." :
      `Popular searches get back ${v.toFixed(1)}% of the results they returned before the incident.`;
}

/* ---------- timeline (two small multiples on one time axis) ---------- */
function renderTimeline(s) {
  const h = s.history.filter((p) => p.health != null);
  if (!h.length) return;
  const health = h.map((p) => [p.t, +(p.health * 100).toFixed(2)]);
  const live = h.map((p) => [p.t, p.flagged_live]);
  const ref = h.map((p) => [p.t, p.flagged_ref]);
  const minH = Math.min(...health.map((p) => p[1]));
  chart("timelineChart").setOption({
    animation: false,
    grid: [{ left: 52, right: 20, top: 14, height: "54%" }, { left: 52, right: 20, top: "74%", bottom: 26 }],
    tooltip: tooltip({ trigger: "axis", axisPointer: { type: "line", lineStyle: { color: C.axis } }, valueFormatter: (v) => v }),
    axisPointer: { link: [{ xAxisIndex: "all" }] },
    xAxis: [
      { ...axisCommon, type: "time", gridIndex: 0, axisLabel: { show: false }, splitLine: { show: false } },
      { ...axisCommon, type: "time", gridIndex: 1, splitLine: { show: false } },
    ],
    yAxis: [
      { ...axisCommon, gridIndex: 0, min: Math.max(0, Math.floor(Math.min(90, minH - 2) / 5) * 5), max: 100, interval: 5,
        axisLabel: { ...axisCommon.axisLabel, formatter: "{value}%" } },
      { ...axisCommon, gridIndex: 1, min: 0, splitNumber: 2 },
    ],
    series: [
      {
        name: "Search health", type: "line", xAxisIndex: 0, yAxisIndex: 0, data: health, showSymbol: false,
        lineStyle: { color: C.blue, width: 2.5, shadowBlur: 12, shadowColor: "rgba(57,135,229,.6)" },
        areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: "rgba(57,135,229,.35)" }, { offset: 1, color: "rgba(57,135,229,0)" }]) },
        markLine: { silent: true, symbol: "none", data: [{ yAxis: 100 }], lineStyle: { color: C.muted, type: "solid", width: 1 },
          label: { formatter: "Healthy reference", color: C.muted, position: "insideEndBottom", fontSize: 11 } },
        markPoint: minH < 99.5 ? { symbol: "circle", symbolSize: 8, itemStyle: { color: C.ink },
          data: [{ type: "min", label: { formatter: (p) => `Low ${p.value}%`, position: "bottom", color: C.ink2, fontSize: 11, distance: 6 } }] } : undefined,
        tooltip: { valueFormatter: (v) => `${v}%` },
      },
      { name: "Live index", type: "line", xAxisIndex: 1, yAxisIndex: 1, data: live, showSymbol: false, lineStyle: { color: C.blue, width: 2 },
        areaStyle: { color: "rgba(57,135,229,.14)" } },
      { name: "Normal background", type: "line", xAxisIndex: 1, yAxisIndex: 1, data: ref, showSymbol: false, lineStyle: { color: C.muted, width: 1.2 } },
    ],
  });
}

/* ---------- activity feed ---------- */
let lastFeedKey = "";
function renderFeed(s) {
  const key = s.events.map((e) => e.time + e.text).join("|");
  if (key === lastFeedKey) return;
  lastFeedKey = key;
  $("feed").innerHTML = s.events.map((e) => `<li><time>${e.time}</time><span class="k ${e.kind}"></span><span>${e.text}</span></li>`).join("");
}

/* ---------- the decision ---------- */
let lastCurveKey = "";
function renderDecision(s) {
  const rounds = s.rounds;
  $("decisionEmpty").classList.toggle("hidden", rounds.length > 0);
  $("decisionBody").classList.toggle("hidden", rounds.length === 0);
  $("roundTabs").innerHTML = rounds.length > 1 ? rounds.map((r, i) =>
    `<button data-r="${i + 1}" class="${i + 1 === selectedRound ? "on" : ""}">Round ${i + 1}<small>${r.rows.length}</small></button>`).join("") : "";
  document.querySelectorAll("#roundTabs button").forEach((b) => b.addEventListener("click", () => {
    selectedRound = +b.dataset.r; lastCurveKey = ""; renderDecision(S);
  }));
  if (!rounds.length) return;
  if (selectedRound > rounds.length) selectedRound = 1;
  const rnd = rounds[selectedRound - 1];
  const rows = rnd.rows;
  const budgetShare = Math.min(1, rnd.budget / Math.max(1e-9, rows.length * s.full_cost));

  renderPriority(rows);
  renderSplit(rows, rnd.budget, s);
  renderWaves(rnd.waves);
  $("safetySub").innerHTML = `Each batch is built in a shadow copy and <b>${s.canary_size}</b> popular logged searches are replayed against it. ` +
    `A batch that would make search worse is rolled back, and its budget is re-planned as exact re-embeds. Showing repair round ${selectedRound}.`;

  const key = `${s.injected}|${selectedRound}|${rows.length}|${rnd.budget}`;
  if (key !== lastCurveKey) {
    lastCurveKey = key;
    fetch(`/api/curve?round=${selectedRound}`).then((r) => r.json()).then((cv) => renderCurve(cv, budgetShare));
  }
}

function renderPriority(rows) {
  const counts = { full_reembed: 0, patch: 0, defer: 0 };
  rows.forEach((r) => counts[r.action]++);
  $("decisionLegend").innerHTML = Object.entries(DECISION).map(([k, d]) =>
    `<span><i class="sw" style="background:${d.color}"></i>${d.label}<b>${fmt(counts[k])}</b></span>`).join("");
  const floor = 0.01;
  const series = ["defer", "patch", "full_reembed"].map((a) => ({
    name: DECISION[a].label, type: "scatter",
    data: rows.filter((r) => r.action === a).map((r) => {
      const jitter = ((Math.sin(r.doc_index * 12.9898) * 43758.5453) % 1) * 0.015;
      return [Math.max(floor, r.traffic_vs_average), r.severity + jitter, r.traffic_vs_average, r.severity, r.doc_index];
    }),
    symbolSize: a === "defer" ? 6 : 9,
    itemStyle: { color: DECISION[a].color, opacity: a === "defer" ? 0.55 : 0.95, borderColor: "rgba(6,9,15,.8)", borderWidth: 1 },
  }));
  chart("priorityChart").setOption({
    animationDurationUpdate: 500,
    grid: { left: 54, right: 18, top: 30, bottom: 46 },
    tooltip: tooltip({ formatter: (p) => `<b>${p.seriesName}</b><br>Search traffic ${p.value[2].toFixed(2)}× average<br>Chance broken ${pct(p.value[3])}` }),
    xAxis: { ...axisCommon, type: "log", min: 0.007, name: "Search traffic (× average document)", nameLocation: "middle", nameGap: 30,
      axisLabel: { ...axisCommon.axisLabel, formatter: (v) => (v <= 0.0101 ? "≤0.01×" : `${+v.toPrecision(2)}×`) }, splitLine: { show: false } },
    yAxis: { ...axisCommon, min: 0.4, max: 1, interval: 0.1, name: "Chance broken", nameGap: 14, nameTextStyle: { ...axisCommon.nameTextStyle, align: "left" }, axisLabel: { ...axisCommon.axisLabel, formatter: (v) => `${Math.round(v * 100)}%` } },
    series,
  }, { replaceMerge: ["series"] });
}

function renderCurve(cv, budgetShare) {
  if (!cv.x.length) return;
  const x = cv.x, b = budgetShare * 100;
  const interp = (arr) => { for (let i = 1; i < x.length; i++) if (x[i] >= b) { const k = (b - x[i - 1]) / (x[i] - x[i - 1] || 1); return arr[i - 1] + k * (arr[i] - arr[i - 1]); } return arr[arr.length - 1]; };
  const atOurs = interp(cv.ours), atFifo = interp(cv.fifo);
  tween("hlOurs", atOurs, (v) => `${Math.round(v)}%`);
  tween("hlFifo", atFifo, (v) => `${Math.round(v)}%`);
  $("hlBudget").textContent = `${Math.round(b)}%`;
  chart("curveChart").setOption({
    animationDuration: 1200,
    grid: { left: 54, right: 24, top: 34, bottom: 46 },
    tooltip: tooltip({ trigger: "axis", formatter: (ps) => {
      const i = ps[0].dataIndex;
      return `Budget ${x[i].toFixed(0)}%<br><span style="color:${C.blue}">●</span> VectorSentry ${cv.ours[i].toFixed(0)}%<br><span style="color:${C.muted}">●</span> First-come-first-served ${cv.fifo[i].toFixed(0)}%`;
    } }),
    xAxis: { ...axisCommon, type: "value", min: 0, max: 100, name: "Budget (% of re-embedding everything)", nameLocation: "middle", nameGap: 30,
      axisLabel: { ...axisCommon.axisLabel, formatter: "{value}%" }, splitLine: { show: false } },
    yAxis: { ...axisCommon, min: 0, max: 100, name: "Affected traffic protected", nameGap: 14, nameTextStyle: { ...axisCommon.nameTextStyle, align: "left" }, axisLabel: { ...axisCommon.axisLabel, formatter: "{value}%" } },
    series: [
      // The shaded gap: VectorSentry's area, with first-come-first-served's
      // area painted over it in the card colour, leaves exactly the band between.
      { name: "oursArea", type: "line", data: x.map((v, i) => [v, cv.ours[i]]), showSymbol: false, lineStyle: { width: 0 }, silent: true, z: 1,
        areaStyle: { opacity: 1, color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: "rgba(57,135,229,.38)" }, { offset: 1, color: "rgba(57,135,229,.08)" }]) },
        tooltip: { show: false } },
      { name: "fifo", type: "line", data: x.map((v, i) => [v, cv.fifo[i]]), showSymbol: false, z: 2, lineStyle: { color: C.muted, width: 2 },
        areaStyle: { color: "rgb(14,19,30)", opacity: 1 } },
      { name: "fifoLabel", type: "scatter", data: [[x[Math.round(x.length * 0.62)], cv.fifo[Math.round(x.length * 0.62)]]], symbolSize: 0, silent: true, z: 4,
        label: { show: true, formatter: "First-come-first-served", position: "bottom", distance: 10, color: C.ink2, fontSize: 12, fontFamily: FONT },
        tooltip: { show: false } },
      { name: "ours", type: "line", data: x.map((v, i) => [v, cv.ours[i]]), showSymbol: false, z: 3,
        lineStyle: { color: C.blue, width: 3, shadowBlur: 14, shadowColor: "rgba(57,135,229,.7)" },
        markLine: { silent: true, symbol: "none", lineStyle: { color: C.ink2, width: 1, type: "solid" }, label: { show: false }, data: [{ xAxis: b }] },
        markPoint: { symbol: "circle", symbolSize: 10, label: { show: true, position: "left", distance: 8, fontSize: 14, fontWeight: 800, color: C.blue, formatter: `${Math.round(atOurs)}%` },
          itemStyle: { color: C.blue, borderColor: "#fff", borderWidth: 2 }, data: [{ coord: [b, atOurs] }] } },
      { name: "oursLabel", type: "scatter", data: [[x[Math.round(x.length * 0.62)], cv.ours[Math.round(x.length * 0.62)]]], symbolSize: 0, silent: true,
        label: { show: true, formatter: "VectorSentry", position: "bottom", distance: 10, color: C.blue, fontSize: 13, fontWeight: 700, fontFamily: FONT }, tooltip: { show: false } },
      { name: "budgetLabel", type: "scatter", data: [[b, 3]], symbolSize: 0, silent: true,
        label: { show: true, formatter: "This round's budget", position: b > 70 ? "left" : "right", distance: 6, color: C.ink2, fontSize: 11, fontFamily: FONT }, tooltip: { show: false } },
      { name: "fifoPoint", type: "scatter", data: [[b, atFifo]], symbolSize: 8, itemStyle: { color: C.muted, borderColor: "#fff", borderWidth: 1.5 },
        label: { show: true, position: "right", distance: 8, color: C.ink2, fontSize: 12, formatter: `${Math.round(atFifo)}%` }, tooltip: { show: false } },
    ],
  }, { replaceMerge: ["series"] });
}

function renderSplit(rows, budgetUnits, s) {
  const cost = { full_reembed: s.full_cost, patch: s.patch_cost };
  const spent = { full_reembed: 0, patch: 0 };
  const traffic = { full_reembed: 0, patch: 0, defer: 0 };
  rows.forEach((r) => { if (cost[r.action]) spent[r.action] += cost[r.action]; traffic[r.action] += r.severity * r.traffic_share; });
  const tTot = traffic.full_reembed + traffic.patch + traffic.defer || 1, bTot = Math.max(budgetUnits, 1e-9);
  const unspent = Math.max(0, bTot - spent.full_reembed - spent.patch);
  const seg = [
    ["Re-embed", C.blue, (traffic.full_reembed / tTot) * 100, (spent.full_reembed / bTot) * 100],
    ["Quick patch", C.orange, (traffic.patch / tTot) * 100, (spent.patch / bTot) * 100],
    ["Later / unspent", C.slate, (traffic.defer / tTot) * 100, (unspent / bTot) * 100],
  ];
  chart("splitChart").setOption({
    grid: { left: 110, right: 18, top: 8, bottom: 8 },
    tooltip: tooltip({ trigger: "item", formatter: (p) => `${p.seriesName}<br>${p.name}: ${p.value.toFixed(1)}%` }),
    xAxis: { type: "value", max: 100, show: false },
    yAxis: { type: "category", data: ["Affected traffic", "Budget"], axisLine: { show: false }, axisTick: { show: false },
      axisLabel: { color: C.ink2, fontFamily: FONT, fontSize: 12 } },
    series: seg.map(([name, color, t, b]) => ({
      name, type: "bar", stack: "s", barWidth: 22, data: [t, b],
      itemStyle: { color, borderColor: "rgba(16,22,34,1)", borderWidth: 2, borderRadius: 3 },
      label: { show: true, color: "#fff", fontFamily: FONT, fontSize: 12, fontWeight: 650,
        formatter: (p) => (p.value >= 8 ? `${p.seriesName} ${Math.round(p.value)}%` : p.value >= 4 ? `${Math.round(p.value)}%` : "") },
    })),
  }, { replaceMerge: ["series"] });
}

function renderWaves(waves) {
  if (!waves || !waves.length) { chart("wavesChart").clear(); return; }
  const data = waves.map((w, i) => ({
    value: w.n_documents, round: w.round, before: w.canary_before, after: w.canary_after, passed: w.passed,
    itemStyle: { color: w.passed ? C.good : C.critical, borderRadius: [4, 4, 0, 0] },
    label: { show: true, position: "top", formatter: w.passed ? "✓" : "✗", color: w.passed ? "#4ade80" : "#ff6b6b", fontSize: 13, fontWeight: 700 },
  }));
  const areas = [];
  const roundsSeen = [...new Set(waves.map((w) => w.round))];
  roundsSeen.forEach((r) => {
    const idx = waves.map((w, i) => (w.round === r ? i : -1)).filter((i) => i >= 0);
    const name = r === 1 ? "First plan" : r === 2 ? "Re-plan: rejected patches → exact re-embeds" : `Re-plan ${r - 1}`;
    areas.push([{ xAxis: `#${idx[0] + 1}`, name, itemStyle: { color: r % 2 ? "rgba(255,255,255,0.0)" : "rgba(57,135,229,0.07)" } }, { xAxis: `#${idx[idx.length - 1] + 1}` }]);
  });
  chart("wavesChart").setOption({
    grid: { left: 52, right: 18, top: 56, bottom: 40 },
    legend: { show: false },
    tooltip: tooltip({ formatter: (p) => `Batch ${p.dataIndex + 1} · ${p.data.passed ? "passed, went live" : "rejected, rolled back"}<br>${p.value} documents<br>Search health ${pct(p.data.before, 1)} → ${pct(p.data.after, 1)}` }),
    xAxis: { ...axisCommon, type: "category", data: waves.map((w, i) => `#${i + 1}`), name: "Batch, in the order it was tested", nameLocation: "middle", nameGap: 26, splitLine: { show: false } },
    yAxis: { ...axisCommon, name: "Documents per batch", nameGap: 14, nameTextStyle: { ...axisCommon.nameTextStyle, align: "left" } },
    series: [{
      type: "bar", data, barMaxWidth: 26,
      markArea: { silent: true, label: { color: C.ink2, fontSize: 11, position: "top", distance: 22, fontFamily: FONT }, data: areas },
    }],
  }, { replaceMerge: ["series"] });
}

/* ---------- incident report ---------- */
function renderReport(s) {
  const show = s.summary && (s.status === "resolved" || s.status === "needs_review");
  $("report").classList.toggle("hidden", !show);
  if (!show) return;
  const sm = s.summary, caught = Math.max(0, sm.initial_flagged - s.flagged_ref);
  $("report").innerHTML = `
    <div class="rh"><div class="kicker">Incident report</div><h2 style="margin:0">${phase(s) === "polishing" ? "Resolved, polishing patches" : "Resolved"}</h2></div>
    <div><div class="rv">${fmt(sm.injected)}</div><div class="rl">documents hit by the provider update</div></div>
    <div><div class="rv">${fmt(caught)} <span style="font-size:15px;color:${C.muted}">${pct(caught / Math.max(1, sm.injected))}</span></div><div class="rl">caught by the fingerprint detector</div></div>
    <div><div class="rv">${fmt(sm.remediated)}</div><div class="rl">repaired, every batch verified</div></div>
    <div><div class="rv">${s.health == null ? "–" : pct(s.health, 1)}</div><div class="rl">search health now</div></div>`;
}

/* ---------- loop ---------- */
let polling = false;
async function poll() {
  if (polling) return;
  polling = true;
  try {
    const r = await fetch("/api/state");
    S = await r.json();
    if (S.budget_level != null && !document.querySelector("#budgetSeg button.on")) setSeg("budgetSeg", S.budget_level);
    renderStatus(S); renderKpis(S); renderGauge(S); renderMap(S); renderTimeline(S); renderFeed(S); renderDecision(S); renderReport(S);
  } catch (e) {
    $("statusText").textContent = "Server unreachable";
    $("statusPill").className = "status-pill warn";
  } finally {
    polling = false;
  }
}

(async function init() {
  layout = await (await fetch("/api/layout")).json();
  await poll();
  setInterval(poll, 1000);
})();
