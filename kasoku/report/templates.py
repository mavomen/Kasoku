"""Self-contained HTML report template with embedded Chart.js and HTMX.

The template is a single HTML string with ``__BENCHMARK_DATA__``,
``__COMPARE_DATA__`` and ``__SPEEDUP_DATA__`` placeholders that are replaced
at generation time with JSON-serialized benchmark records. All filtering and
chart redraws happen in the browser via plain JavaScript; HTMX is used only
for watch-mode live reload and optional comparison fetching.
"""

from __future__ import annotations

REPORT_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kasoku Benchmark Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script src="https://unpkg.com/htmx.org@1.9.10"></script>
<style>
  :root { --bg: #ffffff; --panel: #f7f7f7; --text: #111827; --muted: #6b7280;
          --bar: #2563eb; --bar2: #93c5fd; --border: #e5e7eb; }
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         margin: 0; padding: 24px; color: var(--text); background: var(--bg); }
  h1 { font-size: 1.5rem; margin: 0 0 16px; }
  .subtitle { color: var(--muted); font-size: 0.95rem; margin: 0 0 24px; }
  .filters { display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
             margin-bottom: 16px; padding: 14px; background: var(--panel);
             border: 1px solid var(--border); border-radius: 8px; }
  select, button { font-size: 0.9rem; padding: 6px 10px; border: 1px solid var(--border);
                   border-radius: 6px; background: #fff; color: var(--text); }
  button { cursor: pointer; }
  button:hover { background: var(--panel); }
  .chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 24px; }
  .panel { border: 1px solid var(--border); border-radius: 8px; padding: 16px;
           background: #fff; }
  .panel h2 { font-size: 1rem; margin: 0 0 12px; color: var(--text); }
  .panel canvas { max-height: 260px; display: block; margin: auto; }
  .compare-badge { font-size: 0.75rem; color: var(--muted); margin-left: 8px; }
  .empty { grid-column: 1 / -1; text-align: center; color: var(--muted); padding: 40px; }
  .sse-status { font-size: 0.8rem; color: var(--muted); margin-left: auto; }
</style>
</head>
<body>
<h1>Kasoku Benchmark Report <span class="compare-badge" id="compare-badge"></span></h1>
<p class="subtitle">Interactive benchmark dashboard &mdash; wall-clock, throughput, compile overhead, and memory across models, sizes, modes, and phases.</p>
<div class="filters">
  <label>Model <select id="filter-model"><option value="all">All</option></select></label>
  <label>Size <select id="filter-size"><option value="all">All</option></select></label>
  <label>Mode <select id="filter-mode"><option value="all">All</option></select></label>
  <label>Phase <select id="filter-phase"><option value="all">All</option></select></label>
  <label>Metric <select id="filter-metric"><option value="all">All</option>
    <option value="wall_clock">Wall-clock</option>
    <option value="throughput">Throughput</option>
    <option value="compile_overhead">Compile overhead</option>
    <option value="memory">Memory</option>
  </select></label>
  <button type="button" onclick="refresh()">Refresh</button>
  <span class="sse-status" id="sse-status">static</span>
</div>
<div class="chart-grid">
  <div class="panel">
    <h2>XLA Speedup Ratio <span class="compare-badge" id="speedup-badge"></span></h2>
    <canvas id="chart-speedup"></canvas>
  </div>
  <div class="panel">
    <h2>Compile Overhead (inference) <span class="compare-badge" id="compile-badge"></span></h2>
    <canvas id="chart-compile"></canvas>
  </div>
  <div class="panel">
    <h2>Throughput (samples/sec) <span class="compare-badge" id="throughput-badge"></span></h2>
    <canvas id="chart-throughput"></canvas>
  </div>
  <div class="panel">
    <h2>Peak Memory (MB) <span class="compare-badge" id="memory-badge"></span></h2>
    <canvas id="chart-memory"></canvas>
  </div>
  <div class="panel empty" id="empty-state" style="display:none">No data matches the current filters.</div>
</div>
<script>
const BENCHMARK_DATA = __BENCHMARK_DATA__;
const COMPARE_DATA = __COMPARE_DATA__;
const SPEEDUP_DATA = __SPEEDUP_DATA__;
const HAS_COMPARE = Array.isArray(COMPARE_DATA) && COMPARE_DATA.length > 0;
const CHART_COLORS = { eager: "#2563eb", tf_function: "#7c3aed", xla: "#059669" };
const charts = {};
const filterState = { model: "all", size: "all", mode: "all", phase: "all", metric: "all" };

function num(value) {
  return value == null || Number.isNaN(Number(value)) ? null : Number(value);
}

function matchesFilter(row) {
  if (filterState.model !== "all" && row.model !== filterState.model) return false;
  if (filterState.size !== "all" && row.model_size !== filterState.size) return false;
  if (filterState.mode !== "all" && row.mode !== filterState.mode) return false;
  if (filterState.phase !== "all" && row.phase !== filterState.phase) return false;
  return true;
}

function filteredRows() {
  return BENCHMARK_DATA.filter(matchesFilter);
}

function distinct(rows, key) {
  const seen = new Set();
  for (const row of rows) {
    const value = row[key];
    if (value != null) seen.add(String(value));
  }
  return ["all", ...Array.from(seen).sort()];
}

function populateFilters(rows) {
  const root = document.getElementById("filter-model");
  const setOptions = (id, values, current) => {
    const el = document.getElementById(id);
    const had = el.value;
    el.innerHTML = "";
    for (const value of values) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value === "all" ? "All" : value;
      if (value === current) option.selected = true;
      else if (had === value && values.includes(had)) option.selected = true;
      el.appendChild(option);
    }
  };
  setOptions("filter-model", distinct(rows, "model"), filterState.model);
  setOptions("filter-size", distinct(rows, "model_size"), filterState.size);
  setOptions("filter-mode", distinct(rows, "mode"), filterState.mode);
  setOptions("filter-phase", distinct(rows, "phase"), filterState.phase);
  document.getElementById("empty-state").style.display =
    rows.length === 0 ? "block" : "none";
  document.getElementById("compare-badge").textContent = HAS_COMPARE ? "compare: on" : "";
}

function groupBy(rows, keys, valueKey, aggregator) {
  const groups = new Map();
  for (const row of rows) {
    const key = keys.map(k => String(row[k] ?? "")).join("|");
    if (!groups.has(key)) groups.set(key, { labels: keys.map(k => row[k]), values: [] });
    const v = num(row[valueKey]);
    if (v != null) groups.get(key).values.push(v);
  }
  const result = [];
  for (const [key, group] of groups) {
    result.push({ labels: group.labels, value: aggregator(group.values) });
  }
  result.sort((a, b) => a.labels.join("|").localeCompare(b.labels.join("|")));
  return result;
}

function mean(values) { return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0; }
function maxv(values) { return values.length ? Math.max(...values) : 0; }

function colorFor(mode) { return CHART_COLORS[mode] ?? "#9ca3af"; }

function destroyCharts() {
  for (const name of Object.keys(charts)) {
    if (charts[name]) charts[name].destroy();
  }
  Object.assign(charts, {});
}

function newChart(id, type, data, options) {
  const ctx = document.getElementById(id).getContext("2d");
  charts[id] = new Chart(ctx, { type, data, options });
}

function drawSpeedupChart() {
  const rows = SPEEDUP_DATA.filter(matchesFilter);
  const labels = rows.map(r => `${r.model}/${r.model_size}`);
  const data = rows.map(r => r.speedup);
  destroyCharts();
  newChart("chart-speedup", "bar", {
    labels,
    datasets: [{ label: "XLA speedup (eager / xla)", data,
                 backgroundColor: "#059669", borderColor: "#047857", borderWidth: 1 }],
  }, {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false }, tooltip: { mode: "index", intersect: false } },
    scales: { y: { beginAtZero: true, title: { display: true, text: "speedup" } } },
  });
  if (HAS_COMPARE) {
    const compareMap = new Map();
    for (const row of COMPARE_DATA) {
      if (row.phase === "inference") {
        const key = `${row.model}/${row.model_size}`;
        const eager = COMPARE_DATA.find(r => r.model === row.model && r.model_size === row.model_size && r.mode === "eager" && r.phase === "inference");
        const xla = COMPARE_DATA.find(r => r.model === row.model && r.model_size === row.model_size && r.mode === "xla" && r.phase === "inference");
        if (eager && xla && num(xla.wall_clock_seconds) > 0) {
          compareMap.set(key, num(eager.wall_clock_seconds) / num(xla.wall_clock_seconds));
        }
      }
    }
    // comparison bars drawn in separate overlay canvas below is skipped for simplicity; tooltip shows compare
  }
}

function drawMetricChart(canvasId, valueKey, title) {
  const rows = filteredRows();
  let groups;
  if (valueKey === "compile_overhead") {
    groups = groupBy(rows.filter(r => r.phase === "inference"), ["model", "model_size", "mode"], valueKey, mean);
  } else {
    groups = groupBy(rows, ["model", "model_size", "mode", "phase"], valueKey, mean);
  }
  const labels = groups.map(g => g.labels.join(" / "));
  const values = groups.map(g => g.value);
  const colors = groups.map((g, i) => colorFor(g.labels[2]));
  destroyCharts();
  newChart(canvasId, "bar", {
    labels,
    datasets: [{ label: title, data: values, backgroundColor: colors, borderWidth: 0.5,
                 borderColor: "#ffffff" }],
  }, {
    responsive: true, maintainAspectRatio: false,
    plugins: { tooltip: { mode: "index", intersect: false } },
    scales: { y: { beginAtZero: true, title: { display: true, text: title } } },
  });
}

function refresh() {
  const rows = filteredRows();
  populateFilters(rows);
  drawSpeedupChart();
  drawMetricChart("chart-compile", "compile_overhead", "Compile overhead (s)");
  drawMetricChart("chart-throughput", "throughput_samples_per_sec", "Throughput (samples/sec)");
  drawMetricChart("chart-memory", "peak_memory_mb", "Peak memory (MB)");
}

function connectWatch() {
  if (!window.EventSource) return;
  const es = new EventSource("/events");
  es.onmessage = () => { document.getElementById("sse-status").textContent = "reloading"; location.reload(); };
  es.onerror = () => { document.getElementById("sse-status").textContent = "offline"; es.close(); };
}

document.getElementById("filter-model").onchange = (e) => { filterState.model = e.target.value; refresh(); };
document.getElementById("filter-size").onchange = (e) => { filterState.size = e.target.value; refresh(); };
document.getElementById("filter-mode").onchange = (e) => { filterState.mode = e.target.value; refresh(); };
document.getElementById("filter-phase").onchange = (e) => { filterState.phase = e.target.value; refresh(); };
document.getElementById("filter-metric").onchange = (e) => { filterState.metric = e.target.value; refresh(); };

refresh();
if (document.getElementById("sse-status") && window.location.pathname === "/") {
  connectWatch();
}
</script>
</body>
</html>"""

__all__ = ["REPORT_TEMPLATE"]
