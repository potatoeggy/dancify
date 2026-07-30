const API = "/_dev/scoring/api";
const state = { routineID: null, windowIndex: null, reference: null, attempts: [], sequence: 0 };
const byID = (id) => document.getElementById(id);
const signalKeys = ["horizontalComponent", "verticalComponent", "linearIntensity"];
const signalClasses = ["horizontal", "vertical", "intensity"];

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

async function requestJSON(url, options) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error?.message || `Request failed (${response.status})`);
  return body;
}

function setStatus(message, error = false) {
  const node = byID("status");
  node.textContent = message;
  node.className = error ? "error" : "";
}

function option(value, label, disabled = false) {
  const node = document.createElement("option");
  node.value = value;
  node.textContent = label;
  node.disabled = disabled;
  return node;
}

async function loadRoutines() {
  try {
    const body = await requestJSON(`${API}/routines`);
    const select = byID("routine");
    select.replaceChildren(option("", body.routines.length ? "Select a routine" : "No routines available"));
    for (const routine of body.routines) {
      select.append(option(routine.routineID, `${routine.title} (${routine.duration.toFixed(2)}s)`));
    }
    setStatus("Choose a routine and scoreable window.");
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function loadWindows() {
  state.routineID = byID("routine").value || null;
  state.windowIndex = null;
  state.reference = null;
  clearAttempts();
  clearReference();
  const select = byID("window");
  select.replaceChildren(option("", state.routineID ? "Loading windows…" : "Select a routine"));
  select.disabled = true;
  if (!state.routineID) return;
  try {
    const body = await requestJSON(`${API}/routines/${encodeURIComponent(state.routineID)}/windows`);
    select.replaceChildren(option("", "Select a window"));
    for (const windowValue of body.windows) {
      const duration = windowValue.endTime - windowValue.startTime;
      const label = `Window ${windowValue.index} · ${windowValue.startTime.toFixed(2)}–${windowValue.endTime.toFixed(2)}s${windowValue.scoreable ? "" : " · not scoreable"}`;
      select.append(option(String(windowValue.index), label, !windowValue.scoreable || duration <= 0));
    }
    select.disabled = false;
  } catch (error) {
    setStatus(error.message, true);
  }
}

function renderScoringConfig(config) {
  const weights = `${config.directionWeight.toFixed(3)}/${config.magnitudeWeight.toFixed(3)}/${config.timingWeight.toFixed(3)}`;
  const timingSource = config.sampleSynchronizedTiming ? "sample timestamps" : "normalized indices";
  const coverageRamp = config.smoothCoverageRamp ? "smooth" : "linear legacy";
  byID("scoring-config").textContent = `Active scoring (read-only): ${config.profile} profile · direction/magnitude/timing ${weights} · DTW radius ${config.sakoeChibaRadius} · timing grace/falloff ${(config.timingGraceSeconds * 1000).toFixed(0)}/${(config.timingFalloffSeconds * 1000).toFixed(0)} ms · timing path cost ${config.timingPathCostWeight.toFixed(3)} · valid/full coverage ${(config.minimumCoverage * 100).toFixed(0)}/${(config.fullCoverage * 100).toFixed(0)}% · coverage/sample floors ${(config.coverageQualityFloor * 100).toFixed(0)}/${(config.sampleQualityFloor * 100).toFixed(0)}% · coverage ramp ${coverageRamp} · sample rate ${config.sampleRateHz} Hz · resample gap ${(config.resampleMaxGapSeconds * 1000).toFixed(0)} ms · resampling timestamps ${config.resamplingTimestampMode} · timing source ${timingSource}. MOCK controls cannot change these settings.`;
}

async function loadReference() {
  const raw = byID("window").value;
  state.windowIndex = raw === "" ? null : Number(raw);
  state.reference = null;
  clearAttempts();
  clearReference();
  if (state.windowIndex === null) return;
  try {
    const body = await requestJSON(`${API}/routines/${encodeURIComponent(state.routineID)}/windows/${state.windowIndex}`);
    state.reference = body;
    byID("signal-meaning").textContent = body.signalMeaning;
    renderScoringConfig(body.scoringConfig);
    renderWrists(body.availableWrists);
    renderPlots();
    renderSignalTable(body.referenceSignals);
    byID("run").disabled = !body.window.scoreable || body.availableWrists.length === 0;
    setStatus(body.window.scoreable ? "Ready to run a stateless MOCK attempt." : "This partial window is not scoreable.");
  } catch (error) {
    setStatus(error.message, true);
  }
}

function clearReference() {
  byID("plots").replaceChildren();
  byID("signal-table").replaceChildren();
  byID("wrists").textContent = "Select a scoreable window.";
  byID("run").disabled = true;
}

function renderWrists(wrists) {
  const container = byID("wrists");
  container.replaceChildren();
  for (const wrist of wrists) {
    const label = element("label", undefined);
    const input = document.createElement("input");
    input.type = "checkbox";
    input.name = "activeWrists";
    input.value = wrist;
    input.checked = true;
    label.append(input, document.createTextNode(` ${wrist} wrist`));
    container.append(label);
  }
}

function pathFor(points, key, width, height, maximum) {
  const usable = points.filter((point) => Number.isFinite(point[key]));
  if (!usable.length) return "";
  return usable.map((point, index) => {
    const x = 35 + Math.max(0, Math.min(1, point.offsetSeconds)) * (width - 50);
    const normalized = key === "linearIntensity" ? point[key] / maximum : (point[key] / maximum + 1) / 2;
    const y = 10 + (1 - normalized) * (height - 35);
    return `${index ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
}

function svgLine(className, path) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", "path");
  node.setAttribute("class", className);
  node.setAttribute("d", path);
  return node;
}

function renderPlots(selectedAttempt = state.attempts[0]) {
  const container = byID("plots");
  container.replaceChildren();
  if (!state.reference) return;
  for (const wrist of state.reference.availableWrists) {
    const panel = element("article", undefined, "plot");
    const selectionLabel = selectedAttempt ? `reference vs selected MOCK attempt ${selectedAttempt.sequence}` : "reference only";
    panel.append(element("h3", `${wrist[0].toUpperCase()}${wrist.slice(1)} wrist signal · ${selectionLabel}`));
    const provenance = selectedAttempt
      ? `Solid = reference; dashed = selected MOCK attempt ${selectedAttempt.sequence}.`
      : "Solid = reference; no MOCK attempt selected.";
    panel.append(element("p", provenance, "plot-provenance"));
    const legend = element("div", undefined, "legend");
    legend.append(element("span", "horizontal component", "horizontal"), element("span", "vertical component", "vertical"), element("span", "intensity", "intensity"));
    panel.append(legend);
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 800 260");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", `${wrist} reference wrist acceleration-derived signals${selectedAttempt ? ` with selected MOCK attempt ${selectedAttempt.sequence} overlay` : ""}`);
    const reference = state.reference.referenceSignals[wrist] || [];
    const performance = selectedAttempt?.body.performanceSignals[wrist] || [];
    const all = reference.concat(performance);
    const maximum = Math.max(1, ...all.map((point) => Math.max(Math.abs(point.horizontalComponent || 0), Math.abs(point.verticalComponent || 0), point.linearIntensity || 0)));
    for (const y of [10, 122.5, 235]) {
      const axis = document.createElementNS("http://www.w3.org/2000/svg", "line");
      axis.setAttribute("class", "axis"); axis.setAttribute("x1", "35"); axis.setAttribute("x2", "785"); axis.setAttribute("y1", String(y)); axis.setAttribute("y2", String(y)); svg.append(axis);
    }
    signalKeys.forEach((key, index) => {
      svg.append(svgLine(`reference-${signalClasses[index]}`, pathFor(reference, key, 800, 260, maximum)));
      if (performance.length) svg.append(svgLine(`performance-${signalClasses[index]}`, pathFor(performance, key, 800, 260, maximum)));
    });
    panel.append(svg);
    container.append(panel);
  }
}

function renderSignalTable(signals) {
  const container = byID("signal-table");
  container.replaceChildren();
  for (const [wrist, points] of Object.entries(signals)) {
    if (!points.length) continue;
    const table = element("table");
    table.append(element("caption", `${wrist} reference wrist — first 12 of ${points.length} canonical samples`));
    const head = element("thead"); const row = element("tr");
    ["Offset (s)", "Horizontal", "Vertical", "Intensity"].forEach((text) => row.append(element("th", text)));
    head.append(row); table.append(head);
    const body = element("tbody");
    for (const point of points.slice(0, 12)) {
      const item = element("tr");
      [point.offsetSeconds, point.horizontalComponent, point.verticalComponent, point.linearIntensity].forEach((value) => item.append(element("td", value === null ? "unavailable" : Number(value).toFixed(3))));
      body.append(item);
    }
    table.append(body); container.append(table);
  }
}

function perturbationValues() {
  const result = {};
  for (const name of ["directionRotationDegrees", "intensityScale", "timeShiftMs", "captureCoverage", "sampleQuality", "horizontalConfidence"]) {
    result[name] = Number(byID(name).value);
  }
  return result;
}

async function submitAttempt(event) {
  event.preventDefault();
  const activeWrists = [...document.querySelectorAll('input[name="activeWrists"]:checked')].map((node) => node.value);
  if (!activeWrists.length) { setStatus("Select at least one available wrist.", true); return; }
  byID("run").disabled = true;
  setStatus("Running MOCK attempt through production scoring…");
  try {
    const body = await requestJSON(`${API}/routines/${encodeURIComponent(state.routineID)}/windows/${state.windowIndex}/attempts`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ activeWrists, perturbation: perturbationValues() })
    });
    const attempt = { sequence: ++state.sequence, body };
    state.attempts.unshift(attempt);
    state.attempts = state.attempts.slice(0, 20);
    renderMetrics(attempt);
    renderHistory();
    renderPlots(attempt);
    setStatus(`MOCK attempt ${attempt.sequence} scored. No gameplay/session data was stored.`);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    byID("run").disabled = !state.reference;
  }
}

function percentage(value) { return `${(value * 100).toFixed(1)}%`; }
function renderMetrics(attempt) {
  const metrics = attempt.body.metrics;
  const container = byID("metrics"); container.replaceChildren();
  const values = [["MOCK score", metrics.score.toFixed(1)], ["Validity", metrics.valid ? "VALID" : "INVALID"], ["Coverage", percentage(metrics.coverage)], ["Quality", percentage(metrics.quality)], ["Direction", percentage(metrics.breakdown.direction)], ["Intensity", percentage(metrics.breakdown.magnitude)], ["Timing", percentage(metrics.breakdown.timing)]];
  for (const [label, value] of values) { const card = element("div", label, "metric"); card.prepend(element("strong", value)); container.append(card); }
}

function renderHistory() {
  const body = byID("history"); body.replaceChildren();
  if (!state.attempts.length) { const row = element("tr"); const cell = element("td", "No MOCK attempts yet."); cell.colSpan = 8; row.append(cell); body.append(row); return; }
  for (const attempt of state.attempts) {
    const metrics = attempt.body.metrics; const row = element("tr"); const first = element("td");
    const button = element("button", `MOCK ${attempt.sequence}`); button.type = "button"; button.addEventListener("click", () => { renderMetrics(attempt); renderPlots(attempt); setStatus(`Showing MOCK attempt ${attempt.sequence}.`); }); first.append(button); row.append(first);
    [metrics.score.toFixed(1), metrics.valid ? "yes" : "no", percentage(metrics.coverage), percentage(metrics.quality), percentage(metrics.breakdown.direction), percentage(metrics.breakdown.magnitude), percentage(metrics.breakdown.timing)].forEach((value) => row.append(element("td", value)));
    body.append(row);
  }
}

function resetBaseline() {
  const defaults = { directionRotationDegrees: 0, intensityScale: 1, timeShiftMs: 0, captureCoverage: 1, sampleQuality: 1, horizontalConfidence: 1 };
  for (const [name, value] of Object.entries(defaults)) {
    byID(name).value = value;
    document.querySelector(`[data-for="${name}"]`).value = value;
  }
  document.querySelectorAll('input[name="activeWrists"]').forEach((node) => { node.checked = true; });
  setStatus("Baseline MOCK controls restored.");
}

function clearAttempts() {
  state.attempts = []; state.sequence = 0;
  byID("metrics").replaceChildren();
  renderHistory(); renderPlots();
}

for (const range of document.querySelectorAll('#perturbations input[type="range"]')) {
  const numeric = document.querySelector(`[data-for="${range.id}"]`);
  range.addEventListener("input", () => { numeric.value = range.value; });
  numeric.addEventListener("input", () => { range.value = numeric.value; });
}
byID("routine").addEventListener("change", loadWindows);
byID("window").addEventListener("change", loadReference);
byID("attempt-form").addEventListener("submit", submitAttempt);
byID("reset").addEventListener("click", resetBaseline);
byID("clear").addEventListener("click", () => { clearAttempts(); setStatus("MOCK attempt history cleared from this tab."); });
loadRoutines();
