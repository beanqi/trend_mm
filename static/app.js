const $ = (id) => document.getElementById(id);

function fmtAge(ms) {
  if (ms == null) return "—";
  if (ms < 1000) return `${ms} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

function fmtTime(ms) {
  if (!ms) return "—";
  return new Date(ms).toISOString().replace("T", " ").slice(0, 19) + "Z";
}

async function refreshStatus() {
  const r = await fetch("/api/status");
  const s = await r.json();
  $("connection").textContent = s.connection;
  $("data-age").textContent = fmtAge(s.data_age_ms);
  $("phase").textContent = s.phase;
  $("phase").className = `phase-${s.phase}`;
  $("training").textContent = s.training ? "in progress" : "idle";
  $("model").textContent = `${s.model_version} (${s.model_status})`;
  $("model").className = `status-${s.model_status}`;
  $("next-train").textContent = fmtTime(s.next_train_ms);
  $("alarm").textContent = s.alarm || "none";
  $("granularity-note").textContent = s.depth_granularity_note
    || "Depth is ~20ms batches; 10ms order-flow factors are not true 10ms event-time.";
}

function renderMetrics(id, rows) {
  const el = $(id);
  if (!rows || !rows.length) {
    el.textContent = "no samples";
    return;
  }
  el.innerHTML = rows.map((m) => {
    const ic = m.ic == null ? "—" : m.ic.toFixed(3);
    const mae = m.mae == null ? "—" : m.mae.toFixed(3);
    const rmse = m.rmse == null ? "—" : m.rmse.toFixed(3);
    const dir = m.dir_acc == null ? "—" : (m.dir_acc * 100).toFixed(1) + "%";
    return `<p><strong>${m.horizon}</strong> cov ${(m.coverage * 100).toFixed(1)}% · IC ${ic} · MAE ${mae} · RMSE ${rmse} · dir ${dir}</p>`;
  }).join("");
}

async function refreshMetrics() {
  const r = await fetch("/api/metrics");
  const j = await r.json();
  renderMetrics("m-1h", j["1h"]);
  renderMetrics("m-24h", j["24h"]);
  renderMetrics("m-7d", j["7d"]);
}

async function refreshModels() {
  const r = await fetch("/api/models");
  const j = await r.json();
  const tb = $("models").querySelector("tbody");
  tb.innerHTML = (j.models || []).map((m) => {
    const win = (m.train_start_ms && m.train_end_ms)
      ? `${fmtTime(m.train_start_ms)} → ${fmtTime(m.train_end_ms)}`
      : "cold start";
    let metrics = m.metrics_json || "";
    try {
      const parsed = JSON.parse(m.metrics_json || "{}");
      if (parsed.reason) metrics = parsed.reason;
    } catch (_) {}
    return `<tr>
      <td>${m.version}</td>
      <td class="status-${m.status}">${m.status}</td>
      <td>${win}</td>
      <td>${metrics}</td>
    </tr>`;
  }).join("");
}

function drawCurves(points) {
  const c = $("curve");
  const ctx = c.getContext("2d");
  const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);
  if (!points.length) return;
  const mids = points.map((p) => p.mid);
  const preds = points.map((p) => p.pred);
  const acts = points.map((p) => p.actual);
  const minM = Math.min(...mids), maxM = Math.max(...mids);
  const extent = (vals) => Math.max(0.1, ...vals.filter((v) => v != null).map((v) => Math.abs(v)));
  const predExtent = extent(preds);
  const actualExtent = extent(acts);
  const x = (i) => (i / Math.max(points.length - 1, 1)) * (w - 20) + 10;
  const yMid = (v) => 78 - ((v - minM) / (maxM - minM + 1e-9)) * 58;
  const yTrack = (v, center, halfHeight, scale) => center - (v / scale) * halfHeight;
  const line = (vals, color, yfn) => {
    ctx.beginPath();
    ctx.strokeStyle = color;
    let started = false;
    vals.forEach((v, i) => {
      if (v == null) { started = false; return; }
      const xx = x(i), yy = yfn(v);
      if (!started) { ctx.moveTo(xx, yy); started = true; }
      else ctx.lineTo(xx, yy);
    });
    ctx.stroke();
  };
  ctx.font = "11px system-ui";
  ctx.fillStyle = "#6cb6ff";
  ctx.fillText(`mid ${minM.toFixed(1)} … ${maxM.toFixed(1)}`, 12, 13);
  ctx.strokeStyle = "#334155";
  ctx.beginPath(); ctx.moveTo(10, 140); ctx.lineTo(w - 10, 140); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(10, 230); ctx.lineTo(w - 10, 230); ctx.stroke();
  ctx.fillStyle = "#f0c14b";
  ctx.fillText(`pred ±${predExtent.toFixed(2)} tick`, 12, 101);
  ctx.fillStyle = "#3dd68c";
  ctx.fillText(`actual ±${actualExtent.toFixed(2)} tick`, 12, 191);
  line(mids, "#6cb6ff", yMid);
  line(preds, "#f0c14b", (v) => yTrack(v, 140, 34, predExtent));
  line(acts, "#3dd68c", (v) => yTrack(v, 230, 34, actualExtent));
}

async function refreshSamples() {
  const h = $("horizon").value;
  const to = Date.now();
  const from = to - 5 * 60 * 1000;
  const r = await fetch(`/api/samples?from_ms=${from}&to_ms=${to}&horizon=${encodeURIComponent(h)}&limit=600`);
  const j = await r.json();
  drawCurves(j.points || []);
}

function renderFactors(frame) {
  const tb = $("factors").querySelector("tbody");
  const names = frame.factor_names || [];
  const xs = frame.factors || [];
  const w = frame.weights || [[], [], []];
  const c = frame.contributions || [[], [], []];
  let html = "";
  for (let i = 0; i < 17; i++) {
    html += `<tr>
      <td>${i + 1}</td><td>${names[i] || ""}</td>
      <td>${(xs[i] ?? 0).toFixed(3)}</td>
      <td>${(w[0]?.[i] ?? 0).toFixed(3)}</td><td>${(c[0]?.[i] ?? 0).toFixed(3)}</td>
      <td>${(w[1]?.[i] ?? 0).toFixed(3)}</td><td>${(c[1]?.[i] ?? 0).toFixed(3)}</td>
      <td>${(w[2]?.[i] ?? 0).toFixed(3)}</td><td>${(c[2]?.[i] ?? 0).toFixed(3)}</td>
    </tr>`;
  }
  tb.innerHTML = html;
}

function connectStream() {
  const es = new EventSource("/api/stream");
  es.onmessage = (ev) => {
    try { renderFactors(JSON.parse(ev.data)); } catch (_) {}
  };
}

$("horizon").addEventListener("change", refreshSamples);
refreshStatus();
refreshMetrics();
refreshModels();
refreshSamples();
connectStream();
setInterval(refreshStatus, 1000);
setInterval(refreshMetrics, 15000);
setInterval(refreshModels, 15000);
setInterval(refreshSamples, 2000);
