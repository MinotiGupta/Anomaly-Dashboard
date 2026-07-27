/**
 * AnomalyScope – Frontend Application Logic
 * Handles experiment browsing, data visualisation (Plotly.js),
 * anomaly detection invocation, and human-in-the-loop feedback.
 */

// ── Global state ──
const state = {
  currentExperiment: null,
  currentFile: null,
  fileKey: null,
  timestamps: [],
  channels: {},
  channelNames: [],
  selectedChannel: null,
  anomalyResult: null,
  feedback: {},       // index -> { label, note }
  allChannelsVisible: true,
};

// ── Bootstrap ──
document.addEventListener("DOMContentLoaded", () => {
  loadExperiments();
  updateAlgoParams();
});

// ── API helpers ──
async function api(url, body = null) {
  const opts = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : { method: "GET" };
  const res = await fetch(url, opts);
  return res.json();
}

// ── Toast notifications ──
function toast(msg, type = "info") {
  const el = document.createElement("div");
  const icons = { success: "✅", error: "❌", info: "ℹ️" };
  el.className = `toast ${type}`;
  el.innerHTML = `<span>${icons[type] || "ℹ️"}</span><span>${msg}</span>`;
  document.getElementById("toast-container").appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

// ── Load experiments ──
async function loadExperiments() {
  const experiments = await api("/api/experiments");
  const tree = document.getElementById("experiment-tree");
  if (!experiments.length) {
    tree.innerHTML = '<p style="font-size:12px;color:var(--text-muted)">No experiments found. Upload data to begin.</p>';
    return;
  }
  tree.innerHTML = experiments.map(exp => `
    <div class="exp-folder" id="folder-${exp.name.replace(/\s+/g, '_')}">
      <div class="exp-folder-header" onclick="toggleFolder('${exp.name.replace(/\s+/g, '_')}')">
        <span class="icon">📁</span>
        <span>${exp.name}</span>
        <span style="margin-left:auto;font-size:11px;color:var(--text-muted)">${exp.files.length}</span>
      </div>
      <div class="exp-folder-files">
        ${exp.files.map(f => `
          <div class="exp-file" onclick="selectFile('${exp.name}','${f}')" data-exp="${exp.name}" data-file="${f}">
            <span>📄</span>
            <span>${f.length > 35 ? f.substring(0, 32) + '…' : f}</span>
          </div>
        `).join('')}
      </div>
    </div>
  `).join('');
}

function toggleFolder(id) {
  document.getElementById(`folder-${id}`).classList.toggle("open");
}

// ── Select file ──
async function selectFile(experiment, filename) {
  // Mark active
  document.querySelectorAll(".exp-file").forEach(el => el.classList.remove("active"));
  const active = document.querySelector(`.exp-file[data-exp="${experiment}"][data-file="${filename}"]`);
  if (active) active.classList.add("active");

  state.currentExperiment = experiment;
  state.currentFile = filename;
  state.anomalyResult = null;
  state.feedback = {};

  toast(`Loading ${filename}…`, "info");

  try {
    const data = await api("/api/load", { experiment, filename });
    if (data.error) { toast(data.error, "error"); return; }

    state.timestamps = data.timestamps;
    state.channels = data.channels;
    state.channelNames = data.channel_names;
    state.fileKey = data.file_key;

    // Populate channel selector
    const sel = document.getElementById("select-channel");
    sel.innerHTML = state.channelNames.map(c => `<option value="${c}">${c}</option>`).join('');
    state.selectedChannel = state.channelNames[0];

    // Show sections
    document.getElementById("empty-state").style.display = "none";
    document.getElementById("stats-section").style.display = "block";
    document.getElementById("chart-section").style.display = "block";
    document.getElementById("algo-section").style.display = "block";
    document.getElementById("anomaly-section").style.display = "none";
    document.getElementById("btn-export").style.display = "none";

    await loadStats();
    renderChart();
    toast(`Loaded ${data.num_points} data points`, "success");

  } catch (e) {
    toast("Failed to load file", "error");
    console.error(e);
  }
}

// ── Stats ──
async function loadStats() {
  const channel = document.getElementById("select-channel").value || state.channelNames[0];
  const data = await api("/api/stats", {
    experiment: state.currentExperiment,
    filename: state.currentFile,
    channel,
  });
  if (data.error) return;

  const grid = document.getElementById("stats-grid");
  grid.innerHTML = `
    <div class="stat-card">
      <div class="stat-value">${data.count}</div>
      <div class="stat-label">Data Points</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${data.mean.toFixed(2)}</div>
      <div class="stat-label">Mean</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${data.std.toFixed(2)}</div>
      <div class="stat-label">Std Dev</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${data.min.toFixed(2)}</div>
      <div class="stat-label">Min</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${data.max.toFixed(2)}</div>
      <div class="stat-label">Max</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${data.median.toFixed(2)}</div>
      <div class="stat-label">Median</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${data.skewness.toFixed(3)}</div>
      <div class="stat-label">Skewness</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${data.kurtosis.toFixed(3)}</div>
      <div class="stat-label">Kurtosis</div>
    </div>
  `;
}

// ── Chart ──
const CHANNEL_COLORS = [
  '#6366f1', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b',
  '#ef4444', '#ec4899', '#14b8a6', '#f97316', '#84cc16',
  '#a78bfa', '#22d3ee', '#fb7185', '#34d399',
];

function renderChart() {
  const traces = state.channelNames.map((ch, i) => ({
    x: state.timestamps,
    y: state.channels[ch],
    name: ch,
    type: 'scattergl',
    mode: 'lines',
    line: { color: CHANNEL_COLORS[i % CHANNEL_COLORS.length], width: 1.5 },
    connectgaps: false,
  }));

  // If anomalies exist, overlay them
  if (state.anomalyResult) {
    const ar = state.anomalyResult;
    traces.push({
      x: ar.anomaly_timestamps,
      y: ar.anomaly_values,
      name: `Anomalies (${ar.algorithm})`,
      type: 'scattergl',
      mode: 'markers',
      marker: {
        color: '#f43f5e',
        size: 8,
        symbol: 'diamond',
        line: { color: '#fff', width: 1 },
      },
    });
  }

  const layout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(13,20,37,0.8)',
    font: { family: 'Inter', color: '#8b95b0', size: 12 },
    margin: { t: 30, r: 30, b: 50, l: 60 },
    xaxis: {
      gridcolor: 'rgba(99,130,255,0.08)',
      title: 'Time',
    },
    yaxis: {
      gridcolor: 'rgba(99,130,255,0.08)',
      title: 'Value',
    },
    legend: {
      bgcolor: 'rgba(0,0,0,0)',
      font: { size: 11 },
    },
    hovermode: 'x unified',
  };

  Plotly.react('main-chart', traces, layout, { responsive: true });

  document.getElementById("chart-title").textContent =
    `${state.currentExperiment} / ${state.currentFile}`;
}

function toggleAllChannels() {
  state.allChannelsVisible = !state.allChannelsVisible;
  const visibility = state.channelNames.map(() => state.allChannelsVisible);
  Plotly.restyle('main-chart', { visible: visibility });
}

// ── Algorithm params UI ──
function updateAlgoParams() {
  const algo = document.getElementById("select-algo").value;
  const container = document.getElementById("algo-params");
  const configs = {
    zscore: [
      { id: 'param-threshold', label: 'Threshold', min: 1, max: 6, step: 0.5, value: 3, key: 'threshold' },
    ],
    iqr: [
      { id: 'param-factor', label: 'IQR Factor', min: 0.5, max: 5, step: 0.25, value: 1.5, key: 'factor' },
    ],
    isolation_forest: [
      { id: 'param-contamination', label: 'Contamination', min: 0.01, max: 0.5, step: 0.01, value: 0.05, key: 'contamination' },
    ],
    lof: [
      { id: 'param-neighbors', label: 'Neighbors', min: 5, max: 100, step: 5, value: 20, key: 'n_neighbors' },
      { id: 'param-contamination-lof', label: 'Contamination', min: 0.01, max: 0.5, step: 0.01, value: 0.05, key: 'contamination' },
    ],
    rolling_stats: [
      { id: 'param-window', label: 'Window', min: 5, max: 100, step: 5, value: 20, key: 'window' },
      { id: 'param-sigma', label: 'Sigma', min: 1, max: 6, step: 0.5, value: 3, key: 'sigma' },
    ],
  };

  const params = configs[algo] || [];
  container.innerHTML = params.map(p => `
    <div class="param-row">
      <label for="${p.id}">${p.label}</label>
      <input type="range" id="${p.id}" min="${p.min}" max="${p.max}" step="${p.step}" value="${p.value}"
        data-key="${p.key}" oninput="document.getElementById('${p.id}-val').textContent=this.value">
      <span class="param-val" id="${p.id}-val">${p.value}</span>
    </div>
  `).join('');
}

// ── Run detection ──
async function runDetection() {
  if (!state.currentFile) { toast("Load a file first", "error"); return; }

  const channel = document.getElementById("select-channel").value;
  const algorithm = document.getElementById("select-algo").value;

  // Gather params
  const params = {};
  document.querySelectorAll("#algo-params input[type=range]").forEach(inp => {
    params[inp.dataset.key] = parseFloat(inp.value);
  });

  toast(`Running ${algorithm} on ${channel}…`, "info");

  try {
    const data = await api("/api/detect", {
      experiment: state.currentExperiment,
      filename: state.currentFile,
      channel,
      algorithm,
      params,
    });

    if (data.error) { toast(data.error, "error"); return; }

    state.anomalyResult = data;
    state.selectedChannel = channel;
    state.feedback = {};

    // Load existing feedback
    const existingFb = await api("/api/feedback/get", { file_key: state.fileKey, channel });
    if (Array.isArray(existingFb)) {
      existingFb.forEach(fb => { state.feedback[fb.index] = { label: fb.label, note: fb.note || '' }; });
    }

    // Update chart
    renderChart();

    // Update stats
    const statsGrid = document.getElementById("stats-grid");
    const anomalyCard = `
      <div class="stat-card anomaly">
        <div class="stat-value">${data.anomaly_count}</div>
        <div class="stat-label">Anomalies</div>
      </div>
      <div class="stat-card success">
        <div class="stat-value">${(100 - (data.anomaly_count / data.total_points * 100)).toFixed(1)}%</div>
        <div class="stat-label">Normal</div>
      </div>
    `;
    // Replace last two cards or append
    statsGrid.innerHTML = statsGrid.innerHTML + anomalyCard;

    // Show anomaly table
    renderAnomalyTable(data);
    document.getElementById("anomaly-section").style.display = "block";
    document.getElementById("feedback-section").style.display = "block";
    document.getElementById("btn-export").style.display = "inline-flex";

    toast(`Found ${data.anomaly_count} anomalies out of ${data.total_points} points`, "success");
    updateFeedbackSummary();

  } catch (e) {
    toast("Detection failed", "error");
    console.error(e);
  }
}

// ── Anomaly table ──
function renderAnomalyTable(data) {
  const tbody = document.getElementById("anomaly-tbody");
  document.getElementById("anomaly-count-badge").textContent = data.anomaly_count;

  if (!data.anomaly_indices.length) {
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--text-muted);padding:24px">No anomalies detected</td></tr>';
    return;
  }

  tbody.innerHTML = data.anomaly_indices.map((idx, i) => {
    const fb = state.feedback[idx] || {};
    const confirmedClass = fb.label === 'anomaly' ? 'confirmed' : '';
    const rejectedClass = fb.label === 'normal' ? 'rejected' : '';

    return `
      <tr id="anomaly-row-${idx}">
        <td>${i + 1}</td>
        <td style="font-size:12px">${data.anomaly_timestamps[i]}</td>
        <td><strong>${data.anomaly_values[i] != null ? data.anomaly_values[i].toFixed(4) : 'N/A'}</strong></td>
        <td>${data.anomaly_scores[i].toFixed(3)}</td>
        <td>
          <span class="badge ${fb.label === 'anomaly' ? 'badge-anomaly' : fb.label === 'normal' ? 'badge-normal' : ''}" id="label-${idx}">
            ${fb.label || 'unlabeled'}
          </span>
        </td>
        <td>
          <input type="text" value="${fb.note || ''}" placeholder="Add note…"
            style="width:120px;padding:4px 8px;font-size:11px"
            onchange="setFeedbackNote(${idx}, this.value)" id="note-${idx}">
        </td>
        <td>
          <div class="feedback-btns">
            <button class="fb-btn ${confirmedClass}" onclick="setFeedback(${idx},'anomaly')" title="Confirm anomaly">✓ Anomaly</button>
            <button class="fb-btn ${rejectedClass}" onclick="setFeedback(${idx},'normal')" title="Mark as normal">✗ Normal</button>
          </div>
        </td>
      </tr>
    `;
  }).join('');
}

// ── Feedback actions ──
function setFeedback(idx, label) {
  if (!state.feedback[idx]) state.feedback[idx] = {};
  state.feedback[idx].label = label;

  // Update UI
  const labelEl = document.getElementById(`label-${idx}`);
  if (labelEl) {
    labelEl.className = `badge ${label === 'anomaly' ? 'badge-anomaly' : 'badge-normal'}`;
    labelEl.textContent = label;
  }

  const row = document.getElementById(`anomaly-row-${idx}`);
  if (row) {
    const btns = row.querySelectorAll('.fb-btn');
    btns.forEach(btn => {
      btn.classList.remove('confirmed', 'rejected');
    });
    if (label === 'anomaly') btns[0].classList.add('confirmed');
    if (label === 'normal') btns[1].classList.add('rejected');
  }

  updateFeedbackSummary();
}

function setFeedbackNote(idx, note) {
  if (!state.feedback[idx]) state.feedback[idx] = {};
  state.feedback[idx].note = note;
}

function confirmAll() {
  if (!state.anomalyResult) return;
  state.anomalyResult.anomaly_indices.forEach(idx => setFeedback(idx, 'anomaly'));
  toast("All anomalies confirmed", "success");
}

function rejectAll() {
  if (!state.anomalyResult) return;
  state.anomalyResult.anomaly_indices.forEach(idx => setFeedback(idx, 'normal'));
  toast("All anomalies rejected", "info");
}

function updateFeedbackSummary() {
  const summary = document.getElementById("feedback-summary");
  const total = Object.keys(state.feedback).length;
  const confirmed = Object.values(state.feedback).filter(f => f.label === 'anomaly').length;
  const rejected = Object.values(state.feedback).filter(f => f.label === 'normal').length;
  const unlabeled = state.anomalyResult ? state.anomalyResult.anomaly_count - total : 0;

  summary.innerHTML = `
    <div style="display:flex;flex-direction:column;gap:6px;font-size:12px">
      <div style="display:flex;justify-content:space-between">
        <span style="color:var(--text-secondary)">Confirmed anomalies</span>
        <span style="color:var(--accent-4);font-weight:700">${confirmed}</span>
      </div>
      <div style="display:flex;justify-content:space-between">
        <span style="color:var(--text-secondary)">Marked normal</span>
        <span style="color:var(--accent-5);font-weight:700">${rejected}</span>
      </div>
      <div style="display:flex;justify-content:space-between">
        <span style="color:var(--text-secondary)">Unlabeled</span>
        <span style="color:var(--text-muted);font-weight:700">${Math.max(0, unlabeled)}</span>
      </div>
    </div>
  `;
}

async function submitAllFeedback() {
  if (!state.anomalyResult || !state.fileKey) {
    toast("No anomalies to submit feedback for", "error");
    return;
  }

  const feedbackPoints = Object.entries(state.feedback).map(([idx, fb]) => ({
    index: parseInt(idx),
    value: state.channels[state.selectedChannel]?.[parseInt(idx)],
    timestamp: state.timestamps[parseInt(idx)],
    label: fb.label || 'unlabeled',
    note: fb.note || '',
  }));

  if (!feedbackPoints.length) {
    toast("No feedback to save", "info");
    return;
  }

  try {
    const result = await api("/api/feedback", {
      file_key: state.fileKey,
      channel: state.selectedChannel,
      feedback_points: feedbackPoints,
    });
    toast(`Saved feedback for ${feedbackPoints.length} points`, "success");
  } catch (e) {
    toast("Failed to save feedback", "error");
  }
}

// ── Upload ──
async function handleUpload(input) {
  const file = input.files[0];
  if (!file) return;
  await uploadFile(file, "uploaded");
  input.value = '';
}

async function handleUploadSidebar(input) {
  const file = input.files[0];
  if (!file) return;
  const experiment = document.getElementById("upload-experiment").value.trim() || "uploaded";
  await uploadFile(file, experiment);
  input.value = '';
}

async function uploadFile(file, experiment) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("experiment", experiment);

  toast(`Uploading ${file.name}…`, "info");
  try {
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    const data = await res.json();
    if (data.error) { toast(data.error, "error"); return; }
    toast(`Uploaded ${file.name} to ${experiment}`, "success");
    loadExperiments();
  } catch (e) {
    toast("Upload failed", "error");
  }
}

// ── Export ──
async function exportAnnotated() {
  if (!state.currentFile || !state.anomalyResult) {
    toast("Run detection first", "error");
    return;
  }

  try {
    const res = await fetch("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        experiment: state.currentExperiment,
        filename: state.currentFile,
        channel: state.selectedChannel,
        anomaly_indices: state.anomalyResult.anomaly_indices,
      }),
    });

    if (!res.ok) { toast("Export failed", "error"); return; }

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `annotated_${state.currentFile}`;
    a.click();
    URL.revokeObjectURL(url);
    toast("Exported annotated CSV", "success");
  } catch (e) {
    toast("Export failed", "error");
  }
}
