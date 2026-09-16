const state = {
  api: localStorage.getItem("recall-api") || "http://10.42.0.232:8090",
  insight: localStorage.getItem("recall-insight") || "https://10.42.0.1:8081/static/viewer.html?mode=light&src=0&max_channels=4",
  recorder: null,
  chunks: [],
};

const byId = (id) => document.getElementById(id);
const mediaUrl = (path) => path ? `${state.api}/media/${String(path).replace(/^\/?media\//, "")}` : "";
const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[char]);

async function json(path, options) {
  const response = await fetch(`${state.api}${path}`, options);
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json();
}

function renderSnapshots(items = []) {
  byId("answer-snapshots").innerHTML = items.map((item) => {
    const src = item.frame_path || item.crop_path;
    return src ? `<img src="${mediaUrl(src)}" alt="Event snapshot">` : "";
  }).join("");
}

function renderInventory(items) {
  byId("inventory-count").textContent = `${items.length} items`;
  byId("inventory").innerHTML = items.length ? items.map((item) => `
    <article class="item-card">
      ${item.crop_path ? `<img src="${mediaUrl(item.crop_path)}" alt="${escapeHtml(item.name || item.class)}">` : `<div></div>`}
      <div class="item-copy"><strong>${escapeHtml(item.name || item.class)}</strong><span>${escapeHtml(item.class)}</span><span class="state ${escapeHtml(item.state)}">${escapeHtml(item.state)}</span></div>
    </article>`).join("") : '<div class="empty">No objects indexed</div>';
}

function renderEvents(events) {
  byId("timeline").innerHTML = events.length ? events.map((event) => {
    const image = event.frame_path || event.crop_path;
    const subject = event.object_name || event.object_class || event.person_name || "Room";
    const detail = [event.person_name, event.direction].filter(Boolean).join(" / ");
    return `<article class="event"><time>${new Date(event.ts * 1000).toLocaleTimeString([], {hour:"2-digit",minute:"2-digit",second:"2-digit"})}</time><div><strong>${escapeHtml(subject)}</strong><span>${escapeHtml(event.type.replaceAll("_", " "))}${detail ? ` / ${escapeHtml(detail)}` : ""}</span></div>${image ? `<img src="${mediaUrl(image)}" alt="">` : "<div></div>"}</article>`;
  }).join("") : '<div class="empty">No events in this window</div>';
}

async function refresh() {
  try {
    const [status, inventory, events] = await Promise.all([json("/status"), json("/inventory"), json("/events?window=900")]);
    byId("seg-fps").textContent = `${status.seg_fps.toFixed(1)} fps`;
    byId("pose-fps").textContent = `${status.pose_fps.toFixed(1)} fps`;
    byId("object-count").textContent = status.objects_tracked;
    byId("vlm-calls").textContent = status.vlm_calls;
    byId("vlm-latency").textContent = `${status.vlm_ms_p50} ms`;
    byId("model-name").textContent = status.resident_model;
    const mode = byId("mode-badge");
    mode.textContent = status.perception_mode.toUpperCase();
    mode.className = `badge ${status.perception_mode === "hardware" ? "" : "warning"}`;
    byId("offline-badge").textContent = status.offline ? "OFFLINE" : "NETWORKED";
    renderInventory(inventory);
    renderEvents(events);
  } catch (error) {
    const mode = byId("mode-badge");
    mode.textContent = "DISCONNECTED";
    mode.className = "badge error";
  }
}

async function ask(question) {
  byId("answer").textContent = "Searching visual memory...";
  renderSnapshots([]);
  try {
    const result = await json("/ask", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({question})});
    byId("answer").textContent = result.answer;
    renderSnapshots(result.snapshots);
  } catch (error) {
    byId("answer").textContent = `Recall API unavailable: ${error.message}`;
  }
}

byId("ask-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const question = byId("question").value.trim();
  if (question) ask(question);
});

byId("summary-button").addEventListener("click", async () => {
  byId("answer").textContent = "Summarizing...";
  try {
    const result = await json("/summary", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({window:600})});
    byId("answer").textContent = result.summary;
    renderSnapshots(result.snapshots);
  } catch (error) { byId("answer").textContent = error.message; }
});

byId("mic-button").addEventListener("click", async () => {
  const button = byId("mic-button");
  if (state.recorder?.state === "recording") { state.recorder.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio:true});
    state.chunks = [];
    state.recorder = new MediaRecorder(stream);
    state.recorder.ondataavailable = (event) => state.chunks.push(event.data);
    state.recorder.onstop = async () => {
      button.classList.remove("recording");
      stream.getTracks().forEach((track) => track.stop());
      const form = new FormData();
      form.append("file", new Blob(state.chunks, {type:state.recorder.mimeType}), "question.webm");
      byId("answer").textContent = "Transcribing...";
      try {
        const response = await fetch(`${state.api}/ask_audio`, {method:"POST", body:form});
        const result = await response.json();
        byId("question").value = result.question || "";
        byId("answer").textContent = result.answer || result.detail;
        renderSnapshots(result.snapshots);
      } catch (error) { byId("answer").textContent = error.message; }
    };
    state.recorder.start();
    button.classList.add("recording");
  } catch (error) { byId("answer").textContent = `Microphone unavailable: ${error.message}`; }
});

const dialog = byId("settings-dialog");
byId("settings-button").addEventListener("click", () => { byId("api-url").value = state.api; byId("insight-url").value = state.insight; dialog.showModal(); });
byId("save-settings").addEventListener("click", () => {
  state.api = byId("api-url").value.replace(/\/$/, "");
  state.insight = byId("insight-url").value;
  localStorage.setItem("recall-api", state.api);
  localStorage.setItem("recall-insight", state.insight);
  byId("live-view").src = state.insight;
  setTimeout(refresh, 0);
});

byId("live-view").src = state.insight;
refresh();
setInterval(refresh, 3000);
