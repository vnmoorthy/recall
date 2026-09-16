function storedHttpUrl(key, fallback) {
  try {
    const value = new URL(localStorage.getItem(key) || fallback);
    return /^https?:$/.test(value.protocol) ? value.href.replace(/\/$/, "") : fallback;
  } catch (_error) {
    return fallback;
  }
}

function timeoutSignal(milliseconds) {
  if (typeof AbortSignal !== "undefined" && typeof AbortSignal.timeout === "function") return AbortSignal.timeout(milliseconds);
  const controller = new AbortController();
  setTimeout(() => controller.abort(), milliseconds);
  return controller.signal;
}

const state = {
  api: storedHttpUrl("recall-api", "http://10.42.0.232:8090"),
  insight: storedHttpUrl("recall-insight", "https://10.42.0.1:8081/static/viewer.html?mode=light&src=0&max_channels=4"),
  recorder: null,
  chunks: [],
  refreshing: false,
  refreshQueued: false,
  selectedObject: null,
};

const byId = (id) => document.getElementById(id);
const mediaUrl = (path) => path ? `${state.api}/media/${String(path).replace(/^\/?media\//, "")}` : "";
const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[char]);

async function json(path, options) {
  const response = await fetch(`${state.api}${path}`, {
    ...options,
    signal: options?.signal || timeoutSignal(8000),
  });
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json();
}

function formatAge(seconds) {
  const value = Math.max(0, Math.round(Number(seconds) || 0));
  if (value < 60) return `${value}s ago`;
  if (value < 3600) return `${Math.floor(value / 60)}m ago`;
  return `${Math.floor(value / 3600)}h ago`;
}

function setQuestionBusy(busy) {
  byId("ask-button").disabled = busy;
  byId("mic-button").disabled = busy;
  byId("question").disabled = busy;
}

function renderSnapshots(items = []) {
  byId("answer-snapshots").innerHTML = items.map((item) => {
    const src = item.frame_path || item.crop_path;
    return src ? `<img src="${escapeHtml(mediaUrl(src))}" alt="Event snapshot">` : "";
  }).join("");
}

function renderInventory(items) {
  byId("inventory-count").textContent = `${items.length} items`;
  byId("inventory").innerHTML = items.length ? items.map((item) => `
    <button class="item-card${state.selectedObject === item.id ? " selected" : ""}" type="button" data-object-id="${Number(item.id)}" data-object-name="${escapeHtml(item.name || item.class)}" aria-pressed="${state.selectedObject === item.id}">
      ${item.crop_path ? `<img src="${escapeHtml(mediaUrl(item.crop_path))}" alt="${escapeHtml(item.name || item.class)}">` : `<div></div>`}
      <div class="item-copy"><strong>${escapeHtml(item.name || item.class)}</strong><span>${escapeHtml(item.class)} / ${formatAge(item.seconds_since_seen)}</span><span class="state ${escapeHtml(item.state)}">${escapeHtml(item.state)}</span></div>
    </button>`).join("") : '<div class="empty">No objects indexed</div>';
}

function renderEvents(events) {
  byId("timeline").innerHTML = events.length ? events.map((event) => {
    const image = event.frame_path || event.crop_path;
    const subject = event.object_name || event.object_class || event.person_name || "Room";
    const detail = [event.person_name, event.direction].filter(Boolean).join(" / ");
    return `<article class="event"><time>${new Date(event.ts * 1000).toLocaleTimeString([], {hour:"2-digit",minute:"2-digit",second:"2-digit"})}</time><div><strong>${escapeHtml(subject)}</strong><span>${escapeHtml(event.type.replaceAll("_", " "))}${detail ? ` / ${escapeHtml(detail)}` : ""}</span></div>${image ? `<img src="${escapeHtml(mediaUrl(image))}" alt="">` : "<div></div>"}</article>`;
  }).join("") : '<div class="empty">No events in this window</div>';
}

async function refresh() {
  if (document.hidden) return;
  if (state.refreshing) {
    state.refreshQueued = true;
    return;
  }
  state.refreshing = true;
  try {
    const eventPath = state.selectedObject ? `/objects/${state.selectedObject}/timeline?limit=500` : "/events?window=900&limit=500";
    const [status, inventory, events] = await Promise.all([json("/status"), json("/inventory"), json(eventPath)]);
    byId("seg-fps").textContent = `${status.seg_fps.toFixed(1)} fps`;
    byId("pose-fps").textContent = `${status.pose_fps.toFixed(1)} fps`;
    byId("object-count").textContent = status.objects_tracked;
    byId("vlm-calls").textContent = status.vlm_calls_total ?? status.vlm_calls;
    byId("vlm-calls").title = `${status.vlm_calls || 0} answers; ${status.naming_vlm_calls || 0} naming calls; ${status.naming_vlm_failures || 0} naming failures`;
    byId("vlm-latency").textContent = `${status.vlm_ms_p50} ms`;
    byId("tts-status").textContent = status.tts === "piper-ready" ? "PIPER READY" : status.tts === "piper-warming" ? "WARMING" : "UNAVAILABLE";
    byId("model-name").textContent = status.resident_model;
    byId("model-name").title = status.genai_healthy ? "GenAI server healthy" : "GenAI server unavailable; evidence rules remain active";
    const mode = byId("mode-badge");
    mode.textContent = status.perception_healthy ? status.perception_mode.toUpperCase() : "PERCEPTION ERROR";
    mode.className = `badge ${!status.perception_healthy ? "error" : status.perception_mode === "hardware" ? "" : "warning"}`;
    mode.title = `Metadata errors: ${status.metadata_errors || 0}; frame write errors: ${status.frame_write_errors || 0}; track drops: ${status.track_write_drops || 0}; speech errors: ${status.speech_playback_errors || 0}`;
    byId("offline-badge").textContent = status.offline ? "OFFLINE" : "NETWORKED";
    renderInventory(inventory);
    renderEvents(events);
  } catch (error) {
    const mode = byId("mode-badge");
    mode.textContent = "DISCONNECTED";
    mode.className = "badge error";
  } finally {
    state.refreshing = false;
    if (state.refreshQueued) {
      state.refreshQueued = false;
      setTimeout(refresh, 0);
    }
  }
}

byId("inventory").addEventListener("click", (event) => {
  const card = event.target.closest("[data-object-id]");
  if (!card) return;
  state.selectedObject = Number(card.dataset.objectId);
  byId("timeline-title").textContent = card.dataset.objectName;
  byId("all-events").hidden = false;
  refresh();
});

byId("all-events").addEventListener("click", () => {
  state.selectedObject = null;
  byId("timeline-title").textContent = "Timeline";
  byId("all-events").hidden = true;
  refresh();
});

async function ask(question) {
  setQuestionBusy(true);
  byId("answer").textContent = "Searching visual memory...";
  renderSnapshots([]);
  try {
    const result = await json("/ask", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({question})});
    byId("answer").textContent = result.answer;
    renderSnapshots(result.snapshots);
  } catch (error) {
    byId("answer").textContent = `Recall API unavailable: ${error.message}`;
  } finally {
    setQuestionBusy(false);
  }
}

byId("ask-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const question = byId("question").value.trim();
  if (question) ask(question);
});

byId("summary-button").addEventListener("click", async () => {
  const button = byId("summary-button");
  button.disabled = true;
  byId("answer").textContent = "Summarizing...";
  try {
    const result = await json("/summary", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({window:600})});
    byId("answer").textContent = result.summary;
    renderSnapshots(result.snapshots);
  } catch (error) { byId("answer").textContent = error.message; }
  finally { button.disabled = false; }
});

byId("mic-button").addEventListener("click", async () => {
  const button = byId("mic-button");
  if (state.recorder?.state === "recording") { state.recorder.stop(); return; }
  try {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      throw new Error("audio recording is not supported by this browser");
    }
    const stream = await navigator.mediaDevices.getUserMedia({audio:true});
    state.chunks = [];
    state.recorder = new MediaRecorder(stream);
    state.recorder.ondataavailable = (event) => state.chunks.push(event.data);
    state.recorder.onstop = async () => {
      button.disabled = true;
      button.classList.remove("recording");
      button.setAttribute("aria-pressed", "false");
      stream.getTracks().forEach((track) => track.stop());
      const form = new FormData();
      form.append("file", new Blob(state.chunks, {type:state.recorder.mimeType}), "question.webm");
      byId("answer").textContent = "Transcribing...";
      try {
        const response = await fetch(`${state.api}/ask_audio`, {method:"POST", body:form, signal:timeoutSignal(45000)});
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || `${response.status} transcription failed`);
        byId("question").value = result.question || "";
        byId("answer").textContent = result.answer || result.detail;
        renderSnapshots(result.snapshots);
      } catch (error) { byId("answer").textContent = error.message; }
      finally { button.disabled = false; }
    };
    state.recorder.onerror = (event) => {
      button.classList.remove("recording");
      button.setAttribute("aria-pressed", "false");
      stream.getTracks().forEach((track) => track.stop());
      byId("answer").textContent = `Recording failed: ${event.error?.message || "unknown error"}`;
    };
    state.recorder.start();
    button.classList.add("recording");
    button.setAttribute("aria-pressed", "true");
  } catch (error) { byId("answer").textContent = `Microphone unavailable: ${error.message}`; }
});

const dialog = byId("settings-dialog");
byId("settings-button").addEventListener("click", () => { byId("api-url").value = state.api; byId("insight-url").value = state.insight; dialog.showModal(); });
byId("save-settings").addEventListener("click", (event) => {
  try {
    const api = new URL(byId("api-url").value);
    const insight = new URL(byId("insight-url").value);
    if (!/^https?:$/.test(api.protocol) || !/^https?:$/.test(insight.protocol)) throw new Error();
    state.api = api.href.replace(/\/$/, "");
    state.insight = insight.href;
  } catch (_error) {
    event.preventDefault();
    byId("answer").textContent = "Connection URLs must use HTTP or HTTPS.";
    return;
  }
  localStorage.setItem("recall-api", state.api);
  localStorage.setItem("recall-insight", state.insight);
  byId("live-view").src = state.insight;
  setTimeout(refresh, 0);
});

byId("live-view").src = state.insight;
refresh();
setInterval(refresh, 3000);
document.addEventListener("visibilitychange", refresh);
