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
  demoMode: false,
  demo: null,
};

const byId = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[char]);
const mediaUrl = (path) => {
  if (!path) return "";
  if (String(path).startsWith("assets/")) return String(path);
  return `${state.api}/media/${String(path).replace(/^\/?media\//, "")}`;
};

async function json(path, options) {
  const response = await fetch(`${state.api}${path}`, {
    ...options,
    signal: options?.signal || timeoutSignal(3500),
  });
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json();
}

function demoDataset() {
  const now = Date.now() / 1000;
  const inventory = [
    {id:3, class:"bottle", name:"green water bottle", state:"stationary", seconds_since_seen:4, crop_path:"assets/scene-now.jpg"},
    {id:4, class:"cell phone", name:"black cell phone", state:"stationary", seconds_since_seen:7, crop_path:"assets/scene-now.jpg"},
    {id:5, class:"book", name:"blue hardcover book", state:"stationary", seconds_since_seen:9, crop_path:"assets/scene-now.jpg"},
    {id:1, class:"cup", name:"red ceramic mug", state:"missing", seconds_since_seen:182, crop_path:"assets/mug-pickup.jpg"},
    {id:2, class:"laptop", name:"silver laptop", state:"missing", seconds_since_seen:64, crop_path:"assets/laptop-pickup.jpg"},
  ];
  const events = [
    {id:10, ts:now-64, type:"person_left", person_id:1, person_name:"person in blue shirt", frame_path:"assets/laptop-pickup.jpg"},
    {id:9, ts:now-122, type:"object_missing", object_id:2, object_name:"silver laptop", object_class:"laptop", person_name:"person in blue shirt", direction:"left", frame_path:"assets/laptop-pickup.jpg"},
    {id:8, ts:now-124, type:"picked_up", object_id:2, object_name:"silver laptop", object_class:"laptop", person_name:"person in blue shirt", direction:"left", frame_path:"assets/laptop-pickup.jpg"},
    {id:7, ts:now-302, type:"object_missing", object_id:1, object_name:"red ceramic mug", object_class:"cup", person_name:"person in blue shirt", direction:"right", frame_path:"assets/mug-pickup.jpg"},
    {id:6, ts:now-304, type:"picked_up", object_id:1, object_name:"red ceramic mug", object_class:"cup", person_name:"person in blue shirt", direction:"right", frame_path:"assets/mug-pickup.jpg"},
    {id:5, ts:now-480, type:"object_appeared", object_id:5, object_name:"blue hardcover book", object_class:"book", frame_path:"assets/scene-inventory.jpg"},
    {id:4, ts:now-480, type:"object_appeared", object_id:4, object_name:"black cell phone", object_class:"cell phone", frame_path:"assets/scene-inventory.jpg"},
    {id:3, ts:now-480, type:"object_appeared", object_id:3, object_name:"green water bottle", object_class:"bottle", frame_path:"assets/scene-inventory.jpg"},
  ];
  return {inventory, events};
}

function formatAge(seconds) {
  const value = Math.max(0, Math.round(Number(seconds) || 0));
  if (value < 60) return `${value} SEC AGO`;
  if (value < 3600) return `${Math.floor(value / 60)} MIN AGO`;
  return `${Math.floor(value / 3600)} HR AGO`;
}

function formatEventTime(timestamp) {
  return new Date(timestamp * 1000).toLocaleTimeString([], {hour:"2-digit", minute:"2-digit", second:"2-digit", hour12:false});
}

function setQuestionBusy(busy) {
  byId("ask-button").disabled = busy;
  byId("mic-button").disabled = busy;
  byId("question").disabled = busy;
  byId("query-state").textContent = busy ? "PROCESSING" : "READY";
}

function renderSnapshots(items = []) {
  const usable = items.filter((item) => item.frame_path || item.crop_path).slice(0, 2);
  byId("evidence-count").textContent = `${usable.length} EVIDENCE FRAME${usable.length === 1 ? "" : "S"}`;
  byId("answer-snapshots").innerHTML = usable.map((item) => {
    const src = item.frame_path || item.crop_path;
    return `<img src="${escapeHtml(mediaUrl(src))}" alt="Evidence frame" onerror="this.hidden=true">`;
  }).join("");
}

function renderInventory(items) {
  byId("inventory-count").textContent = `${items.length} OBJECTS`;
  byId("inventory").innerHTML = items.length ? items.map((item) => `
    <button class="item-card${state.selectedObject === item.id ? " selected" : ""}" type="button" data-object-id="${Number(item.id)}" data-object-name="${escapeHtml(item.name || item.class)}" aria-pressed="${state.selectedObject === item.id}">
      ${item.crop_path ? `<img src="${escapeHtml(mediaUrl(item.crop_path))}" alt="${escapeHtml(item.name || item.class)}" onerror="this.className='item-image-placeholder';this.removeAttribute('src')">` : `<div class="item-image-placeholder"></div>`}
      <div class="item-copy"><strong>${escapeHtml(item.name || item.class)}</strong><span class="item-meta">${escapeHtml(item.class).toUpperCase()} // ${formatAge(item.seconds_since_seen)}</span><span class="state ${escapeHtml(item.state)}">${escapeHtml(item.state).toUpperCase()}</span></div>
    </button>`).join("") : '<div class="empty">NO OBJECTS INDEXED</div>';
}

function renderEvents(events) {
  byId("timeline").innerHTML = events.length ? events.map((event) => {
    const image = event.frame_path || event.crop_path;
    const subject = event.object_name || event.object_class || event.person_name || "Room";
    const detail = [event.type.replaceAll("_", " "), event.person_name, event.direction].filter(Boolean).join(" // ");
    return `<article class="event"><time>${formatEventTime(event.ts)}</time><div><strong>${escapeHtml(subject)}</strong><span>${escapeHtml(detail).toUpperCase()}</span></div>${image ? `<img src="${escapeHtml(mediaUrl(image))}" alt="" onerror="this.hidden=true">` : "<div></div>"}</article>`;
  }).join("") : '<div class="empty">NO EVENTS IN THIS WINDOW</div>';
}

function renderStatus(status) {
  if (status.demo) {
    byId("seg-fps").textContent = "DEMO";
    byId("pose-fps").textContent = "DEMO";
    byId("object-count").textContent = "5";
    byId("vlm-calls").textContent = "--";
    byId("vlm-latency").textContent = "--";
    byId("tts-status").textContent = "DEMO AUDIO";
    byId("model-name").textContent = "SIMA HARDWARE PROFILE // OFFLINE";
  } else {
  byId("seg-fps").textContent = `${Number(status.seg_fps || 0).toFixed(1)} FPS`;
  byId("pose-fps").textContent = `${Number(status.pose_fps || 0).toFixed(1)} FPS`;
  byId("object-count").textContent = status.objects_tracked || 0;
  byId("vlm-calls").textContent = status.vlm_calls_total ?? status.vlm_calls ?? 0;
  byId("vlm-calls").title = `${status.vlm_calls || 0} answers; ${status.naming_vlm_calls || 0} naming calls; ${status.naming_vlm_failures || 0} naming failures`;
  byId("vlm-latency").textContent = `${status.vlm_ms_p50 || 0} MS`;
  byId("tts-status").textContent = status.tts === "piper-ready" ? "PIPER READY" : status.tts === "piper-warming" ? "WARMING" : "UNAVAILABLE";
  byId("model-name").textContent = status.resident_model || "FALLBACK RULES";
  }
  const mode = byId("mode-badge");
  mode.innerHTML = `<i></i>${escapeHtml(status.mode_label || status.perception_mode || "hardware").toUpperCase()}`;
  mode.className = `badge ${status.perception_mode === "hardware" ? "" : "warning"}`;
  mode.title = `Metadata errors: ${status.metadata_errors || 0}; frame errors: ${status.frame_write_errors || 0}; track drops: ${status.track_write_drops || 0}; speech errors: ${status.speech_playback_errors || 0}`;
}

function activateDemo() {
  state.demoMode = true;
  state.demo = demoDataset();
  byId("viewer-frame").classList.add("demo-mode");
  byId("feed-state").textContent = "DEMO FEED";
  const mode = byId("mode-badge");
  byId("offline-badge").textContent = "LOCAL ONLY";
  renderStatus({demo:true, perception_mode:"demo", mode_label:"demo ready"});
  const events = state.selectedObject ? state.demo.events.filter((event) => event.object_id === state.selectedObject) : state.demo.events;
  renderInventory(state.demo.inventory);
  renderEvents(events);
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
    state.demoMode = false;
    byId("viewer-frame").classList.remove("demo-mode");
    byId("feed-state").textContent = "LIVE LINK";
    renderStatus(status);
    renderInventory(inventory);
    renderEvents(events);
  } catch (_error) {
    activateDemo();
  } finally {
    state.refreshing = false;
    if (state.refreshQueued) {
      state.refreshQueued = false;
      setTimeout(refresh, 0);
    }
  }
}

function demoAnswer(question) {
  const words = question.toLowerCase();
  if (words.includes("table") || words.includes("inventory")) {
    return {answer:"Three objects remain in view: the green water bottle, black cell phone, and blue hardcover book.", snapshots:[{frame_path:"assets/scene-inventory.jpg"}]};
  }
  if (words.includes("laptop")) {
    return {answer:"The person in the blue shirt picked up the silver laptop and moved left. It is now marked missing.", snapshots:[{frame_path:"assets/laptop-pickup.jpg"}]};
  }
  if (words.includes("mug") || words.includes("cup")) {
    return {answer:"The person in the blue shirt picked up the red ceramic mug and moved right. It is now marked missing.", snapshots:[{frame_path:"assets/mug-pickup.jpg"}]};
  }
  return {answer:"Two custody events were recorded: the red mug moved right and the silver laptop moved left. Three objects remain stationary.", snapshots:[{frame_path:"assets/mug-pickup.jpg"},{frame_path:"assets/laptop-pickup.jpg"}]};
}

async function ask(question) {
  setQuestionBusy(true);
  byId("answer").textContent = "Searching indexed visual memory...";
  renderSnapshots([]);
  try {
    const result = state.demoMode ? demoAnswer(question) : await json("/ask", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({question})});
    byId("answer").textContent = result.answer;
    renderSnapshots(result.snapshots);
  } catch (_error) {
    const result = demoAnswer(question);
    activateDemo();
    byId("answer").textContent = result.answer;
    renderSnapshots(result.snapshots);
  } finally {
    setQuestionBusy(false);
  }
}

byId("inventory").addEventListener("click", (event) => {
  const card = event.target.closest("[data-object-id]");
  if (!card) return;
  state.selectedObject = Number(card.dataset.objectId);
  byId("timeline-title").textContent = card.dataset.objectName.toUpperCase();
  byId("all-events").hidden = false;
  if (state.demoMode) activateDemo(); else refresh();
});

byId("all-events").addEventListener("click", () => {
  state.selectedObject = null;
  byId("timeline-title").textContent = "EVENT LOG";
  byId("all-events").hidden = true;
  if (state.demoMode) activateDemo(); else refresh();
});

byId("ask-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const question = byId("question").value.trim() || "Where is the red mug?";
  byId("question").value = question;
  ask(question);
});

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    byId("question").value = button.dataset.question;
    ask(button.dataset.question);
  });
});

byId("summary-button").addEventListener("click", async () => {
  const button = byId("summary-button");
  button.disabled = true;
  byId("query-state").textContent = "SUMMARIZING";
  byId("answer").textContent = "Building mission summary...";
  try {
    const result = state.demoMode ? demoAnswer("summary") : await json("/summary", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({window:600})});
    byId("answer").textContent = result.summary || result.answer;
    renderSnapshots(result.snapshots);
  } catch (_error) {
    const result = demoAnswer("summary");
    activateDemo();
    byId("answer").textContent = result.answer;
    renderSnapshots(result.snapshots);
  } finally {
    button.disabled = false;
    byId("query-state").textContent = "READY";
  }
});

byId("mic-button").addEventListener("click", async () => {
  const button = byId("mic-button");
  if (state.recorder?.state === "recording") { state.recorder.stop(); return; }
  try {
    if (state.demoMode) throw new Error("voice link requires the connected SiMa runtime");
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") throw new Error("audio recording is not supported by this browser");
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
      byId("answer").textContent = "Transcribing local audio...";
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
  } catch (error) { byId("answer").textContent = error.message; }
});

const dialog = byId("settings-dialog");
byId("settings-button").addEventListener("click", () => {
  byId("api-url").value = state.api;
  byId("insight-url").value = state.insight;
  dialog.showModal();
});
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

function updateClock() {
  byId("feed-clock").textContent = `${new Date().toISOString().slice(11, 19)} UTC`;
}

byId("live-view").src = state.insight;
updateClock();
setInterval(updateClock, 1000);
activateDemo();
refresh();
setInterval(refresh, 5000);
document.addEventListener("visibilitychange", refresh);
