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
  demoBase: null,
  demoPhase: 0,
  scenarioTimer: null,
  scenarioRunning: false,
  scenarioPlayed: false,
};

const byId = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[char]);
const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
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
    {id:3, class:"bottle", name:"green water bottle", state:"stationary", seconds_since_seen:4, crop_path:"assets/scene-now.svg"},
    {id:4, class:"cell phone", name:"black cell phone", state:"stationary", seconds_since_seen:7, crop_path:"assets/scene-now.svg"},
    {id:5, class:"book", name:"blue hardcover book", state:"stationary", seconds_since_seen:9, crop_path:"assets/scene-now.svg"},
    {id:1, class:"cup", name:"red ceramic mug", state:"missing", seconds_since_seen:182, crop_path:"assets/mug-pickup.svg"},
    {id:2, class:"laptop", name:"silver laptop", state:"missing", seconds_since_seen:64, crop_path:"assets/laptop-pickup.svg"},
  ];
  const events = [
    {id:10, ts:now-64, type:"person_left", person_id:1, person_name:"person in blue shirt", frame_path:"assets/laptop-pickup.svg"},
    {id:9, ts:now-122, type:"object_missing", object_id:2, object_name:"silver laptop", object_class:"laptop", person_name:"person in blue shirt", direction:"left", frame_path:"assets/laptop-pickup.svg"},
    {id:8, ts:now-124, type:"picked_up", object_id:2, object_name:"silver laptop", object_class:"laptop", person_name:"person in blue shirt", direction:"left", frame_path:"assets/laptop-pickup.svg"},
    {id:7, ts:now-302, type:"object_missing", object_id:1, object_name:"red ceramic mug", object_class:"cup", person_name:"person in blue shirt", direction:"right", frame_path:"assets/mug-pickup.svg"},
    {id:6, ts:now-304, type:"picked_up", object_id:1, object_name:"red ceramic mug", object_class:"cup", person_name:"person in blue shirt", direction:"right", frame_path:"assets/mug-pickup.svg"},
    {id:5, ts:now-480, type:"object_appeared", object_id:5, object_name:"blue hardcover book", object_class:"book", frame_path:"assets/scene-inventory.svg"},
    {id:4, ts:now-480, type:"object_appeared", object_id:4, object_name:"black cell phone", object_class:"cell phone", frame_path:"assets/scene-inventory.svg"},
    {id:3, ts:now-480, type:"object_appeared", object_id:3, object_name:"green water bottle", object_class:"bottle", frame_path:"assets/scene-inventory.svg"},
    {id:2, ts:now-480, type:"object_appeared", object_id:2, object_name:"silver laptop", object_class:"laptop", frame_path:"assets/scene-inventory.svg"},
    {id:1, ts:now-480, type:"object_appeared", object_id:1, object_name:"red ceramic mug", object_class:"cup", frame_path:"assets/scene-inventory.svg"},
  ];
  return {inventory, events};
}

function demoStateForPhase(phase) {
  if (!state.demoBase) state.demoBase = demoDataset();
  const cutoff = [5, 7, 9, 10][phase] ?? 10;
  const inventory = state.demoBase.inventory.map((item) => {
    const missing = (item.id === 1 && phase >= 1) || (item.id === 2 && phase >= 2);
    return {...item, state:missing ? "missing" : "stationary", seconds_since_seen:missing ? (item.id === 1 ? 182 : 64) : item.seconds_since_seen};
  });
  return {inventory, events:state.demoBase.events.filter((event) => event.id <= cutoff)};
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
  byId("mic-button").disabled = busy || state.demoMode;
  byId("question").disabled = busy;
  byId("query-state").textContent = busy ? "PROCESSING" : "READY";
}

const classToken = (value) => String(value || "").toLowerCase().trim().replace(/\s+/g, "-");

function renderSnapshots(items = []) {
  const usable = items.filter((item) => item.frame_path || item.crop_path).slice(0, 2);
  byId("evidence-count").textContent = `${usable.length} EVIDENCE FRAME${usable.length === 1 ? "" : "S"}`;
  byId("answer-snapshots").innerHTML = usable.map((item, index) => {
    const src = item.frame_path || item.crop_path;
    const caption = String(src).split("/").pop().replace(/\.[a-z0-9]+$/i, "").replaceAll("-", " ").toUpperCase();
    return `<figure class="snapshot"><img src="${escapeHtml(mediaUrl(src))}" alt="Evidence frame" onerror="this.hidden=true"><figcaption><span>FRAME ${String(index + 1).padStart(2, "0")}</span><span>${escapeHtml(caption)}</span></figcaption></figure>`;
  }).join("");
}

function renderInventory(items) {
  byId("inventory-count").textContent = `${items.length} OBJECTS`;
  byId("inventory").innerHTML = items.length ? items.map((item) => `
    <button class="item-card${state.selectedObject === item.id ? " selected" : ""}" type="button" data-object-id="${Number(item.id)}" data-object-name="${escapeHtml(item.name || item.class)}" data-class="${escapeHtml(classToken(item.class))}" aria-pressed="${state.selectedObject === item.id}">
      ${item.crop_path ? `<img src="${escapeHtml(mediaUrl(item.crop_path))}" alt="${escapeHtml(item.name || item.class)}" onerror="this.className='item-image-placeholder';this.removeAttribute('src')">` : `<div class="item-image-placeholder"></div>`}
      <div class="item-copy"><strong>${escapeHtml(item.name || item.class)}</strong><span class="item-meta">${escapeHtml(item.class).toUpperCase()} · ${formatAge(item.seconds_since_seen)}</span><span class="state ${escapeHtml(item.state)}">${escapeHtml(item.state).toUpperCase()}</span></div>
    </button>`).join("") : '<div class="empty">NO OBJECTS INDEXED</div>';
}

function renderEvents(events) {
  byId("timeline").innerHTML = events.length ? events.map((event) => {
    const image = event.frame_path || event.crop_path;
    const subject = event.object_name || event.object_class || event.person_name || "Room";
    const detail = [event.type.replaceAll("_", " "), event.person_name, event.direction].filter(Boolean).join(" · ");
    return `<article class="event" data-class="${escapeHtml(classToken(event.object_class || (event.person_name ? "person" : "")))}"><span class="event-dot" aria-hidden="true"></span><time>${formatEventTime(event.ts)}</time><div class="event-copy"><strong>${escapeHtml(subject)}</strong><span>${escapeHtml(detail).toUpperCase()}</span></div>${image ? `<img src="${escapeHtml(mediaUrl(image))}" alt="" onerror="this.hidden=true">` : "<div></div>"}</article>`;
  }).join("") : '<div class="empty">NO EVENTS IN THIS WINDOW</div>';
}

function renderStatus(status) {
  if (status.demo) {
    byId("seg-fps").textContent = "DEMO";
    byId("pose-fps").textContent = "DEMO";
    byId("object-count").textContent = status.objects_tracked || "5";
    byId("vlm-calls").textContent = "--";
    byId("vlm-latency").textContent = "--";
    byId("tts-status").textContent = "VOICE LINK OFFLINE";
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
  byId("model-name").title = byId("model-name").textContent;
  const mode = byId("mode-badge");
  mode.innerHTML = `<i></i>${escapeHtml(status.mode_label || status.perception_mode || "hardware").toUpperCase()}`;
  mode.className = `badge ${status.perception_mode === "hardware" ? "" : "warning"}`;
  mode.title = `Metadata errors: ${status.metadata_errors || 0}; frame errors: ${status.frame_write_errors || 0}; track drops: ${status.track_write_drops || 0}; speech errors: ${status.speech_playback_errors || 0}`;
}

const scenarioSteps = [
  {kicker:"01 // INVENTORY LOCK", message:"5 OBJECTS ACQUIRED", className:"", question:"What's on the table?", answer:"Five objects indexed: red ceramic mug, silver laptop, green water bottle, black cell phone, and blue hardcover book.", snapshots:[{frame_path:"assets/scene-inventory.svg"}]},
  {kicker:"02 // PICKUP DETECTED", message:"RED CERAMIC MUG // EXIT RIGHT", className:"event", question:"Where is the red mug?", answer:"Pickup detected. The person in the blue shirt moved the red ceramic mug right and out of frame.", snapshots:[{frame_path:"assets/mug-pickup.svg"}]},
  {kicker:"03 // CUSTODY EVENT", message:"SILVER LAPTOP // EXIT LEFT", className:"event", question:"Who took my laptop?", answer:"Custody event recorded. The person in the blue shirt picked up the silver laptop and moved left.", snapshots:[{frame_path:"assets/laptop-pickup.svg"}]},
  {kicker:"04 // MEMORY READY", message:"2 EVENTS // 2 EVIDENCE FRAMES", className:"complete", question:"What happened in the last ten minutes?", answer:"The red mug moved right and the silver laptop moved left with the person in the blue shirt. Three objects remain stationary.", snapshots:[{frame_path:"assets/mug-pickup.svg"},{frame_path:"assets/laptop-pickup.svg"}]},
];

function updateScenarioButton() {
  const button = byId("scenario-button");
  button.classList.toggle("running", state.scenarioRunning);
  const iconUse = byId("scenario-icon").querySelector("use");
  if (iconUse) iconUse.setAttribute("href", state.scenarioRunning ? "#i-stop" : state.scenarioPlayed ? "#i-replay" : "#i-play");
  byId("scenario-action").textContent = state.scenarioRunning ? "STOP SCENARIO" : state.scenarioPlayed ? "REPLAY SCENARIO" : "RUN SCENARIO";
}

function stopScenario(completed = false) {
  clearTimeout(state.scenarioTimer);
  state.scenarioTimer = null;
  state.scenarioRunning = false;
  updateScenarioButton();
  byId("query-state").textContent = completed ? "COMPLETE" : "PAUSED";
}

function renderDemoPhase(phase, announce = false) {
  state.demoPhase = Math.max(0, Math.min(phase, scenarioSteps.length - 1));
  state.demo = demoStateForPhase(state.demoPhase);
  const step = scenarioSteps[state.demoPhase];
  const alert = byId("scenario-alert");
  alert.className = `scenario-alert ${step.className}`.trim();
  byId("scenario-kicker").textContent = step.kicker;
  byId("scenario-message").textContent = step.message;
  document.querySelectorAll("#scenario-progress [data-step]").forEach((item) => {
    const itemStep = Number(item.dataset.step);
    item.classList.toggle("done", itemStep < state.demoPhase);
    item.classList.toggle("active", itemStep === state.demoPhase);
  });
  const video = byId("demo-view");
  try { video.currentTime = [0, 6, 11, 16][state.demoPhase]; } catch (_error) { /* Video metadata is still loading. */ }
  video.play().catch(() => {});
  const events = state.selectedObject ? state.demo.events.filter((event) => event.object_id === state.selectedObject) : state.demo.events;
  renderInventory(state.demo.inventory);
  renderEvents(events);
  if (announce) {
    byId("question").value = step.question;
    byId("answer").textContent = step.answer;
    renderSnapshots(step.snapshots);
  }
}

function activateDemo() {
  state.demoMode = true;
  byId("viewer-frame").classList.add("demo-mode");
  byId("perception-title").textContent = "RECORDED PERCEPTION";
  byId("feed-state").textContent = "LOCAL REPLAY";
  byId("offline-badge").textContent = "LOCAL ONLY";
  byId("mic-button").disabled = true;
  byId("mic-button").title = "Voice input requires the connected SiMa runtime";
  renderStatus({demo:true, objects_tracked:5, perception_mode:"demo", mode_label:"simulation"});
  renderDemoPhase(state.demoPhase);
}

function runScenario() {
  if (state.scenarioRunning) {
    stopScenario(false);
    return;
  }
  state.demoMode = true;
  state.scenarioRunning = true;
  state.scenarioPlayed = true;
  state.selectedObject = null;
  byId("timeline-title").textContent = "EVENT LOG";
  byId("all-events").hidden = true;
  byId("viewer-frame").classList.add("demo-mode");
  updateScenarioButton();
  let phase = 0;
  const advance = () => {
    byId("query-state").textContent = `SCENARIO ${phase + 1}/4`;
    renderDemoPhase(phase, true);
    if (phase < scenarioSteps.length - 1) {
      phase += 1;
      state.scenarioTimer = setTimeout(advance, 1800);
    } else {
      state.scenarioTimer = setTimeout(() => stopScenario(true), 1000);
    }
  };
  advance();
}

async function refresh() {
  if (document.hidden || state.scenarioRunning) return;
  if (state.refreshing) {
    state.refreshQueued = true;
    return;
  }
  state.refreshing = true;
  try {
    const eventPath = state.selectedObject ? `/objects/${state.selectedObject}/timeline?limit=500` : "/events?window=900&limit=500";
    const [status, inventory, events] = await Promise.all([json("/status"), json("/inventory"), json(eventPath)]);
    if (state.scenarioRunning) stopScenario(false);
    state.demoMode = false;
    byId("viewer-frame").classList.remove("demo-mode");
    byId("perception-title").textContent = "LIVE PERCEPTION";
    byId("feed-state").textContent = "LIVE LINK";
    byId("query-state").textContent = "READY";
    byId("mic-button").disabled = false;
    byId("mic-button").title = "Ask by voice";
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
    const names = state.demo.inventory.filter((item) => item.state !== "missing").map((item) => item.name);
    return {answer:`${names.length} objects remain in view: ${names.join(", ")}.`, snapshots:[{frame_path:"assets/scene-inventory.svg"}]};
  }
  if (words.includes("laptop")) {
    const missing = state.demo.inventory.find((item) => item.id === 2)?.state === "missing";
    return missing
      ? {answer:"The person in the blue shirt picked up the silver laptop and moved left. It is now marked missing.", snapshots:[{frame_path:"assets/laptop-pickup.svg"}]}
      : {answer:"The silver laptop is still present near the center of the table.", snapshots:[{frame_path:"assets/scene-inventory.svg"}]};
  }
  if (words.includes("mug") || words.includes("cup")) {
    const missing = state.demo.inventory.find((item) => item.id === 1)?.state === "missing";
    return missing
      ? {answer:"The person in the blue shirt picked up the red ceramic mug and moved right. It is now marked missing.", snapshots:[{frame_path:"assets/mug-pickup.svg"}]}
      : {answer:"The red ceramic mug is still present on the left side of the table.", snapshots:[{frame_path:"assets/scene-inventory.svg"}]};
  }
  const custodyEvents = state.demo.events.filter((event) => event.type === "picked_up");
  if (!custodyEvents.length) return {answer:"Five objects were indexed and remain stationary. No custody events have been recorded.", snapshots:[{frame_path:"assets/scene-inventory.svg"}]};
  return {answer:`${custodyEvents.length} custody event${custodyEvents.length === 1 ? " was" : "s were"} recorded. ${state.demo.inventory.filter((item) => item.state !== "missing").length} objects remain stationary.`, snapshots:custodyEvents.slice(0, 2).map((event) => ({frame_path:event.frame_path}))};
}

async function ask(question) {
  if (state.scenarioRunning) stopScenario(false);
  setQuestionBusy(true);
  byId("answer").textContent = "Searching indexed visual memory...";
  renderSnapshots([]);
  try {
    if (state.demoMode) await delay(650);
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

byId("scenario-button").addEventListener("click", runScenario);

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    byId("question").value = button.dataset.question;
    ask(button.dataset.question);
  });
});

byId("summary-button").addEventListener("click", async () => {
  if (state.scenarioRunning) stopScenario(false);
  const button = byId("summary-button");
  button.disabled = true;
  byId("query-state").textContent = "SUMMARIZING";
  byId("answer").textContent = "Building mission summary...";
  try {
    if (state.demoMode) await delay(750);
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

async function runDiagnostics() {
  const rerun = byId("rerun-diagnostics");
  rerun.disabled = true;
  byId("diagnostic-count").textContent = "CHECKING PRODUCT PATHS";
  byId("diagnostic-headline").textContent = "VERIFYING LOCAL AND HARDWARE MODES";
  byId("diagnostic-results").innerHTML = '<div class="empty">RUNNING SYSTEM CHECK</div>';

  const video = byId("demo-view");
  const startTime = video.currentTime;
  await delay(500);
  const videoReady = !video.error && (video.readyState >= 2 || video.currentTime > startTime);
  const objectCount = byId("inventory").querySelectorAll("[data-object-id]").length;
  const eventCount = byId("timeline").querySelectorAll(".event").length;
  const queryReady = Boolean(state.demo && demoAnswer("Where is the red mug?").answer);

  let apiStatus = null;
  try {
    apiStatus = await json("/status", {signal:timeoutSignal(2200)});
  } catch (_error) {
    apiStatus = null;
  }
  const apiOnline = Boolean(apiStatus);
  const hardwareLive = apiOnline
    && apiStatus.perception_mode === "hardware"
    && Number(apiStatus.seg_fps) > 0
    && Number(apiStatus.pose_fps) > 0;

  const checks = [
    {name:"BROWSER INTERFACE", detail:"Core controls and status panels are mounted.", ok:Boolean(byId("ask-form") && byId("inventory") && byId("timeline")), required:true},
    {name:"RECORDED FEED", detail:videoReady ? "The bundled local video is decoding and advancing." : "The bundled video did not advance.", ok:videoReady, required:true},
    {name:"MEMORY WORKFLOW", detail:`${objectCount} objects // ${eventCount} events // query ${queryReady ? "ready" : "failed"}`, ok:objectCount > 0 && eventCount > 0 && queryReady, required:true},
    {name:"SIMA API", detail:apiOnline ? `${state.api} responded.` : `${state.api} is unreachable from this browser.`, ok:apiOnline, offline:!apiOnline},
    {name:"LIVE PERCEPTION", detail:hardwareLive ? `${Number(apiStatus.seg_fps).toFixed(1)} seg FPS // ${Number(apiStatus.pose_fps).toFixed(1)} pose FPS` : "Requires a connected DevKit in hardware mode with nonzero FPS.", ok:hardwareLive, offline:!hardwareLive},
  ];
  const localChecks = checks.filter((check) => check.required);
  const localPassed = localChecks.filter((check) => check.ok).length;
  byId("diagnostic-results").innerHTML = checks.map((check) => {
    const status = check.ok ? "PASS" : check.offline ? "OFFLINE" : "FAIL";
    const statusClass = check.ok ? "pass" : check.offline ? "offline" : "fail";
    return `<div class="diagnostic-row"><div><strong>${escapeHtml(check.name)}</strong><span>${escapeHtml(check.detail)}</span></div><p class="diagnostic-result ${statusClass}">${status}</p></div>`;
  }).join("");
  byId("diagnostic-count").textContent = `${localPassed}/${localChecks.length} LOCAL CHECKS PASS`;
  byId("diagnostic-headline").textContent = localPassed !== localChecks.length
    ? "LOCAL DEMO CHECK FAILED"
    : hardwareLive ? "LIVE PERCEPTION ONLINE" : "LOCAL DEMO READY // HARDWARE OFFLINE";
  rerun.disabled = false;
}

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

const diagnosticsDialog = byId("diagnostics-dialog");
byId("self-test-button").addEventListener("click", () => {
  diagnosticsDialog.showModal();
  runDiagnostics();
});
byId("rerun-diagnostics").addEventListener("click", runDiagnostics);

function updateClock() {
  byId("feed-clock").textContent = `${new Date().toISOString().slice(11, 19)} UTC`;
}

byId("live-view").src = state.insight;
updateClock();
setInterval(updateClock, 1000);
activateDemo();
updateScenarioButton();
refresh();
setTimeout(() => {
  if (state.demoMode && !state.scenarioPlayed) runScenario();
}, 900);
setInterval(refresh, 15000);
document.addEventListener("visibilitychange", refresh);
