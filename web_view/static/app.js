const EDGES = [
  [0, 1], [0, 2], [0, 3],
  [1, 4], [4, 7], [7, 10],
  [2, 5], [5, 8], [8, 11],
  [3, 6], [6, 9], [9, 12], [12, 15],
  [9, 13], [13, 16], [16, 18], [18, 20],
  [9, 14], [14, 17], [17, 19], [19, 21],
];

class NullViewport {
  setMotion(_motion) {}
  tick(_dt) {}
}

function buildSkeletonViewportClass(THREE, OrbitControls) {
  return class SkeletonViewport {
    constructor(canvasId, jointColor = 0x0f5f94, boneColor = 0x264653) {
      this.canvas = document.getElementById(canvasId);
      this.scene = new THREE.Scene();
      this.scene.background = new THREE.Color(0xf8faf7);
      this.camera = new THREE.PerspectiveCamera(40, 1.0, 0.01, 1000);
      {
        const baseOffset = new THREE.Vector3(2.4, 1.4, 2.8).multiplyScalar(1.0 / 1.5);
        baseOffset.applyAxisAngle(new THREE.Vector3(0, 1, 0), -Math.PI / 8.0);
        this.camera.position.copy(baseOffset);
      }

      this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true });
      this.renderer.setPixelRatio(window.devicePixelRatio);

      this.controls = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.enableZoom = false;
      this.controls.target.set(0, 0.9, 0);

      const hemi = new THREE.HemisphereLight(0xffffff, 0x98a1aa, 1.2);
      this.scene.add(hemi);
      this.scene.add(new THREE.GridHelper(6, 18, 0xcad1ca, 0xe7ebe7));

      this.joints = [];
      const sphereGeo = new THREE.SphereGeometry(0.025, 12, 12);
      const sphereMat = new THREE.MeshStandardMaterial({ color: jointColor });
      for (let i = 0; i < 22; i += 1) {
        const s = new THREE.Mesh(sphereGeo, sphereMat);
        this.scene.add(s);
        this.joints.push(s);
      }

      this.bonePos = new Float32Array(EDGES.length * 2 * 3);
      const boneGeo = new THREE.BufferGeometry();
      boneGeo.setAttribute("position", new THREE.BufferAttribute(this.bonePos, 3));
      this.bones = new THREE.LineSegments(
        boneGeo,
        new THREE.LineBasicMaterial({ color: boneColor, linewidth: 1 })
      );
      this.scene.add(this.bones);

      this.motion = null;
      this.frame = 0;
      this.accum = 0.0;
      this.playing = true;
      this._resize();
      window.addEventListener("resize", () => this._resize());
    }

    _resize() {
      const w = this.canvas.clientWidth || 640;
      const h = this.canvas.clientHeight || 360;
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h, false);
    }

    setMotion(motion) {
      if (!Array.isArray(motion) || motion.length === 0) {
        this.motion = null;
        return;
      }
      this.motion = motion;
      this.frame = 0;
      this.accum = 0.0;
      this._resize();
      this._fitCamera();
      this._renderFrame();
    }

    _fitCamera() {
      if (!this.motion) return;
      const f0 = this.motion[0];
      if (!f0 || !f0.length) return;
      const box = new THREE.Box3();
      for (const p of f0) {
        box.expandByPoint(new THREE.Vector3(p[0], p[1], p[2]));
      }
      const center = box.getCenter(new THREE.Vector3());
      this.controls.target.copy(center);
      const size = box.getSize(new THREE.Vector3()).length();
      const baseDist = Math.max(2.0, size * 2.4);
      const dist = baseDist / 1.5;
      const viewOffset = new THREE.Vector3(dist * 0.8, dist * 0.55, dist * 0.9);
      viewOffset.applyAxisAngle(new THREE.Vector3(0, 1, 0), -Math.PI / 8.0);
      this.camera.position.copy(center.clone().add(viewOffset));
      this.camera.lookAt(center);
      this.controls.update();
    }

    _renderFrame() {
      if (!this.motion) return;
      const f = this.motion[this.frame];
      if (!f) return;
      for (let i = 0; i < this.joints.length; i += 1) {
        const p = f[i];
        if (!p) continue;
        this.joints[i].position.set(p[0], p[1], p[2]);
      }
      for (let e = 0; e < EDGES.length; e += 1) {
        const [a, b] = EDGES[e];
        const pa = f[a];
        const pb = f[b];
        const base = e * 6;
        this.bonePos[base + 0] = pa[0];
        this.bonePos[base + 1] = pa[1];
        this.bonePos[base + 2] = pa[2];
        this.bonePos[base + 3] = pb[0];
        this.bonePos[base + 4] = pb[1];
        this.bonePos[base + 5] = pb[2];
      }
      this.bones.geometry.attributes.position.needsUpdate = true;
    }

    tick(dt) {
      if (this.canvas.offsetParent === null) return;
      if (this.motion && this.playing) {
        this.accum += dt;
        const step = 1.0 / 30.0;
        while (this.accum >= step) {
          this.accum -= step;
          this.frame = (this.frame + 1) % this.motion.length;
        }
        this._renderFrame();
      }
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    }
  };
}

const els = {
  modelName: document.getElementById("modelName"),
  modelNameList: document.getElementById("model-name-list"),
  modelFilename: document.getElementById("modelFilename"),
  vaeCheckpoint: document.getElementById("vaeCheckpoint"),
  dataset: document.getElementById("dataset"),
  samplePath: document.getElementById("samplePath"),
  camera: document.getElementById("camera"),
  conditionFrames: document.getElementById("conditionFrames"),
  steps: document.getElementById("steps"),
  solver: document.getElementById("solver"),
  styleId: document.getElementById("styleId"),
  seed: document.getElementById("seed"),
  start: document.getElementById("start"),
  device: document.getElementById("device"),
  outName: document.getElementById("outName"),
  useEma: document.getElementById("useEma"),
  loadLatestBtn: document.getElementById("loadLatestBtn"),
  replicateBtn: document.getElementById("replicateBtn"),
  generateBtn: document.getElementById("generateBtn"),
  generateSpinner: document.getElementById("generateSpinner"),
  clearBtn: document.getElementById("clearBtn"),
  statusLog: document.getElementById("statusLog"),
  clipVideo: document.getElementById("clipVideo"),
  noVideoMsg: document.getElementById("noVideoMsg"),
  framesMeta: document.getElementById("framesMeta"),
  framesGrid: document.getElementById("framesGrid"),
  lightbox: document.getElementById("lightbox"),
  lightboxImage: document.getElementById("lightboxImage"),
  lightboxPrev: document.getElementById("lightboxPrev"),
  lightboxNext: document.getElementById("lightboxNext"),
  lightboxIndex: document.getElementById("lightboxIndex"),
  genTitle: document.getElementById("genTitle"),
  condTitle: document.getElementById("condTitle"),
  openBlendBtn: document.getElementById("openBlendBtn"),
  comparisonHeaderSpinner: document.getElementById("comparisonHeaderSpinner"),
  blendDialog: document.getElementById("blendDialog"),
  closeBlendDialog: document.getElementById("closeBlendDialog"),
  blendSampleMeta: document.getElementById("blendSampleMeta"),
  stickFrame1: document.getElementById("stickFrame1"),
  stickFrame2: document.getElementById("stickFrame2"),
  stickFrame3: document.getElementById("stickFrame3"),
  stickSourceFrames: document.getElementById("stickSourceFrames"),
  comparisonText: document.getElementById("comparisonText"),
  comparisonTextMeta: document.getElementById("comparisonTextMeta"),
  randomizeCaptionBtn: document.getElementById("randomizeCaptionBtn"),
  buildBlendBtn: document.getElementById("buildBlendBtn"),
  blendSpinner: document.getElementById("blendSpinner"),
  blendDownload: document.getElementById("blendDownload"),
  baselineBlendDownload: document.getElementById("baselineBlendDownload"),
  blendStatus: document.getElementById("blendStatus"),
  baselinePanel: document.getElementById("baselinePanel"),
  baselineSampleMeta: document.getElementById("baselineSampleMeta"),
  baselineCaption: document.getElementById("baselineCaption"),
  stickSketchGrid: document.getElementById("stickSketchGrid"),
};

let genView = new NullViewport();
let condView = new NullViewport();
let mldView = new NullViewport();
let stickmotionView = new NullViewport();
let SkeletonViewportType = null;
let currentReplicateCommand = "";
let styleNameById = new Map([[0, "Unknown"]]);
let currentFrameUrls = [];
let lightboxOpen = false;
let lightboxIndex = 0;
let currentFrameInfo = {};
let currentComparisonSource = null;
let currentComparisonCaption = null;
let activeComparisonJobId = null;
let comparisonPollTimer = null;
let comparisonBusy = false;
let comparisonCaptionLoading = false;
let captionRequestSerial = 0;
let latestResultAvailable = false;
let resultRequestBusy = false;
let viewportsReady = false;
const FORM_STATE_KEY = "flowmimic_web_form_state_v1";
let defaultsCache = {
  default_steps: 8,
  default_solver: "heun",
  default_dataset: "auto",
  default_style_id: null,
  default_device: "",
  default_vae_checkpoint: "",
  configured_vae_checkpoint: "",
};

function appendLog(text) {
  if (!text) return;
  els.statusLog.textContent += `${text}\n`;
  els.statusLog.scrollTop = els.statusLog.scrollHeight;
}

function clearLog() {
  els.statusLog.textContent = "";
}

function updateResultRequestControls() {
  els.generateBtn.disabled = resultRequestBusy;
  els.loadLatestBtn.disabled =
    resultRequestBusy || !latestResultAvailable || !viewportsReady;
  els.generateSpinner.style.display = resultRequestBusy ? "inline-block" : "none";
}

function setResultRequestBusy(busy) {
  resultRequestBusy = busy;
  updateResultRequestControls();
}

function sketchFrameInputs() {
  return [els.stickFrame1, els.stickFrame2, els.stickFrame3];
}

function selectedSketchFrames() {
  return sketchFrameInputs().map((input) => parseOptionalInt(input.value));
}

function updateComparisonCaptionMeta() {
  if (!currentComparisonCaption) {
    els.comparisonTextMeta.textContent = comparisonCaptionLoading
      ? "Selecting a random camera-matched description..."
      : "No description selected.";
    els.comparisonTextMeta.removeAttribute("title");
    return;
  }
  const edited = els.comparisonText.value.trim() !== currentComparisonCaption.text;
  const state = edited ? "Edited" : "Original";
  els.comparisonTextMeta.textContent =
    `Description ${currentComparisonCaption.index + 1} of ${currentComparisonCaption.count} | ${state}`;
  els.comparisonTextMeta.title = currentComparisonCaption.source || "";
}

function updateComparisonControls() {
  const hasCaption = Boolean(
    currentComparisonCaption && els.comparisonText.value.trim()
  );
  els.buildBlendBtn.disabled =
    comparisonBusy || comparisonCaptionLoading || !currentComparisonSource || !hasCaption;
  sketchFrameInputs().forEach((input) => {
    input.disabled = comparisonBusy;
  });
  els.comparisonText.disabled = comparisonBusy || comparisonCaptionLoading;
  els.randomizeCaptionBtn.disabled =
    comparisonBusy || comparisonCaptionLoading || !currentComparisonSource;
  els.openBlendBtn.disabled = comparisonBusy || !currentComparisonSource;
  els.blendSpinner.style.display = comparisonBusy ? "inline-block" : "none";
  els.comparisonHeaderSpinner.style.display = comparisonBusy ? "inline-block" : "none";
}

function setComparisonBusy(busy) {
  comparisonBusy = busy;
  updateComparisonControls();
}

async function loadRandomComparisonCaption(excludeCurrent = false) {
  if (!currentComparisonSource) return;
  const requestSerial = ++captionRequestSerial;
  const sourceResultId = currentComparisonSource.resultId;
  comparisonCaptionLoading = true;
  updateComparisonCaptionMeta();
  updateComparisonControls();
  els.blendStatus.textContent = "Loading text description";
  try {
    const payload = { result_id: sourceResultId };
    if (excludeCurrent && currentComparisonCaption) {
      payload.exclude_index = currentComparisonCaption.index;
    }
    const res = await fetch("./api/comparison-caption", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(
        comparisonErrorText(data, `Description request failed (${res.status})`)
      );
    }
    if (
      requestSerial !== captionRequestSerial ||
      currentComparisonSource?.resultId !== sourceResultId
    ) {
      return;
    }
    currentComparisonCaption = {
      index: Number(data.index),
      count: Number(data.count),
      text: String(data.text || ""),
      source: String(data.source || ""),
    };
    els.comparisonText.value = currentComparisonCaption.text;
    els.blendStatus.textContent = "Ready";
  } catch (err) {
    if (requestSerial !== captionRequestSerial) return;
    currentComparisonCaption = null;
    els.comparisonText.value = "";
    els.blendStatus.textContent = `Description failed: ${err}`;
    appendLog(`Comparison description request failed: ${err}`);
  } finally {
    if (requestSerial === captionRequestSerial) {
      comparisonCaptionLoading = false;
      updateComparisonCaptionMeta();
      updateComparisonControls();
    }
  }
}

function updateSketchSourceFrames() {
  if (!currentComparisonSource) {
    els.stickSourceFrames.textContent = "";
    return;
  }
  const frames = selectedSketchFrames();
  const start = Number(currentComparisonSource.meta?.start || 0);
  if (frames.some((frame) => frame == null)) {
    els.stickSourceFrames.textContent = "Enter three clip-local frame indices.";
    return;
  }
  els.stickSourceFrames.textContent =
    `Clip-local: ${frames.join(", ")} | Source: ${frames.map((frame) => frame + start).join(", ")}`;
}

function clearBaselineResults() {
  els.baselinePanel.hidden = true;
  els.baselineSampleMeta.textContent = "";
  els.baselineCaption.textContent = "";
  els.stickSketchGrid.innerHTML = "";
  els.baselineBlendDownload.hidden = true;
  els.baselineBlendDownload.removeAttribute("href");
  mldView.setMotion(null);
  stickmotionView.setMotion(null);
}

function resetComparisonExport() {
  captionRequestSerial += 1;
  if (comparisonPollTimer != null) {
    window.clearTimeout(comparisonPollTimer);
    comparisonPollTimer = null;
  }
  activeComparisonJobId = null;
  currentComparisonSource = null;
  currentComparisonCaption = null;
  comparisonCaptionLoading = false;
  comparisonBusy = false;
  els.openBlendBtn.disabled = true;
  els.blendDownload.hidden = true;
  els.blendDownload.removeAttribute("href");
  els.blendStatus.textContent = "Ready";
  els.blendSampleMeta.textContent = "";
  els.comparisonText.value = "";
  els.comparisonText.placeholder = "Loading a description for this sample...";
  if (els.blendDialog.open) els.blendDialog.close();
  clearBaselineResults();
  setComparisonBusy(false);
  updateComparisonCaptionMeta();
  updateSketchSourceFrames();
}

function setComparisonSource(data) {
  resetComparisonExport();
  if (data?.meta?.dataset !== "aist" || !data.result_id) {
    return;
  }
  const seqLen = Number(data.meta.seq_len || 0);
  if (seqLen !== 196) {
    return;
  }
  currentComparisonSource = {
    resultId: data.result_id,
    motionFilename: data.generated_motion_name || "result_smpl22.npy",
    meta: data.meta,
  };
  const maxFrame = seqLen - 1;
  sketchFrameInputs().forEach((input) => {
    input.max = String(maxFrame);
  });
  els.stickFrame1.value = "24";
  els.stickFrame2.value = "98";
  els.stickFrame3.value = "171";
  const sampleId = String(data.meta.path || "").split("/").pop()?.replace(/\.pkl$/, "") || "AIST++";
  const camera = data.meta.camera || "?";
  const start = Number(data.meta.start || 0);
  els.blendSampleMeta.textContent = `${sampleId} | camera ${camera} | start ${start}`;
  els.openBlendBtn.disabled = false;
  setComparisonBusy(false);
  updateSketchSourceFrames();
}

function getFormState() {
  return {
    modelName: els.modelName.value,
    modelFilename: els.modelFilename.value,
    vaeCheckpoint: els.vaeCheckpoint.value,
    dataset: els.dataset.value,
    samplePath: els.samplePath.value,
    camera: els.camera.value,
    conditionFrames: els.conditionFrames.value,
    steps: els.steps.value,
    solver: els.solver.value,
    styleId: els.styleId.value,
    seed: els.seed.value,
    start: els.start.value,
    device: els.device.value,
    outName: els.outName.value,
    useEma: els.useEma.checked,
  };
}

function applyFormState(state) {
  if (!state || typeof state !== "object") return;
  if (typeof state.modelName === "string") els.modelName.value = state.modelName;
  if (typeof state.modelFilename === "string") els.modelFilename.value = state.modelFilename;
  if (typeof state.vaeCheckpoint === "string") els.vaeCheckpoint.value = state.vaeCheckpoint;
  if (typeof state.dataset === "string") els.dataset.value = state.dataset;
  if (typeof state.samplePath === "string") els.samplePath.value = state.samplePath;
  if (typeof state.camera === "string") els.camera.value = state.camera;
  if (typeof state.conditionFrames === "string") els.conditionFrames.value = state.conditionFrames;
  if (typeof state.steps === "string") els.steps.value = state.steps;
  if (typeof state.solver === "string") els.solver.value = state.solver;
  if (typeof state.styleId === "string") setStyleSelectValue(state.styleId);
  if (typeof state.seed === "string") els.seed.value = state.seed;
  if (typeof state.start === "string") els.start.value = state.start;
  if (typeof state.device === "string") els.device.value = state.device;
  if (typeof state.outName === "string") els.outName.value = state.outName;
  if (typeof state.useEma === "boolean") els.useEma.checked = state.useEma;
}

function saveFormState() {
  try {
    window.localStorage.setItem(FORM_STATE_KEY, JSON.stringify(getFormState()));
  } catch (_err) {
    // Ignore storage failures (private mode/quota/etc.).
  }
}

function loadFormState() {
  try {
    const raw = window.localStorage.getItem(FORM_STATE_KEY);
    if (!raw) return;
    applyFormState(JSON.parse(raw));
  } catch (_err) {
    // Ignore malformed storage payloads.
  }
}

function clearStaleDefaultVaeOverride() {
  if (
    !defaultsCache.default_vae_checkpoint &&
    defaultsCache.configured_vae_checkpoint &&
    els.vaeCheckpoint.value === defaultsCache.configured_vae_checkpoint
  ) {
    els.vaeCheckpoint.value = "";
    saveFormState();
  }
}

function parseOptionalInt(value) {
  if (value === "" || value == null) return null;
  const n = Number(value);
  if (!Number.isFinite(n)) return null;
  return Math.trunc(n);
}

function splitShellCommand(cmd) {
  const out = [];
  let cur = "";
  let quote = null;
  for (let i = 0; i < cmd.length; i += 1) {
    const ch = cmd[i];
    if (quote === "'") {
      if (ch === "'") {
        quote = null;
      } else {
        cur += ch;
      }
      continue;
    }
    if (quote === "\"") {
      if (ch === "\"") {
        quote = null;
      } else if (ch === "\\" && i + 1 < cmd.length) {
        i += 1;
        cur += cmd[i];
      } else {
        cur += ch;
      }
      continue;
    }
    if (ch === "'" || ch === "\"") {
      quote = ch;
      continue;
    }
    if (/\s/.test(ch)) {
      if (cur) {
        out.push(cur);
        cur = "";
      }
      continue;
    }
    if (ch === "\\" && i + 1 < cmd.length) {
      i += 1;
      cur += cmd[i];
      continue;
    }
    cur += ch;
  }
  if (cur) out.push(cur);
  return out;
}

function parseReplicateArgs(cmd) {
  const tokens = splitShellCommand(cmd || "");
  const args = {};
  for (let i = 0; i < tokens.length; i += 1) {
    const t = tokens[i];
    if (!t.startsWith("--")) continue;
    const key = t.slice(2);
    if (key === "use-ema") {
      args.use_ema = true;
      continue;
    }
    if (i + 1 < tokens.length && !tokens[i + 1].startsWith("--")) {
      args[key] = tokens[i + 1];
      i += 1;
    } else {
      args[key] = "";
    }
  }
  return args;
}

function setStyleSelectValue(raw) {
  if (raw == null || raw === "") {
    els.styleId.value = "";
    return;
  }
  const value = String(raw);
  const hasOption = Array.from(els.styleId.options).some((o) => o.value === value);
  if (!hasOption) {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = value;
    els.styleId.appendChild(opt);
  }
  els.styleId.value = value;
}

function applyReplicateCommand(cmd) {
  const a = parseReplicateArgs(cmd);
  const ckpt = a.checkpoint || "";
  if (ckpt) {
    const parts = ckpt.split("/").filter(Boolean);
    const flowIdx = parts.findIndex((p, idx) => p === "flow" && idx > 0 && parts[idx - 1] === "checkpoints");
    if (flowIdx >= 0 && flowIdx + 2 < parts.length) {
      els.modelName.value = parts[flowIdx + 1];
      els.modelFilename.value = parts[parts.length - 1];
    }
  }
  els.vaeCheckpoint.value = a["vae-checkpoint"] || "";
  els.dataset.value = a.dataset || "auto";
  els.samplePath.value = a["sample-path"] || "";
  els.camera.value = a.camera || "";
  els.conditionFrames.value = a["cond-frames"] || "";
  els.steps.value = a.steps || "8";
  els.solver.value = a.solver || "heun";
  setStyleSelectValue(a["style-id"] || "");
  els.seed.value = a.seed || "";
  els.start.value = a.start || "";
  els.device.value = a.device || "";
  els.outName.value = a.out || "result_smpl22.npy";
  els.useEma.checked = Boolean(a.use_ema);
}

async function initViewports() {
  try {
    const THREE = await import("three");
    const { OrbitControls } = await import(
      "https://unpkg.com/three@0.164.1/examples/jsm/controls/OrbitControls.js"
    );
    const SkeletonViewport = buildSkeletonViewportClass(THREE, OrbitControls);
    SkeletonViewportType = SkeletonViewport;
    genView = new SkeletonViewport("genCanvas", 0x0f5f94, 0x264653);
    condView = new SkeletonViewport("condCanvas", 0x2d7a64, 0x28594d);
    appendLog("3D viewer ready.");
  } catch (err) {
    genView = new NullViewport();
    condView = new NullViewport();
    mldView = new NullViewport();
    stickmotionView = new NullViewport();
    SkeletonViewportType = null;
    appendLog(`3D viewer disabled (failed to load three.js): ${err}`);
  }
}

async function loadDefaults() {
  const res = await fetch("./api/defaults");
  if (!res.ok) return;
  const data = await res.json();
  defaultsCache = {
    ...defaultsCache,
    ...data,
  };
  latestResultAvailable = Boolean(data.last_meta_exists);
  updateResultRequestControls();
  els.vaeCheckpoint.value = data.default_vae_checkpoint || "";
  els.steps.value = data.default_steps || 8;
  els.solver.value = data.default_solver || "heun";
  els.dataset.value = data.default_dataset || "auto";
  if (Array.isArray(data.style_options) && els.styleId) {
    styleNameById = new Map([[0, "Unknown"]]);
    els.styleId.innerHTML = "";
    for (const item of data.style_options) {
      const opt = document.createElement("option");
      opt.value = item.id == null ? "" : String(item.id);
      opt.textContent = item.label;
      els.styleId.appendChild(opt);
      if (item.id != null && typeof item.label === "string") {
        const m = item.label.match(/^(\d+):\s*(.+?)(?:\s*-\s*.+)?$/);
        if (m) styleNameById.set(Number(item.id), m[2]);
      }
    }
    els.styleId.value = data.default_style_id == null ? "" : String(data.default_style_id);
  }
  if (Array.isArray(data.model_names)) {
    for (const c of data.model_names || []) {
      const opt = document.createElement("option");
      opt.value = c;
      els.modelNameList.appendChild(opt);
    }
    if (Array.isArray(data.model_names) && data.model_names.length > 0) {
      const preferred = data.model_names.find((x) => x.includes("reflow_0_solver"));
      els.modelName.value = data.default_model_name || preferred || data.model_names[data.model_names.length - 1];
    }
  }
  els.modelFilename.value = data.default_model_filename || "flow_round0_last.pt";
}

function resetArgsKeepCheckpoints() {
  els.dataset.value = defaultsCache.default_dataset || "auto";
  els.samplePath.value = "";
  els.camera.value = "";
  els.conditionFrames.value = defaultsCache.default_condition_frames == null
    ? ""
    : String(defaultsCache.default_condition_frames);
  els.steps.value = String(defaultsCache.default_steps || 8);
  els.solver.value = defaultsCache.default_solver || "heun";
  els.styleId.value = defaultsCache.default_style_id == null ? "" : String(defaultsCache.default_style_id);
  els.seed.value = "";
  els.start.value = "";
  els.device.value = defaultsCache.default_device || "";
  els.outName.value = "result_smpl22.npy";
  els.useEma.checked = false;

  currentReplicateCommand = "";
  els.replicateBtn.disabled = true;

  clearLog();
  closeLightbox();
  setVideo(null);
  setFrames([], {});
  genView.setMotion(null);
  condView.setMotion(null);
  resetComparisonExport();
  if (els.genTitle) {
    els.genTitle.textContent = "Generated (result_smpl22.npy)";
  }
  if (els.condTitle) {
    els.condTitle.textContent = "Condition Reference (cond_clip_smpl22.npy)";
  }
  appendLog("Arguments reset to defaults (checkpoints preserved). Outputs cleared.");
  saveFormState();
}

function updateFramesMeta() {
  if (!els.framesMeta) return;
  const total = Number(currentFrameInfo.total || currentFrameUrls.length || 0);
  const shown = Number(currentFrameInfo.shown || currentFrameUrls.length || 0);
  if (!total) {
    els.framesMeta.textContent = "";
    return;
  }
  const limit = Number(currentFrameInfo.limit || 0);
  const limitText = shown < total && limit > 0 ? ` (preview capped at ${limit})` : "";
  els.framesMeta.textContent = shown < total
    ? `Frames: ${shown} / ${total}${limitText}`
    : `Frames: ${shown}`;
}

function setFrames(urls, info = {}) {
  currentFrameUrls = Array.isArray(urls) ? urls.slice() : [];
  currentFrameInfo = info && typeof info === "object" ? info : {};
  els.framesGrid.innerHTML = "";
  updateFramesMeta();
  if (!currentFrameUrls || currentFrameUrls.length === 0) return;
  const previewIndices = Array.isArray(currentFrameInfo.preview_indices)
    ? currentFrameInfo.preview_indices
    : [];
  for (let i = 0; i < currentFrameUrls.length; i += 1) {
    const u = currentFrameUrls[i];
    const img = document.createElement("img");
    img.src = u;
    img.loading = "lazy";
    const frameNo = previewIndices[i];
    img.alt = frameNo == null ? `Condition frame ${i + 1}` : `Condition frame ${frameNo}`;
    img.title = frameNo == null ? `Condition frame ${i + 1}` : `Condition frame ${frameNo}`;
    img.addEventListener("click", () => openLightbox(i));
    els.framesGrid.appendChild(img);
  }
}

function setVideo(url) {
  if (!url) {
    els.clipVideo.style.display = "none";
    els.clipVideo.removeAttribute("src");
    els.noVideoMsg.style.display = "block";
    return;
  }
  els.clipVideo.src = url;
  els.clipVideo.style.display = "block";
  els.noVideoMsg.style.display = "none";
}

function updateLightboxView() {
  if (!currentFrameUrls.length) return;
  const idx = ((lightboxIndex % currentFrameUrls.length) + currentFrameUrls.length) % currentFrameUrls.length;
  lightboxIndex = idx;
  els.lightboxImage.src = currentFrameUrls[idx];
  const previewIndices = Array.isArray(currentFrameInfo.preview_indices)
    ? currentFrameInfo.preview_indices
    : [];
  const frameNo = previewIndices[idx];
  const frameText = frameNo == null ? "" : ` - frame ${frameNo}`;
  els.lightboxIndex.textContent = `${idx + 1} / ${currentFrameUrls.length}${frameText}`;
}

function openLightbox(idx) {
  if (!currentFrameUrls.length) return;
  lightboxIndex = idx;
  updateLightboxView();
  lightboxOpen = true;
  els.lightbox.classList.add("open");
  els.lightbox.setAttribute("aria-hidden", "false");
}

function closeLightbox() {
  lightboxOpen = false;
  els.lightbox.classList.remove("open");
  els.lightbox.setAttribute("aria-hidden", "true");
}

function stepLightbox(delta) {
  if (!lightboxOpen || !currentFrameUrls.length) return;
  lightboxIndex += delta;
  updateLightboxView();
}

function formatStyle(idValue) {
  const id = Number(idValue);
  if (!Number.isFinite(id)) return "0:Unknown";
  const name = styleNameById.get(id) || "Unknown";
  return `${id}:${name}`;
}

function updateViewportTitles(meta) {
  const genText = formatStyle(meta?.style_id ?? 0);
  const condText = formatStyle(meta?.cond_style_id ?? 0);
  if (els.genTitle) {
    els.genTitle.textContent = `Generated (${genText})`;
  }
  if (els.condTitle) {
    els.condTitle.textContent = `Condition Reference (${condText})`;
  }
}

function comparisonErrorText(data, fallback) {
  if (typeof data?.detail === "string") return data.detail;
  if (typeof data?.error === "string") return data.error;
  return fallback;
}

function canvasContext(canvas) {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const width = Math.max(180, Math.round(canvas.clientWidth || 320));
  const height = Math.max(160, Math.round(canvas.clientHeight || 220));
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const context = canvas.getContext("2d");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);
  return { context, width, height };
}

function projectedPointTransform(paths, width, height, padding = 18) {
  const points = [];
  for (const path of paths) {
    for (const point of path || []) {
      if (
        Array.isArray(point) &&
        Number.isFinite(point[0]) &&
        Number.isFinite(point[1])
      ) {
        points.push(point);
      }
    }
  }
  if (!points.length) return null;
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanX = Math.max(maxX - minX, 1e-6);
  const spanY = Math.max(maxY - minY, 1e-6);
  const scale = Math.min(
    (width - padding * 2) / spanX,
    (height - padding * 2) / spanY
  );
  const contentWidth = spanX * scale;
  const contentHeight = spanY * scale;
  const offsetX = (width - contentWidth) * 0.5;
  const offsetY = (height - contentHeight) * 0.5;
  return (point) => [
    offsetX + (point[0] - minX) * scale,
    height - offsetY - (point[1] - minY) * scale,
  ];
}

function drawPaths(canvas, paths, color, lineWidth = 2) {
  const { context, width, height } = canvasContext(canvas);
  context.fillStyle = "#f9faf8";
  context.fillRect(0, 0, width, height);
  const transform = projectedPointTransform(paths, width, height);
  if (!transform) return;
  context.strokeStyle = color;
  context.lineWidth = lineWidth;
  context.lineCap = "round";
  context.lineJoin = "round";
  for (const path of paths) {
    if (!Array.isArray(path) || path.length < 2) continue;
    context.beginPath();
    path.forEach((point, index) => {
      const [x, y] = transform(point);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.stroke();
  }
  return { context, transform };
}

function renderStickMotionConditions(data) {
  els.stickSketchGrid.innerHTML = "";
  const localFrames = data.stickman_frame_indices || [];
  const sourceFrames = data.stickman_source_frame_indices || [];
  const canvases = [];
  for (let index = 0; index < (data.stickman_tracks || []).length; index += 1) {
    const figure = document.createElement("figure");
    figure.className = "stick-sketch";
    const canvas = document.createElement("canvas");
    canvas.className = "condition-canvas";
    const caption = document.createElement("figcaption");
    const local = localFrames[index] ?? "?";
    const source = sourceFrames[index];
    caption.textContent = source == null
      ? `Frame ${local}`
      : `Frame ${local} | source ${source}`;
    figure.append(canvas, caption);
    els.stickSketchGrid.appendChild(figure);
    canvases.push([canvas, data.stickman_tracks[index]]);
  }
  window.requestAnimationFrame(() => {
    for (const [canvas, tracks] of canvases) {
      drawPaths(canvas, tracks, "#263238", 2.2);
    }
  });
}

async function loadComparisonResults(resultsUrl) {
  const res = await fetch(resultsUrl, { cache: "no-store" });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(comparisonErrorText(data, `Result request failed (${res.status})`));
  }
  els.baselinePanel.hidden = false;
  if (SkeletonViewportType && mldView instanceof NullViewport) {
    mldView = new SkeletonViewportType("mldCanvas", 0xe36b32, 0x9e3f1e);
    stickmotionView = new SkeletonViewportType("stickmotionCanvas", 0x9a67b2, 0x633974);
  }
  els.baselineSampleMeta.textContent = `${data.sample_id} | start ${data.clip_start}`;
  els.baselineCaption.textContent = data.mld_text === data.stickmotion_text
    ? data.mld_text
    : `MLD: ${data.mld_text} | StickMotion: ${data.stickmotion_text}`;
  mldView.setMotion(data.mld_motion);
  stickmotionView.setMotion(data.stickmotion_motion);
  renderStickMotionConditions(data);
  els.baselinePanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function pollComparisonJob(jobId, statusUrl) {
  if (activeComparisonJobId !== jobId) return;
  try {
    const res = await fetch(statusUrl, { cache: "no-store" });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(comparisonErrorText(data, `Status request failed (${res.status})`));
    }
    if (activeComparisonJobId !== jobId) return;
    els.blendStatus.textContent = data.stage || data.status || "Working";
    if (data.status === "complete") {
      setComparisonBusy(false);
      els.blendDownload.href = data.download_url;
      els.blendDownload.hidden = false;
      els.baselineBlendDownload.href = data.download_url;
      els.baselineBlendDownload.hidden = false;
      appendLog(`Comparison blend ready: ${data.download_url}`);
      try {
        await loadComparisonResults(data.results_url);
      } catch (err) {
        appendLog(`Comparison visualization failed: ${err}`);
      }
      comparisonPollTimer = null;
      return;
    }
    if (data.status === "failed") {
      setComparisonBusy(false);
      const message = data.error || "Comparison blend failed.";
      els.blendStatus.textContent = message;
      appendLog(`Comparison blend failed: ${message}`);
      if (data.log_url) appendLog(`build log: ${data.log_url}`);
      comparisonPollTimer = null;
      return;
    }
    comparisonPollTimer = window.setTimeout(
      () => pollComparisonJob(jobId, statusUrl),
      1500
    );
  } catch (err) {
    if (activeComparisonJobId !== jobId) return;
    els.blendStatus.textContent = `Status check failed: ${err}`;
    comparisonPollTimer = window.setTimeout(
      () => pollComparisonJob(jobId, statusUrl),
      3000
    );
  }
}

async function onBuildBlend() {
  if (!currentComparisonSource) return;
  const frames = selectedSketchFrames();
  const captionText = els.comparisonText.value.replaceAll("#", " ").trim();
  const seqLen = Number(currentComparisonSource.meta?.seq_len || 0);
  if (
    frames.some((frame) => frame == null || frame < 0 || frame >= seqLen) ||
    new Set(frames).size !== 3
  ) {
    els.blendStatus.textContent = `Select three distinct frames within [0, ${seqLen - 1}].`;
    return;
  }
  if (!currentComparisonCaption || !captionText) {
    els.blendStatus.textContent = "Select or enter a text description first.";
    return;
  }
  if (comparisonPollTimer != null) {
    window.clearTimeout(comparisonPollTimer);
    comparisonPollTimer = null;
  }
  activeComparisonJobId = null;
  els.blendDownload.hidden = true;
  els.blendDownload.removeAttribute("href");
  setComparisonBusy(true);
  els.blendStatus.textContent = "Submitting build";
  try {
    const res = await fetch("./api/comparison-jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        result_id: currentComparisonSource.resultId,
        motion_filename: currentComparisonSource.motionFilename,
        stickmotion_sketch_frames: frames,
        caption_index: currentComparisonCaption.index,
        caption_text: captionText,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(comparisonErrorText(data, `Build request failed (${res.status})`));
    }
    activeComparisonJobId = data.job_id;
    els.blendStatus.textContent = data.stage || "Queued";
    appendLog(
      `Comparison blend queued: description ${currentComparisonCaption.index + 1}/${currentComparisonCaption.count}, sketches [${frames.join(", ")}], job ${data.job_id}`
    );
    pollComparisonJob(data.job_id, data.status_url);
  } catch (err) {
    setComparisonBusy(false);
    els.blendStatus.textContent = `Build failed: ${err}`;
    appendLog(`Comparison build request failed: ${err}`);
  }
}

function displayGeneratedResult(data) {
  currentReplicateCommand = data.meta?.replicate_command || "";
  els.replicateBtn.disabled = !currentReplicateCommand;
  if (currentReplicateCommand) {
    appendLog(`replicate_command: ${currentReplicateCommand}`);
  }
  updateViewportTitles(data.meta || {});
  genView.setMotion(data.generated_motion);
  condView.setMotion(data.condition_motion);
  setVideo(data.video_url);
  setFrames(data.frame_urls || [], data.condition_frame_info || {});
  setComparisonSource(data);
  latestResultAvailable = true;
  updateResultRequestControls();
}

async function onLoadLatest() {
  setResultRequestBusy(true);
  clearLog();
  appendLog("Loading latest generated result ...");
  try {
    const res = await fetch("./api/results/latest", { cache: "no-store" });
    const data = await res.json();
    if (!res.ok) {
      if (res.status === 404) {
        latestResultAvailable = false;
      }
      throw new Error(
        comparisonErrorText(data, `Latest result request failed (${res.status})`)
      );
    }
    displayGeneratedResult(data);
    appendLog(`Loaded existing result: ${data.result_dir}`);
    appendLog("No generation or media extraction was run.");
    if (!data.condition_motion || !data.video_url) {
      appendLog("Some condition media is unavailable for this saved result.");
    }
  } catch (err) {
    appendLog(`Latest result load failed: ${err}`);
  } finally {
    setResultRequestBusy(false);
  }
}

async function onGenerate() {
  const payload = {
    model_name: els.modelName.value.trim() || null,
    model_filename: els.modelFilename.value.trim() || null,
    vae_checkpoint: els.vaeCheckpoint.value.trim() || null,
    dataset: els.dataset.value,
    sample_path: els.samplePath.value.trim() || null,
    camera: els.camera.value.trim() || null,
    condition_frames: parseOptionalInt(els.conditionFrames.value),
    steps: parseOptionalInt(els.steps.value) ?? 8,
    solver: els.solver.value,
    style_id: parseOptionalInt(els.styleId?.value),
    seed: parseOptionalInt(els.seed.value),
    start: parseOptionalInt(els.start.value),
    device: els.device.value.trim() || null,
    out: els.outName.value.trim() || "result_smpl22.npy",
    use_ema: els.useEma.checked,
    out_dir: "output/flow",
  };
  if (payload.style_id == null) {
    delete payload.style_id;
  }

  if (!payload.model_name || !payload.model_filename) {
    appendLog("model name and model filename are required.");
    return;
  }

  setResultRequestBusy(true);
  resetComparisonExport();
  clearLog();
  appendLog("Running sample_flow.py ...");
  appendLog(
    `Condition frames: ${payload.condition_frames == null ? "checkpoint default" : payload.condition_frames}`
  );
  try {
    const res = await fetch("./api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      appendLog("Generation failed.");
      appendLog(JSON.stringify(data, null, 2));
      return;
    }

    appendLog("sample_flow.py finished.");
    appendLog("extract_cond_media.py finished.");
    if (data.sample_run?.stdout) appendLog(data.sample_run.stdout);
    if (data.sample_run?.stderr) appendLog(data.sample_run.stderr);
    if (data.extract_run?.stdout) appendLog(data.extract_run.stdout);
    if (data.extract_run?.stderr) appendLog(data.extract_run.stderr);
    appendLog(`result_dir: ${data.result_dir}`);
    displayGeneratedResult(data);
  } catch (err) {
    appendLog(`Request failed: ${err}`);
  } finally {
    setResultRequestBusy(false);
  }
}

function onReplicate() {
  if (!currentReplicateCommand) {
    appendLog("No replicate command available yet.");
    return;
  }
  applyReplicateCommand(currentReplicateCommand);
  appendLog("Replicate command loaded into form.");
  saveFormState();
}

function onLightboxKeydown(ev) {
  if (!lightboxOpen) return;
  if (ev.key === "Escape") {
    ev.preventDefault();
    closeLightbox();
    return;
  }
  if (ev.key === "ArrowLeft") {
    ev.preventDefault();
    stepLightbox(-1);
    return;
  }
  if (ev.key === "ArrowRight") {
    ev.preventDefault();
    stepLightbox(1);
  }
}

els.generateBtn.addEventListener("click", onGenerate);
els.loadLatestBtn.addEventListener("click", onLoadLatest);
els.replicateBtn.addEventListener("click", onReplicate);
els.clearBtn.addEventListener("click", resetArgsKeepCheckpoints);
els.openBlendBtn.addEventListener("click", () => {
  if (!currentComparisonSource) return;
  els.blendDialog.showModal();
  if (!currentComparisonCaption && !comparisonCaptionLoading) {
    loadRandomComparisonCaption();
  }
});
els.closeBlendDialog.addEventListener("click", () => els.blendDialog.close());
els.blendDialog.addEventListener("click", (ev) => {
  if (ev.target === els.blendDialog) els.blendDialog.close();
});
els.buildBlendBtn.addEventListener("click", onBuildBlend);
els.randomizeCaptionBtn.addEventListener("click", () => {
  loadRandomComparisonCaption(true);
});
els.comparisonText.addEventListener("input", () => {
  updateComparisonCaptionMeta();
  updateComparisonControls();
});
sketchFrameInputs().forEach((input) => {
  input.addEventListener("input", updateSketchSourceFrames);
});
els.lightboxPrev.addEventListener("click", () => stepLightbox(-1));
els.lightboxNext.addEventListener("click", () => stepLightbox(1));
els.lightbox.addEventListener("click", (ev) => {
  const t = ev.target;
  if (t === els.lightboxImage) return;
  if (t === els.lightboxPrev || t === els.lightboxNext) return;
  closeLightbox();
});
window.addEventListener("keydown", onLightboxKeydown);
[
  els.modelName,
  els.modelFilename,
  els.vaeCheckpoint,
  els.dataset,
  els.samplePath,
  els.camera,
  els.conditionFrames,
  els.steps,
  els.solver,
  els.styleId,
  els.seed,
  els.start,
  els.device,
  els.outName,
  els.useEma,
].forEach((el) => {
  if (!el) return;
  el.addEventListener("input", saveFormState);
  el.addEventListener("change", saveFormState);
});

let lastTs = performance.now();
function animate(ts) {
  const dt = Math.max(0.0, (ts - lastTs) / 1000.0);
  lastTs = ts;
  genView.tick(dt);
  condView.tick(dt);
  mldView.tick(dt);
  stickmotionView.tick(dt);
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);

async function boot() {
  resetComparisonExport();
  await loadDefaults();
  loadFormState();
  clearStaleDefaultVaeOverride();
  await initViewports();
  viewportsReady = true;
  updateResultRequestControls();
}

boot();
