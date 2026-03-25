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
    constructor(canvasId) {
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
      const sphereMat = new THREE.MeshStandardMaterial({ color: 0x0f5f94 });
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
        new THREE.LineBasicMaterial({ color: 0x264653, linewidth: 1 })
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
  steps: document.getElementById("steps"),
  solver: document.getElementById("solver"),
  styleId: document.getElementById("styleId"),
  seed: document.getElementById("seed"),
  start: document.getElementById("start"),
  device: document.getElementById("device"),
  outName: document.getElementById("outName"),
  useEma: document.getElementById("useEma"),
  replicateBtn: document.getElementById("replicateBtn"),
  generateBtn: document.getElementById("generateBtn"),
  generateSpinner: document.getElementById("generateSpinner"),
  clearBtn: document.getElementById("clearBtn"),
  statusLog: document.getElementById("statusLog"),
  clipVideo: document.getElementById("clipVideo"),
  noVideoMsg: document.getElementById("noVideoMsg"),
  framesGrid: document.getElementById("framesGrid"),
  lightbox: document.getElementById("lightbox"),
  lightboxImage: document.getElementById("lightboxImage"),
  lightboxPrev: document.getElementById("lightboxPrev"),
  lightboxNext: document.getElementById("lightboxNext"),
  lightboxIndex: document.getElementById("lightboxIndex"),
  genTitle: document.getElementById("genTitle"),
  condTitle: document.getElementById("condTitle"),
};

let genView = new NullViewport();
let condView = new NullViewport();
let currentReplicateCommand = "";
let styleNameById = new Map([[0, "Unknown"]]);
let currentFrameUrls = [];
let lightboxOpen = false;
let lightboxIndex = 0;
let defaultsCache = {
  default_steps: 8,
  default_solver: "heun",
  default_dataset: "auto",
  default_style_id: null,
  default_device: "",
};

function appendLog(text) {
  if (!text) return;
  els.statusLog.textContent += `${text}\n`;
  els.statusLog.scrollTop = els.statusLog.scrollHeight;
}

function clearLog() {
  els.statusLog.textContent = "";
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
    genView = new SkeletonViewport("genCanvas");
    condView = new SkeletonViewport("condCanvas");
    appendLog("3D viewer ready.");
  } catch (err) {
    genView = new NullViewport();
    condView = new NullViewport();
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
  setFrames([]);
  genView.setMotion(null);
  condView.setMotion(null);
  if (els.genTitle) {
    els.genTitle.textContent = "Generated (result_smpl22.npy)";
  }
  if (els.condTitle) {
    els.condTitle.textContent = "Condition Reference (cond_clip_smpl22.npy)";
  }
  appendLog("Arguments reset to defaults (checkpoints preserved). Outputs cleared.");
}

function setFrames(urls) {
  currentFrameUrls = Array.isArray(urls) ? urls.slice() : [];
  els.framesGrid.innerHTML = "";
  if (!currentFrameUrls || currentFrameUrls.length === 0) return;
  for (let i = 0; i < currentFrameUrls.length; i += 1) {
    const u = currentFrameUrls[i];
    const img = document.createElement("img");
    img.src = u;
    img.loading = "lazy";
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
  els.lightboxIndex.textContent = `${idx + 1} / ${currentFrameUrls.length}`;
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

async function onGenerate() {
  const payload = {
    model_name: els.modelName.value.trim() || null,
    model_filename: els.modelFilename.value.trim() || null,
    vae_checkpoint: els.vaeCheckpoint.value.trim() || null,
    dataset: els.dataset.value,
    sample_path: els.samplePath.value.trim() || null,
    camera: els.camera.value.trim() || null,
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

  els.generateBtn.disabled = true;
  if (els.generateSpinner) els.generateSpinner.style.display = "inline-block";
  clearLog();
  appendLog("Running sample_flow.py ...");
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
    if (data.meta?.replicate_command) {
      currentReplicateCommand = data.meta.replicate_command;
      els.replicateBtn.disabled = false;
      appendLog(`replicate_command: ${currentReplicateCommand}`);
    }
    updateViewportTitles(data.meta || {});

    genView.setMotion(data.generated_motion);
    condView.setMotion(data.condition_motion);
    setVideo(data.video_url);
    setFrames(data.frame_urls || []);
  } catch (err) {
    appendLog(`Request failed: ${err}`);
  } finally {
    els.generateBtn.disabled = false;
    if (els.generateSpinner) els.generateSpinner.style.display = "none";
  }
}

function onReplicate() {
  if (!currentReplicateCommand) {
    appendLog("No replicate command available yet.");
    return;
  }
  applyReplicateCommand(currentReplicateCommand);
  appendLog("Replicate command loaded into form.");
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
els.replicateBtn.addEventListener("click", onReplicate);
els.clearBtn.addEventListener("click", resetArgsKeepCheckpoints);
els.lightboxPrev.addEventListener("click", () => stepLightbox(-1));
els.lightboxNext.addEventListener("click", () => stepLightbox(1));
els.lightbox.addEventListener("click", (ev) => {
  const t = ev.target;
  if (t === els.lightboxImage) return;
  if (t === els.lightboxPrev || t === els.lightboxNext) return;
  closeLightbox();
});
window.addEventListener("keydown", onLightboxKeydown);

let lastTs = performance.now();
function animate(ts) {
  const dt = Math.max(0.0, (ts - lastTs) / 1000.0);
  lastTs = ts;
  genView.tick(dt);
  condView.tick(dt);
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);

loadDefaults();
initViewports();
