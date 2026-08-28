const EDGES = [
  [0, 1], [0, 2], [0, 3],
  [1, 4], [4, 7], [7, 10],
  [2, 5], [5, 8], [8, 11],
  [3, 6], [6, 9], [9, 12], [12, 15],
  [9, 13], [13, 16], [16, 18], [18, 20],
  [9, 14], [14, 17], [17, 19], [19, 21],
];

const SMPL22_BONES = [
  "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
  "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
  "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
  "left_elbow", "right_elbow", "left_wrist", "right_wrist",
];

const PRIMARY_CHILD = new Map([
  [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10],
  [8, 11], [9, 12], [12, 15], [13, 16], [14, 17], [16, 18], [17, 19],
  [18, 20], [19, 21],
]);

const ORIENTATION_EDGE = new Map([
  ...PRIMARY_CHILD.entries()].map(([joint, child]) => [joint, [joint, child]])
);
// The pelvis uses its hip triangle; pointing it at spine1 shears the waist.
ORIENTATION_EDGE.delete(0);
ORIENTATION_EDGE.set(15, [12, 15]);
ORIENTATION_EDGE.set(20, [18, 20]);
ORIENTATION_EDGE.set(21, [19, 21]);

class NullViewport {
  setMotion(_motion) {}
  setVisualizationMode(_mode) {}
  tick(_dt) {}
}

function buildMotionViewportClass(THREE, OrbitControls, rigTemplate, cloneRig) {
  const unitScale = new THREE.Vector3(1, 1, 1);

  function pelvisBasisQuaternion(points) {
    const x = points[1].clone().sub(points[2]);
    const hipCenter = points[1].clone().add(points[2]).multiplyScalar(0.5);
    const y = points[0].clone().sub(hipCenter);
    if (x.lengthSq() < 1e-10 || y.lengthSq() < 1e-10) return null;
    x.normalize();
    const z = x.clone().cross(y).normalize();
    if (z.lengthSq() < 1e-10) return null;
    y.copy(z).cross(x).normalize();
    return new THREE.Quaternion().setFromRotationMatrix(
      new THREE.Matrix4().makeBasis(x, y, z)
    );
  }

  return class MotionViewport {
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
      this.renderer.outputColorSpace = THREE.SRGBColorSpace;

      this.controls = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.enableZoom = false;
      this.controls.target.set(0, 0.9, 0);

      const hemi = new THREE.HemisphereLight(0xffffff, 0x98a1aa, 1.2);
      this.scene.add(hemi);
      const key = new THREE.DirectionalLight(0xffffff, 1.4);
      key.position.set(2.5, 4.0, 3.0);
      this.scene.add(key);
      this.scene.add(new THREE.GridHelper(6, 18, 0xcad1ca, 0xe7ebe7));

      this.skeletonGroup = new THREE.Group();
      this.scene.add(this.skeletonGroup);
      this.joints = [];
      const sphereGeo = new THREE.SphereGeometry(0.025, 12, 12);
      const sphereMat = new THREE.MeshStandardMaterial({ color: jointColor });
      for (let i = 0; i < 22; i += 1) {
        const s = new THREE.Mesh(sphereGeo, sphereMat);
        this.skeletonGroup.add(s);
        this.joints.push(s);
      }

      this.bonePos = new Float32Array(EDGES.length * 2 * 3);
      const boneGeo = new THREE.BufferGeometry();
      boneGeo.setAttribute("position", new THREE.BufferAttribute(this.bonePos, 3));
      this.bones = new THREE.LineSegments(
        boneGeo,
        new THREE.LineBasicMaterial({ color: boneColor, linewidth: 1 })
      );
      this.skeletonGroup.add(this.bones);

      this.rigRoot = null;
      this.rigBones = [];
      this.rigRest = [];
      this.rigRestBasis = null;
      if (rigTemplate && cloneRig) this._initRig(rigTemplate, cloneRig);

      this.motion = null;
      this.frame = 0;
      this.accum = 0.0;
      this.playing = true;
      this._resize();
      window.addEventListener("resize", () => this._resize());
    }

    _initRig(template, cloneRig) {
      this.rigRoot = cloneRig(template);
      const discard = [];
      this.rigRoot.traverse((obj) => {
        if (obj.isCamera || obj.isLight || (obj.isMesh && !obj.isSkinnedMesh)) {
          discard.push(obj);
        }
      });
      for (const obj of discard) obj.parent?.remove(obj);
      this.scene.add(this.rigRoot);
      this.rigRoot.updateMatrixWorld(true);
      this.rigBones = SMPL22_BONES.map((name) => this.rigRoot.getObjectByName(name));
      if (this.rigBones.some((bone) => !bone || !bone.isBone)) {
        this.scene.remove(this.rigRoot);
        this.rigRoot = null;
        this.rigBones = [];
        return;
      }
      const restPoints = this.rigBones.map((bone) => bone.getWorldPosition(new THREE.Vector3()));
      this.rigRestBasis = pelvisBasisQuaternion(restPoints);
      this.rigRest = this.rigBones.map((bone, index) => {
        const edge = ORIENTATION_EDGE.get(index);
        return {
          quaternion: bone.getWorldQuaternion(new THREE.Quaternion()),
          direction: edge == null
            ? null
            : restPoints[edge[1]].clone().sub(restPoints[edge[0]]).normalize(),
        };
      });
      this.rigRoot.visible = false;
    }

    setVisualizationMode(mode) {
      const useRig = mode === "rigged" && this.rigRoot != null;
      this.skeletonGroup.visible = !useRig;
      if (this.rigRoot) this.rigRoot.visible = useRig;
      if (this.motion) this._renderFrame();
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
      if (this.rigRoot?.visible) this._setRigPose(f);
    }

    _setRigPose(frame) {
      if (!this.rigRestBasis) return;
      const points = frame.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
      const targetBasis = pelvisBasisQuaternion(points);
      if (!targetBasis) return;
      const basisDelta = targetBasis.clone().multiply(this.rigRestBasis.clone().invert());
      const desiredWorld = new Map();
      const inverseParent = new THREE.Matrix4();
      const localMatrix = new THREE.Matrix4();
      for (let index = 0; index < this.rigBones.length; index += 1) {
        const bone = this.rigBones[index];
        const rest = this.rigRest[index];
        const worldQuaternion = basisDelta.clone().multiply(rest.quaternion);
        const edge = ORIENTATION_EDGE.get(index);
        if (edge && rest.direction) {
          const source = rest.direction.clone().applyQuaternion(basisDelta).normalize();
          const target = points[edge[1]].clone().sub(points[edge[0]]);
          if (target.lengthSq() > 1e-10) {
            const align = new THREE.Quaternion().setFromUnitVectors(source, target.normalize());
            worldQuaternion.premultiply(align);
          }
        }
        const worldMatrix = new THREE.Matrix4().compose(
          points[index], worldQuaternion, unitScale
        );
        desiredWorld.set(bone, worldMatrix);
        const parentWorld = desiredWorld.get(bone.parent) || bone.parent.matrixWorld;
        inverseParent.copy(parentWorld).invert();
        localMatrix.multiplyMatrices(inverseParent, worldMatrix);
        localMatrix.decompose(bone.position, bone.quaternion, bone.scale);
        bone.updateMatrix();
        bone.updateMatrixWorld(true);
      }
      this.rigRoot.updateMatrixWorld(true);
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
  checkpointPreset: document.getElementById("checkpointPreset"),
  modelName: document.getElementById("modelName"),
  modelNameList: document.getElementById("model-name-list"),
  modelFilename: document.getElementById("modelFilename"),
  vaeCheckpoint: document.getElementById("vaeCheckpoint"),
  vaeCheckpointList: document.getElementById("vae-checkpoint-list"),
  dataset: document.getElementById("dataset"),
  samplePath: document.getElementById("samplePath"),
  camera: document.getElementById("camera"),
  conditionFrames: document.getElementById("conditionFrames"),
  conditionPattern: document.getElementById("conditionPattern"),
  guidanceScaleSlider: document.getElementById("guidanceScaleSlider"),
  guidanceScaleInput: document.getElementById("guidanceScaleInput"),
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
  visualizationModes: [...document.querySelectorAll('input[name="visualizationMode"]')],
  riggedMode: document.getElementById("visualizationRigged"),
  openBlendBtn: document.getElementById("openBlendBtn"),
  comparisonHeaderSpinner: document.getElementById("comparisonHeaderSpinner"),
  comparisonHeaderStatus: document.getElementById("comparisonHeaderStatus"),
  blendDialog: document.getElementById("blendDialog"),
  closeBlendDialog: document.getElementById("closeBlendDialog"),
  blendSampleMeta: document.getElementById("blendSampleMeta"),
  stickFrame1: document.getElementById("stickFrame1"),
  stickFrame2: document.getElementById("stickFrame2"),
  stickFrame3: document.getElementById("stickFrame3"),
  stickSourceFrames: document.getElementById("stickSourceFrames"),
  comparisonDevice: document.getElementById("comparisonDevice"),
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
let MotionViewportType = null;
let currentVisualizationMode = "skeleton";
let currentReplicateCommand = "";
let styleNameById = new Map([[0, "Unknown"]]);
let currentFrameUrls = [];
let currentVideoUrl = "";
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
let currentComparisonIdentity = null;
let currentBaselineIdentity = null;
let comparisonRestoreJobId = null;
let latestResultAvailable = false;
let resultRequestBusy = false;
let activeGenerationJobId = null;
let generationPollTimer = null;
let generationPreviewLoaded = false;
let generationVideoLoaded = false;
let generationMotionLoaded = false;
let generationStage = "";
let viewportsReady = false;
const FORM_STATE_KEY = "flowmimic_web_form_state_v1";
const FORM_STATE_VERSION = 5;
let defaultsCache = {
  default_steps: 8,
  default_solver: "heun",
  default_guidance_scale: 5.0,
  default_condition_pattern: "even",
  default_dataset: "aist",
  default_style_id: null,
  default_device: "",
  default_vae_checkpoint: "",
  configured_vae_checkpoint: "",
  rigged_model_url: "./assets/smpl22_rigged_calibrated.glb",
  checkpoint_presets: [],
};

function checkpointPresetById(id) {
  return (defaultsCache.checkpoint_presets || []).find((item) => item.id === id) || null;
}

function syncCheckpointPreset() {
  const preset = (defaultsCache.checkpoint_presets || []).find(
    (item) => item.model_name === els.modelName.value.trim()
      && item.model_filename === els.modelFilename.value.trim()
  );
  els.checkpointPreset.value = preset?.id || "";
}

function applyCheckpointPreset(id, persist = true) {
  const preset = checkpointPresetById(id);
  if (!preset) return;
  els.modelName.value = preset.model_name;
  els.modelFilename.value = preset.model_filename;
  els.vaeCheckpoint.value = "";
  els.steps.value = String(preset.steps);
  renderGuidanceScale(preset.guidance_scale);
  els.useEma.checked = true;
  if (persist) saveFormState();
}

function appendLog(text) {
  if (!text) return;
  els.statusLog.textContent += `${text}\n`;
  els.statusLog.scrollTop = els.statusLog.scrollHeight;
}

function clearLog() {
  els.statusLog.textContent = "";
}

function guidanceScaleBounds() {
  return {
    min: Number(els.guidanceScaleSlider.min),
    max: Number(els.guidanceScaleSlider.max),
  };
}

function parseGuidanceScale(value) {
  if (value === "" || value == null) return null;
  const number = Number(value);
  const { min, max } = guidanceScaleBounds();
  if (!Number.isFinite(number) || number < min || number > max) return null;
  return number;
}

function formatGuidanceScaleInput(value) {
  return Number(value).toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
}

function renderGuidanceScale(value, updateInput = true) {
  const number = parseGuidanceScale(value);
  if (number == null) return false;
  els.guidanceScaleSlider.value = String(number);
  if (updateInput) {
    els.guidanceScaleInput.value = formatGuidanceScaleInput(number);
  }
  els.guidanceScaleInput.setCustomValidity("");
  return true;
}

function onGuidanceSliderInput() {
  renderGuidanceScale(els.guidanceScaleSlider.value);
  saveFormState();
}

function onGuidanceNumberInput(commit = false) {
  const number = parseGuidanceScale(els.guidanceScaleInput.value);
  if (number == null) {
    const { min, max } = guidanceScaleBounds();
    els.guidanceScaleInput.setCustomValidity(
      `Enter a guidance scale between ${min} and ${max}.`
    );
    if (commit) {
      renderGuidanceScale(els.guidanceScaleSlider.value);
    }
    return;
  }
  renderGuidanceScale(number, commit);
  saveFormState();
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
  els.comparisonDevice.disabled = comparisonBusy;
  els.comparisonText.disabled = comparisonBusy || comparisonCaptionLoading;
  els.randomizeCaptionBtn.disabled =
    comparisonBusy || comparisonCaptionLoading || !currentComparisonSource;
  els.openBlendBtn.disabled = !currentComparisonSource;
  els.blendSpinner.style.display = comparisonBusy ? "inline-block" : "none";
  els.comparisonHeaderSpinner.style.display = comparisonBusy ? "inline-block" : "none";
}

function setComparisonBusy(busy) {
  comparisonBusy = busy;
  updateComparisonControls();
}

function setComparisonJobStage(message = "", state = "working") {
  els.comparisonHeaderStatus.textContent = message;
  els.comparisonHeaderStatus.dataset.state = message ? state : "";
}

function setBlendDownloadPending(pending) {
  els.blendDownload.hidden = !pending;
  els.blendDownload.removeAttribute("href");
  if (pending) {
    els.blendDownload.setAttribute("aria-disabled", "true");
    els.blendDownload.setAttribute("tabindex", "-1");
  } else {
    els.blendDownload.removeAttribute("aria-disabled");
    els.blendDownload.removeAttribute("tabindex");
  }
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
  currentBaselineIdentity = null;
}

function comparisonIdentity(meta) {
  const rawPath = String(meta?.path || "");
  const sampleId = rawPath.split("/").pop()?.replace(/\.[^.]+$/, "") || "";
  if (!sampleId) return null;
  return `${sampleId}::${Number(meta?.start || 0)}`;
}

function resetComparisonExport() {
  captionRequestSerial += 1;
  if (comparisonPollTimer != null) {
    window.clearTimeout(comparisonPollTimer);
    comparisonPollTimer = null;
  }
  activeComparisonJobId = null;
  currentComparisonSource = null;
  currentComparisonIdentity = null;
  comparisonRestoreJobId = null;
  currentComparisonCaption = null;
  comparisonCaptionLoading = false;
  comparisonBusy = false;
  els.openBlendBtn.disabled = true;
  els.blendDownload.hidden = true;
  els.blendDownload.removeAttribute("href");
  els.blendDownload.removeAttribute("aria-disabled");
  els.blendDownload.removeAttribute("tabindex");
  els.blendStatus.textContent = "Ready";
  setComparisonJobStage();
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
  if (data?.meta?.dataset !== "aist" || !data.result_id) {
    resetComparisonExport();
    return;
  }
  const seqLen = Number(data.meta.seq_len || 0);
  if (seqLen !== 196) {
    resetComparisonExport();
    return;
  }
  const nextIdentity = comparisonIdentity(data.meta);
  const sameClip = nextIdentity != null && nextIdentity === currentComparisonIdentity;
  if (!sameClip) {
    resetComparisonExport();
  } else {
    els.blendDownload.hidden = true;
    els.blendDownload.removeAttribute("href");
    els.baselineBlendDownload.hidden = true;
    els.baselineBlendDownload.removeAttribute("href");
    setComparisonJobStage();
  }
  currentComparisonSource = {
    resultId: data.result_id,
    motionFilename: data.generated_motion_name || "result_smpl22.npy",
    meta: data.meta,
  };
  currentComparisonIdentity = nextIdentity;
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
  restoreComparisonForResult(data);
}

function getFormState() {
  return {
    version: FORM_STATE_VERSION,
    checkpointPreset: els.checkpointPreset.value,
    modelName: els.modelName.value,
    modelFilename: els.modelFilename.value,
    vaeCheckpoint: els.vaeCheckpoint.value,
    dataset: els.dataset.value,
    samplePath: els.samplePath.value,
    camera: els.camera.value,
    conditionFrames: els.conditionFrames.value,
    conditionPattern: els.conditionPattern.value,
    guidanceScale: els.guidanceScaleInput.value,
    steps: els.steps.value,
    solver: els.solver.value,
    styleId: els.styleId.value,
    seed: els.seed.value,
    start: els.start.value,
    device: els.device.value,
    outName: els.outName.value,
    useEma: els.useEma.checked,
    visualizationMode: currentVisualizationMode,
  };
}

function applyFormState(state) {
  if (!state || typeof state !== "object") return;
  if (typeof state.checkpointPreset === "string") {
    els.checkpointPreset.value = state.checkpointPreset;
  }
  if (typeof state.modelName === "string") els.modelName.value = state.modelName;
  if (typeof state.modelFilename === "string") els.modelFilename.value = state.modelFilename;
  if (typeof state.vaeCheckpoint === "string") els.vaeCheckpoint.value = state.vaeCheckpoint;
  if (typeof state.dataset === "string") {
    els.dataset.value = state.version === FORM_STATE_VERSION
      ? state.dataset
      : (state.dataset === "auto" ? "aist" : state.dataset);
  }
  if (typeof state.samplePath === "string") els.samplePath.value = state.samplePath;
  if (typeof state.camera === "string") els.camera.value = state.camera;
  if (typeof state.conditionFrames === "string") els.conditionFrames.value = state.conditionFrames;
  if (typeof state.conditionPattern === "string") {
    els.conditionPattern.value = state.conditionPattern;
  }
  if (
    typeof state.guidanceScale === "string"
    || typeof state.guidanceScale === "number"
  ) {
    renderGuidanceScale(state.guidanceScale);
  }
  if (typeof state.steps === "string") els.steps.value = state.steps;
  if (typeof state.solver === "string") els.solver.value = state.solver;
  if (typeof state.styleId === "string") setStyleSelectValue(state.styleId);
  if (typeof state.seed === "string") els.seed.value = state.seed;
  if (typeof state.start === "string") els.start.value = state.start;
  if (typeof state.device === "string") els.device.value = state.device;
  if (typeof state.outName === "string") els.outName.value = state.outName;
  if (typeof state.useEma === "boolean") {
    els.useEma.checked = state.version === FORM_STATE_VERSION ? state.useEma : true;
  }
  if (state.visualizationMode === "skeleton" || state.visualizationMode === "rigged") {
    currentVisualizationMode = state.visualizationMode;
    const input = els.visualizationModes.find((item) => item.value === state.visualizationMode);
    if (input) input.checked = true;
  }
  syncCheckpointPreset();
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
    const state = JSON.parse(raw);
    applyFormState(state);
    if (state.version !== FORM_STATE_VERSION) saveFormState();
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

function parseOptionalFloat(value) {
  if (value === "" || value == null) return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
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
  els.dataset.value = a.dataset || "aist";
  els.samplePath.value = a["sample-path"] || "";
  els.camera.value = a.camera || "";
  els.conditionFrames.value = a["cond-frames"] || "";
  els.conditionPattern.value = a["cond-pattern"] || "even";
  renderGuidanceScale(a["guidance-scale"] || defaultsCache.default_guidance_scale || 5.0);
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
    const [{ OrbitControls }, { GLTFLoader }, SkeletonUtils] = await Promise.all([
      import("three/addons/controls/OrbitControls.js"),
      import("three/addons/loaders/GLTFLoader.js"),
      import("three/addons/utils/SkeletonUtils.js"),
    ]);
    let rigTemplate = null;
    try {
      const gltf = await new GLTFLoader().loadAsync(defaultsCache.rigged_model_url);
      rigTemplate = gltf.scene;
    } catch (err) {
      if (els.riggedMode) els.riggedMode.disabled = true;
      currentVisualizationMode = "skeleton";
      els.visualizationModes.find((item) => item.value === "skeleton").checked = true;
      appendLog(`Rigged model unavailable; using skeleton view: ${err}`);
    }
    const MotionViewport = buildMotionViewportClass(
      THREE,
      OrbitControls,
      rigTemplate,
      rigTemplate ? SkeletonUtils.clone : null
    );
    MotionViewportType = MotionViewport;
    genView = new MotionViewport("genCanvas", 0x0f5f94, 0x264653);
    condView = new MotionViewport("condCanvas", 0x2d7a64, 0x28594d);
    setVisualizationMode(currentVisualizationMode, false);
    appendLog(rigTemplate ? "3D viewer ready with skeleton and rigged models." : "3D skeleton viewer ready.");
  } catch (err) {
    genView = new NullViewport();
    condView = new NullViewport();
    mldView = new NullViewport();
    stickmotionView = new NullViewport();
    MotionViewportType = null;
    appendLog(`3D viewer disabled (failed to load three.js): ${err}`);
  }
}

function setVisualizationMode(mode, persist = true) {
  currentVisualizationMode = mode === "rigged" && !els.riggedMode?.disabled
    ? "rigged"
    : "skeleton";
  for (const input of els.visualizationModes) {
    input.checked = input.value === currentVisualizationMode;
  }
  for (const viewport of [genView, condView, mldView, stickmotionView]) {
    viewport.setVisualizationMode(currentVisualizationMode);
  }
  if (persist) saveFormState();
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
  els.checkpointPreset.innerHTML = '<option value="">Custom checkpoint</option>';
  for (const preset of data.checkpoint_presets || []) {
    const opt = document.createElement("option");
    opt.value = preset.id;
    opt.textContent = preset.label;
    els.checkpointPreset.appendChild(opt);
  }
  els.vaeCheckpointList.innerHTML = "";
  for (const alias of data.vae_checkpoint_aliases || []) {
    const opt = document.createElement("option");
    opt.value = alias.path;
    opt.label = alias.label;
    els.vaeCheckpointList.appendChild(opt);
  }
  els.vaeCheckpoint.value = data.default_vae_checkpoint || "";
  els.conditionPattern.value = data.default_condition_pattern || "even";
  renderGuidanceScale(data.default_guidance_scale ?? 5.0);
  els.steps.value = data.default_steps || 8;
  els.solver.value = data.default_solver || "heun";
  els.dataset.value = data.default_dataset || "aist";
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
      const preferred = data.model_names.find((x) => x === "deployed")
        || data.model_names.find((x) => x.includes("reflow_0_solver"));
      els.modelName.value = data.default_model_name || preferred || data.model_names[data.model_names.length - 1];
    }
  }
  els.modelFilename.value = data.default_model_filename || "round0.pt";
  els.checkpointPreset.value = data.default_checkpoint_preset || "";
}

function resetArgsKeepCheckpoints() {
  els.dataset.value = defaultsCache.default_dataset || "aist";
  els.samplePath.value = "";
  els.camera.value = "";
  els.conditionFrames.value = defaultsCache.default_condition_frames == null
    ? ""
    : String(defaultsCache.default_condition_frames);
  els.conditionPattern.value = defaultsCache.default_condition_pattern || "even";
  renderGuidanceScale(defaultsCache.default_guidance_scale ?? 5.0);
  els.steps.value = String(defaultsCache.default_steps || 8);
  els.solver.value = defaultsCache.default_solver || "heun";
  els.styleId.value = defaultsCache.default_style_id == null ? "" : String(defaultsCache.default_style_id);
  els.seed.value = "";
  els.start.value = "";
  els.device.value = defaultsCache.default_device || "";
  els.outName.value = "result_smpl22.npy";
  els.useEma.checked = true;

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
  if (!currentFrameUrls || currentFrameUrls.length === 0) return Promise.resolve();
  const previewIndices = Array.isArray(currentFrameInfo.preview_indices)
    ? currentFrameInfo.preview_indices
    : [];
  const imageReady = [];
  for (let i = 0; i < currentFrameUrls.length; i += 1) {
    const u = currentFrameUrls[i];
    const img = document.createElement("img");
    img.loading = "eager";
    img.decoding = "async";
    const frameNo = previewIndices[i];
    img.alt = frameNo == null ? `Condition frame ${i + 1}` : `Condition frame ${frameNo}`;
    img.title = frameNo == null ? `Condition frame ${i + 1}` : `Condition frame ${frameNo}`;
    img.addEventListener("click", () => openLightbox(i));
    imageReady.push(new Promise((resolve) => {
      const settle = () => resolve();
      img.addEventListener("load", settle, { once: true });
      img.addEventListener("error", settle, { once: true });
    }));
    img.src = u;
    els.framesGrid.appendChild(img);
  }
  return Promise.race([
    Promise.all(imageReady),
    new Promise((resolve) => window.setTimeout(resolve, 5000)),
  ]);
}

function setVideo(url) {
  const nextUrl = typeof url === "string" ? url : "";
  if (nextUrl && nextUrl === currentVideoUrl && els.clipVideo.readyState > 0) {
    els.clipVideo.style.display = "block";
    els.noVideoMsg.style.display = "none";
    return;
  }
  els.clipVideo.pause();
  currentVideoUrl = "";
  els.clipVideo.removeAttribute("src");
  els.clipVideo.load();
  if (!url) {
    els.clipVideo.style.display = "none";
    els.noVideoMsg.style.display = "block";
    return;
  }
  currentVideoUrl = nextUrl;
  els.clipVideo.src = nextUrl;
  els.clipVideo.load();
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

async function loadComparisonResults(resultsUrl, { scroll = true } = {}) {
  const res = await fetch(resultsUrl, { cache: "no-store" });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(comparisonErrorText(data, `Result request failed (${res.status})`));
  }
  els.baselinePanel.hidden = false;
  if (MotionViewportType && mldView instanceof NullViewport) {
    mldView = new MotionViewportType("mldCanvas", 0xe36b32, 0x9e3f1e);
    stickmotionView = new MotionViewportType("stickmotionCanvas", 0x9a67b2, 0x633974);
    mldView.setVisualizationMode(currentVisualizationMode);
    stickmotionView.setVisualizationMode(currentVisualizationMode);
  }
  els.baselineSampleMeta.textContent = `${data.sample_id} | start ${data.clip_start}`;
  currentBaselineIdentity = `${data.sample_id}::${Number(data.clip_start || 0)}`;
  els.baselineCaption.textContent = data.mld_text === data.stickmotion_text
    ? data.mld_text
    : `MLD: ${data.mld_text} | StickMotion: ${data.stickmotion_text}`;
  mldView.setMotion(data.mld_motion);
  stickmotionView.setMotion(data.stickmotion_motion);
  renderStickMotionConditions(data);
  if (scroll) {
    els.baselinePanel.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function restoreComparisonForResult(data) {
  const comparison = data?.comparison;
  if (!comparison?.results_url || !currentComparisonIdentity) return;
  if (currentBaselineIdentity === currentComparisonIdentity) {
    if (comparison.matches_current_flow && comparison.download_url) {
      els.baselineBlendDownload.href = comparison.download_url;
      els.baselineBlendDownload.hidden = false;
      els.blendDownload.href = comparison.download_url;
      els.blendDownload.hidden = false;
    }
    return;
  }
  if (comparisonRestoreJobId === comparison.job_id) return;
  comparisonRestoreJobId = comparison.job_id;
  loadComparisonResults(comparison.results_url, { scroll: false })
    .then(() => {
      if (comparison.matches_current_flow && comparison.download_url) {
        els.baselineBlendDownload.href = comparison.download_url;
        els.baselineBlendDownload.hidden = false;
        els.blendDownload.href = comparison.download_url;
        els.blendDownload.hidden = false;
      }
      appendLog("Restored comparison results for this clip.");
    })
    .catch((err) => {
      if (comparisonRestoreJobId === comparison.job_id) {
        comparisonRestoreJobId = null;
      }
      appendLog(`Comparison restore failed: ${err}`);
    });
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
    const stage = data.stage || data.status || "Working";
    els.blendStatus.textContent = stage;
    setComparisonJobStage(stage);
    if (data.status === "complete") {
      setComparisonBusy(false);
      setComparisonJobStage("Comparison ready", "complete");
      els.blendDownload.href = data.download_url;
      els.blendDownload.hidden = false;
      els.blendDownload.removeAttribute("aria-disabled");
      els.blendDownload.removeAttribute("tabindex");
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
      setBlendDownloadPending(false);
      setComparisonJobStage(`Comparison failed: ${message}`, "error");
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
    setComparisonJobStage("Status unavailable; retrying...", "error");
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
  setBlendDownloadPending(true);
  setComparisonBusy(true);
  els.blendStatus.textContent = "Submitting build";
  setComparisonJobStage("Submitting comparison...");
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
        visualization_mode: currentVisualizationMode,
        device: els.comparisonDevice.value.trim() || null,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(comparisonErrorText(data, `Build request failed (${res.status})`));
    }
    activeComparisonJobId = data.job_id;
    els.blendStatus.textContent = data.stage || "Queued";
    setComparisonJobStage(data.stage || "Queued");
    appendLog(
      `Comparison blend queued: description ${currentComparisonCaption.index + 1}/${currentComparisonCaption.count}, sketches [${frames.join(", ")}], job ${data.job_id}`
    );
    pollComparisonJob(data.job_id, data.status_url);
  } catch (err) {
    setComparisonBusy(false);
    setBlendDownloadPending(false);
    els.blendStatus.textContent = `Build failed: ${err}`;
    setComparisonJobStage(`Comparison failed: ${err}`, "error");
    appendLog(`Comparison build request failed: ${err}`);
  }
}

function displayGeneratedResult(data, { preserveConditionMedia = false } = {}) {
  currentReplicateCommand = data.meta?.replicate_command || "";
  els.replicateBtn.disabled = !currentReplicateCommand;
  if (currentReplicateCommand) {
    appendLog(`replicate_command: ${currentReplicateCommand}`);
  }
  updateViewportTitles(data.meta || {});
  genView.setMotion(data.generated_motion);
  condView.setMotion(data.condition_motion);
  if (!preserveConditionMedia) {
    setVideo(data.video_url);
    setFrames(data.frame_urls || [], data.condition_frame_info || {});
  }
  setComparisonSource(data);
  latestResultAvailable = true;
  updateResultRequestControls();
}

async function displayConditionPreview(data) {
  updateViewportTitles(data.meta || {});
  if (data.condition_motion) {
    condView.setMotion(data.condition_motion);
  }
  setVideo(data.video_url);
  await setFrames(data.frame_urls || [], data.condition_frame_info || {});
}

async function fetchGenerationResult(resultUrl, mode) {
  const requestUrl = mode !== "motion"
    ? `${resultUrl}${resultUrl.includes("?") ? "&" : "?"}preview_only=true`
    : resultUrl;
  const res = await fetch(requestUrl, { cache: "no-store" });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(
      comparisonErrorText(data, `Generation result request failed (${res.status})`)
    );
  }
  if (mode === "preview") {
    await displayConditionPreview(data);
    appendLog("Condition frames are ready.");
  } else if (mode === "video") {
    await displayConditionPreview(data);
    appendLog("Condition video clip is ready.");
  } else {
    displayGeneratedResult(data, {
      preserveConditionMedia: generationPreviewLoaded,
    });
    appendLog(`result_dir: ${data.result_dir}`);
    appendLog("FlowMimic motion is ready.");
  }
  return data;
}

async function pollGenerationJob(jobId, statusUrl, resultUrl) {
  if (activeGenerationJobId !== jobId) return;
  try {
    const res = await fetch(statusUrl, { cache: "no-store" });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(
        comparisonErrorText(data, `Generation status failed (${res.status})`)
      );
    }
    if (activeGenerationJobId !== jobId) return;
    const stage = data.stage || data.status || "Working";
    if (stage !== generationStage) {
      generationStage = stage;
      appendLog(stage);
    }
    if (data.preview_ready && !generationPreviewLoaded) {
      generationPreviewLoaded = true;
      try {
        const previewData = await fetchGenerationResult(resultUrl, "preview");
        generationVideoLoaded = Boolean(previewData.video_url);
      } catch (err) {
        generationPreviewLoaded = false;
        appendLog(`Condition preview load failed: ${err}`);
      }
    }
    if (data.video_ready && !generationVideoLoaded) {
      generationVideoLoaded = true;
      try {
        const videoData = await fetchGenerationResult(resultUrl, "video");
        generationVideoLoaded = Boolean(videoData.video_url);
      } catch (err) {
        generationVideoLoaded = false;
        appendLog(`Condition video load failed: ${err}`);
      }
    }
    if (data.motion_ready && !generationMotionLoaded) {
      generationMotionLoaded = true;
      try {
        await fetchGenerationResult(resultUrl, "motion");
      } catch (err) {
        generationMotionLoaded = false;
        appendLog(`Generated motion load failed: ${err}`);
      }
    }
    if (data.status === "failed") {
      appendLog(`Generation failed: ${data.error || "unknown error"}`);
      if (data.sample_log_url) appendLog(`sample log: ${data.sample_log_url}`);
      if (data.extract_log_url) appendLog(`media log: ${data.extract_log_url}`);
      activeGenerationJobId = null;
      generationPollTimer = null;
      setResultRequestBusy(false);
      return;
    }
    if (data.status === "complete" && generationMotionLoaded) {
      if (data.preview_error) {
        appendLog(`Condition preview unavailable: ${data.preview_error}`);
      }
      activeGenerationJobId = null;
      generationPollTimer = null;
      setResultRequestBusy(false);
      return;
    }
    generationPollTimer = window.setTimeout(
      () => pollGenerationJob(jobId, statusUrl, resultUrl),
      600
    );
  } catch (err) {
    if (activeGenerationJobId !== jobId) return;
    appendLog(`Generation status unavailable; retrying: ${err}`);
    generationPollTimer = window.setTimeout(
      () => pollGenerationJob(jobId, statusUrl, resultUrl),
      1800
    );
  }
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
    condition_pattern: els.conditionPattern.value,
    guidance_scale: parseOptionalFloat(els.guidanceScaleInput.value),
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
  if (parseGuidanceScale(payload.guidance_scale) == null) {
    appendLog("CFG guidance must be between 0.0 and 5.0.");
    els.guidanceScaleInput.reportValidity();
    return;
  }

  setResultRequestBusy(true);
  if (generationPollTimer != null) {
    window.clearTimeout(generationPollTimer);
    generationPollTimer = null;
  }
  activeGenerationJobId = null;
  generationPreviewLoaded = false;
  generationVideoLoaded = false;
  generationMotionLoaded = false;
  generationStage = "";
  clearLog();
  appendLog("Starting FlowMimic generation ...");
  appendLog(
    `Condition frames: ${payload.condition_frames == null ? "checkpoint default" : payload.condition_frames}`
  );
  appendLog(`Condition pattern: ${payload.condition_pattern}`);
  appendLog(`CFG guidance: ${payload.guidance_scale.toFixed(2)}`);
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
      setResultRequestBusy(false);
      return;
    }
    activeGenerationJobId = data.job_id;
    generationStage = data.stage || "Queued";
    appendLog(`Generation job ${data.job_id}: ${generationStage}`);
    pollGenerationJob(data.job_id, data.status_url, data.result_url);
  } catch (err) {
    appendLog(`Request failed: ${err}`);
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
els.visualizationModes.forEach((input) => {
  input.addEventListener("change", () => {
    if (input.checked) setVisualizationMode(input.value);
  });
});
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
els.clipVideo.addEventListener("error", () => {
  if (!currentVideoUrl) return;
  appendLog(`Video clip failed to load: ${currentVideoUrl}`);
  els.clipVideo.style.display = "none";
  els.noVideoMsg.style.display = "block";
});
els.lightbox.addEventListener("click", (ev) => {
  const t = ev.target;
  if (t === els.lightboxImage) return;
  if (t === els.lightboxPrev || t === els.lightboxNext) return;
  closeLightbox();
});
window.addEventListener("keydown", onLightboxKeydown);
els.guidanceScaleSlider.addEventListener("input", onGuidanceSliderInput);
els.guidanceScaleSlider.addEventListener("change", onGuidanceSliderInput);
els.guidanceScaleInput.addEventListener("input", () => onGuidanceNumberInput(false));
els.guidanceScaleInput.addEventListener("change", () => onGuidanceNumberInput(true));
els.checkpointPreset.addEventListener("change", () => {
  applyCheckpointPreset(els.checkpointPreset.value);
});
for (const el of [els.modelName, els.modelFilename]) {
  el.addEventListener("input", syncCheckpointPreset);
  el.addEventListener("change", syncCheckpointPreset);
}
[
  els.checkpointPreset,
  els.modelName,
  els.modelFilename,
  els.vaeCheckpoint,
  els.dataset,
  els.samplePath,
  els.camera,
  els.conditionFrames,
  els.conditionPattern,
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
