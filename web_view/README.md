# FlowMimic Web View

Simple FastAPI web app for flow sampling + condition-media extraction + visualization.

## Install

```bash
pip install fastapi uvicorn

# Required only when comparison text is edited.
conda run -n mld python -m spacy download en_core_web_sm
```

## Run

From project root:

```bash
uvicorn web_view.app:app --host 0.0.0.0 --port 8000 --reload
```

Open:

```text
http://localhost:8000/flowmimic/
```

URL prefix is configurable with env var `FLOWMIMIC_BASE_PATH` (default: `/flowmimic`).

## Notes

- Backend calls:
  - `flowmimic/scripts/sample_flow.py`
  - `flowmimic/tools/extract_cond_media.py`
- `Condition Frames` selects the condition-token count; leaving it at `checkpoint default` preserves the checkpoint's evaluation density.
- `Checkpoint Preset` exposes stable aliases for the selected deployed models. `Deployed Round 0` resolves to `checkpoints/flow/deployed/round0.pt`; `Deployed Reflow Round 1` resolves to `checkpoints/flow/deployed/reflow1.pt`. The server recreates these git-ignored symlinks when their local checkpoint targets are available. Round 1 uses the same sampling path as Round 0 and defaults to one-step Heun inference. Both presets use EMA and CFG `5.0`.
- The matched deployed autoencoder is also available as `checkpoints/vqvae/deployed/motion_vqvae.pt`. Normally leave `VAE Checkpoint` blank so the sampler reads the matched VQ-VAE and latent-statistics paths from flow-checkpoint metadata.
- `Condition Pattern` selects even, random, or boundary-gap condition timestamps. The chosen pattern and indices are preserved in `result_meta.json` and the replicate command.
- `CFG Guidance` supports synchronized slider and exact numeric entry from 0.0 to 5.0. Scale 1.0 preserves the conditional prediction, while 5.0 is the selected deployment scale for the current Round 0 and Reflow Round 1 checkpoints.
- The web module currently assumes output root is `output/flow`.
- **Generate** starts a persisted background job under `output/flow/generation_jobs/`. Once the sampler resolves the clip and condition indices, condition-media extraction runs in parallel with flow inference, so the page can show the selected frames and video before the generated motion is ready.
- **Load Last** restores the run referenced by `output/flow/last` without rerunning sampling or condition-media extraction. It also restores the newest completed MLD/StickMotion comparison for the same sample path and clip start.
- Web-generated motion runs are marked in `result_meta.json`; the server keeps the newest 10 and removes older web runs after a successful generation. Set `FLOWMIMIC_WEB_RESULT_RETENTION` to change this limit. CLI-generated runs and other research outputs are not pruned.
- Generated, condition, MLD, and StickMotion motions are rendered at 30 FPS with mouse orbit controls. The 3D Motion switch changes every viewport between the SMPL22 skeleton and the calibrated rig in `web_view/assets/smpl22_rigged_calibrated.glb`; the selection is persisted in the browser. The original `smpl22_rigged.glb` in the same asset folder is used automatically when the calibrated asset is absent.
- After an AIST++ test/validation motion is generated at length 196, use **Generate Comparison** in the 3D Motion header. The modal randomly selects a camera-matched description, supports rerolling or editing it, and accepts three distinct clip-local StickMotion sketch frames with source-frame offsets.
- The confirmed description is shared by MLD and StickMotion. If it was edited, the comparison worker regenerates StickMotion's POS/lemma tokens with `flowmimic/tools/process_aist_text_tokens.py`; the MLD Python environment must provide spaCy and `en_core_web_sm`.
- Comparison creation runs MLD, StickMotion, and Blender as a persisted background job under `output/flow/comparison_jobs/`. It reuses the exact FlowMimic motion already displayed in the web view and builds the `.blend` with the currently selected skeleton or rigged-model visualization. MLD uses the selected retrained AIST-only diffusion checkpoint `runs/mld/mld/aist_ik263_mld_196_aistmvh_vae/checkpoints/epoch=2499.ckpt` with its AIST+MVHumanNet VAE and matched normalization assets. StickMotion uses the retrained no-locus checkpoint `runs/stickmotion/human_ml3d/aist_remodiffuse_no_locus_260730/epoch=591-step=25644.ckpt`; its ground-truth trajectory branch is disabled. MLD and StickMotion viewer results remain visible when FlowMimic regenerates the same sample path and clip start; the old `.blend` download is hidden because that scene contains the previous FlowMimic motion.
- When the job completes, the page adds MLD and StickMotion motion viewers plus text and sketch conditions, and enables the `.blend` download in both the modal and results panel.
- Comparison jobs use GPU 0 by default. The comparison dialog can override both baseline inference devices with `cpu`, `cuda`, or `cuda:<index>`. For the default path, override baseline GPU indices with `FLOWMIMIC_COMPARISON_MLD_GPU` and `FLOWMIMIC_COMPARISON_STICKMOTION_GPU`; override selected artifacts with `FLOWMIMIC_COMPARISON_MLD_CHECKPOINT`, `FLOWMIMIC_COMPARISON_MLD_CONFIG`, `FLOWMIMIC_COMPARISON_MLD_ASSETS_CONFIG`, `FLOWMIMIC_COMPARISON_STICKMOTION_CHECKPOINT`, and `FLOWMIMIC_COMPARISON_STICKMOTION_CONFIG`; override Blender with `FLOWMIMIC_BLENDER`.
