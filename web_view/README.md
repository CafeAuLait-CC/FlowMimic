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
- `Condition Frames` selects a true uniformly spaced condition-token count.
  Leaving it at `checkpoint default` preserves the checkpoint's evaluation density.
- The web module currently assumes output root is `output/flow`.
- **Load Last** restores the run referenced by `output/flow/last` without rerunning sampling or condition-media extraction.
- Generated and condition motions are rendered as SMPL22 skeletons at 30 FPS with mouse orbit controls.
- After an AIST++ test/validation motion is generated at length 196, use **Generate Comparison** in the 3D Motion header. The modal randomly selects a camera-matched description, supports rerolling or editing it, and accepts three distinct clip-local StickMotion sketch frames with source-frame offsets.
- The confirmed description is shared by MLD and StickMotion. If it was edited, the comparison worker regenerates StickMotion's POS/lemma tokens with `flowmimic/tools/process_aist_text_tokens.py`; the MLD Python environment must provide spaCy and `en_core_web_sm`.
- Comparison creation runs MLD, StickMotion, and Blender as a persisted background job under `output/flow/comparison_jobs/`. It reuses the exact FlowMimic motion already displayed in the web view.
- When the job completes, the page adds MLD and StickMotion motion viewers plus text and sketch conditions, and enables the `.blend` download in both the modal and results panel.
- Comparison jobs use GPU 0 by default. Override baseline devices with `FLOWMIMIC_COMPARISON_MLD_GPU` and `FLOWMIMIC_COMPARISON_STICKMOTION_GPU`; override Blender with `FLOWMIMIC_BLENDER`.
