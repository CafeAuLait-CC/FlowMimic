# FlowMimic Web View

Simple FastAPI web app for flow sampling + condition-media extraction + visualization.

## Install

```bash
pip install fastapi uvicorn
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
- The web module currently assumes output root is `output/flow`.
- Generated and condition motions are rendered as SMPL22 skeletons at 30 FPS with mouse orbit controls.
