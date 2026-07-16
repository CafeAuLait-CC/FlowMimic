# FlowMimic SMPL Visualization

This folder adapts the SMPL fitting and Blender rendering pipeline from MLD:
<https://github.com/ChenFengYe/motion-latent-diffusion>.

Set the Python executable for your `mld` environment once per shell:

```bash
MLD_PYTHON="/path/to/miniconda3/envs/mld/bin/python"
```

The input is FlowMimic's `result_smpl22.npy`, which stores `(T, 22, 3)` joints in Blender Z-up space. The fitting script converts it back to Y-up before SMPLify, then writes a fitted SMPL mesh sequence and metadata:

```bash
"$MLD_PYTHON" visualize/fit_smpl.py \
  --input output/flow/last/result_smpl22.npy \
  --out-dir output/flow/last/visualize \
  --device auto
```

To use a custom SMPL definition, place it at `motion-latent-diffusion/deps/smpl_models/smpl/SMPL_CUSTOM.pkl` and pass `--gender custom`. The file must have the same SMPL model-definition structure as `SMPL_FEMALE.pkl`, including `v_template`, `shapedirs`, `posedirs`, `J_regressor`, `weights`, `kintree_table`, and `f`.

If your custom pickle is a saved SMPL output with fields like `betas`, `vertices`, and `joints`, convert it first:

```bash
"$MLD_PYTHON" visualize/make_custom_smpl.py \
  --shape-pkl motion-latent-diffusion/deps/smpl_models/smpl/SMPL_CUSTOM.pkl \
  --base motion-latent-diffusion/deps/smpl_models/smpl/SMPL_FEMALE.pkl \
  --out motion-latent-diffusion/deps/smpl_models/smpl/SMPL_CUSTOM.pkl \
  --overwrite
```

This backs up the old file as `SMPL_CUSTOM.pkl.bak` and writes a real SMPL model-definition pickle where `beta=0` is the custom shape.

Render the fitted mesh with Blender:

```bash
blender --background --python visualize/render_mesh_blender.py -- \
  --mesh output/flow/last/visualize/result_smpl22/result_smpl22_mesh.npy \
  --faces output/flow/last/visualize/result_smpl22/result_smpl22_faces.npy \
  --out output/flow/last/visualize/result_smpl22/result_smpl22_smpl.mp4 \
  --res med \
  --fps 30
```

To render with all visible GPUs, pass `--render-device all` through the wrapper, or `--device all` when calling `render_mesh_blender.py` directly. Do not set `CUDA_VISIBLE_DEVICES=0` if you want Blender to see both GPUs; use no CUDA visibility override or `CUDA_VISIBLE_DEVICES=0,1`.

Rendering can also be split across multiple Blender processes. Use `--render-workers 0` to start one worker per render device:

```bash
"$MLD_PYTHON" visualize/run_visualization.py \
  --input output/flow/last/result_smpl22.npy \
  --fit-python "$MLD_PYTHON" \
  --device cuda \
  --optimizer adam \
  --num-smplify-iters 30 \
  --render-device all \
  --render-workers 0 \
  --overwrite
```

Each render worker handles an independent frame range, so this is safe for animation output. It can improve GPU utilization, but each Blender process has its own startup overhead and GPU memory allocation.

Or run both stages through the wrapper:

```bash
"$MLD_PYTHON" visualize/run_visualization.py \
  --input output/flow/last/result_smpl22.npy \
  --fit-python "$MLD_PYTHON" \
  --blender blender
```

For a much faster visualization pass, use Adam and fewer iterations:

```bash
"$MLD_PYTHON" visualize/run_visualization.py \
  --input output/flow/last/result_smpl22.npy \
  --fit-python "$MLD_PYTHON" \
  --device cuda:0 \
  --optimizer adam \
  --num-smplify-iters 30 \
  --overwrite
```

`--optimizer lbfgs --num-smplify-iters 100` is closer to the original MLD behavior but is much slower.

For CPU parallel fitting, split the sequence across workers:

```bash
"$MLD_PYTHON" visualize/run_visualization.py \
  --input output/flow/last/result_smpl22.npy \
  --fit-python "$MLD_PYTHON" \
  --device cpu \
  --optimizer adam \
  --num-smplify-iters 30 \
  --fit-workers 16 \
  --worker-threads 4 \
  --overwrite
```

Parallel fitting warm-starts only within each chunk, so it can be slightly less temporally consistent at worker boundaries. It is mainly intended for fast visualization. Avoid multiple workers on one GPU unless you have enough memory and have measured that it helps.

Parallel fitting shows one parent-process progress bar. MLD's inner per-frame optimizer bars and worker log lines are suppressed by default; pass `--show-inner-progress` or `--worker-log` only when debugging.

For multi-GPU fitting, use `--device cuda` without a device number. This expands to all CUDA-visible GPUs and uses one fitting worker per GPU by default:

```bash
"$MLD_PYTHON" visualize/run_visualization.py \
  --input output/flow/last/result_smpl22.npy \
  --fit-python "$MLD_PYTHON" \
  --device cuda \
  --optimizer adam \
  --num-smplify-iters 30 \
  --render-device all \
  --overwrite
```

Use `--fit-devices cuda:0,cuda:1` to select a subset, or set `CUDA_VISIBLE_DEVICES=0,1` before the command. Use `--device cuda:0` when you want the old single-GPU behavior.

For a quick smoke test:

```bash
OMP_NUM_THREADS=128 MKL_NUM_THREADS=128 \
"$MLD_PYTHON" visualize/run_visualization.py \
  --input output/flow/last/result_smpl22.npy \
  --fit-python "$MLD_PYTHON" \
  --device cpu \
  --num-threads 128 \
  --max-frames 3 \
  --num-smplify-iters 2 \
  --res low \
  --overwrite
```

Notes:

- MLD's README renders from the original `.npy` folder after fitting, but `fit.py` actually writes `*_mesh.npy` inside the fitting save folder. These scripts keep the fitted mesh path explicit.
- MLD's standalone `deps/smpl_models/smpl.faces` does not match the `smplx` model faces used by this local SMPL fit. The fitting script writes a `*_faces.npy` sidecar, and the renderer uses that sidecar to avoid opaque sheet artifacts.
- Blender rendering must run through Blender's Python (`blender --background --python ...`), because `bpy` is not importable from normal Python.
- The renderer avoids MLD's older Blender scene helpers and uses Blender 5-compatible APIs.
