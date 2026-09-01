#!/usr/bin/env python3
"""Cache-free MediaPipe BODY25 conditions for FlowMimic evaluation."""

import argparse
import atexit
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import zlib

import numpy as np


class RuntimeMediaPipeAISTProvider:
    """Extract AIST conditions in a persistent subprocess and cache them in RAM."""

    def __init__(
        self,
        *,
        python_executable,
        model_path,
        video_dir,
        max_target_frames,
        confidence_threshold=0.4,
        log_progress=True,
    ):
        self.python_executable = str(Path(python_executable).resolve())
        self.model_path = str(Path(model_path).resolve())
        self.video_dir = Path(video_dir).resolve()
        self.max_target_frames = int(max_target_frames)
        self.confidence_threshold = float(confidence_threshold)
        self.log_progress = bool(log_progress)
        self._cache = {}
        self._process = None
        self.extracted_videos = 0
        self.extraction_seconds = 0.0
        atexit.register(self.close)

    def _start(self):
        if self._process is not None and self._process.poll() is None:
            return
        script = Path(__file__).resolve()
        self._process = subprocess.Popen(
            [
                self.python_executable,
                str(script),
                "--worker",
                "--model",
                self.model_path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def _video_path(self, motion_path, camera):
        name = Path(motion_path).stem
        camera = str(camera or "01")
        video_name = name.replace("_cAll_", f"_c{camera}_") + ".mp4"
        path = self.video_dir / video_name
        if not path.is_file():
            raise FileNotFoundError(f"AIST video not found for MediaPipe: {path}")
        return path

    def __call__(
        self,
        motion_path,
        *,
        camera=None,
        src_fps=None,
        target_fps=None,
        return_conf=False,
    ):
        if target_fps is None:
            raise ValueError("target_fps is required for runtime MediaPipe extraction")
        video_path = self._video_path(motion_path, camera)
        key = (str(video_path), float(target_fps), self.max_target_frames)
        if key not in self._cache:
            self._cache[key] = self._extract(video_path, target_fps)
        coords, vis, conf = self._cache[key]
        if return_conf:
            return coords, vis, conf
        return coords, vis

    def _extract(self, video_path, target_fps):
        self._start()
        request = {
            "video": str(video_path),
            "target_fps": float(target_fps),
            "max_frames": self.max_target_frames,
        }
        started = time.perf_counter()
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        self._process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self._process.stdin.flush()
        response_line = self._process.stdout.readline()
        if not response_line:
            return_code = self._process.poll()
            raise RuntimeError(
                "MediaPipe worker stopped without returning a response "
                f"(returncode={return_code})"
            )
        response = json.loads(response_line)
        if not response.get("ok"):
            raise RuntimeError(
                f"MediaPipe extraction failed for {video_path}: "
                f"{response.get('error', 'unknown worker error')}"
            )
        shape = tuple(int(value) for value in response["shape"])
        packed = base64.b64decode(response["data"])
        body25 = np.frombuffer(zlib.decompress(packed), dtype=np.float32).reshape(shape)
        coords, vis, conf = _prepare_flowmimic_body25(
            body25,
            confidence_threshold=self.confidence_threshold,
        )
        self.extracted_videos += 1
        self.extraction_seconds += time.perf_counter() - started
        if self.log_progress and (
            self.extracted_videos == 1 or self.extracted_videos % 25 == 0
        ):
            average = self.extraction_seconds / self.extracted_videos
            print(
                f"MediaPipe runtime extraction: {self.extracted_videos} videos, "
                f"{average:.2f} s/video",
                flush=True,
            )
        return coords, vis, conf

    def preload(self, motion_camera_pairs, *, target_fps, workers=1):
        """Extract unique videos concurrently while retaining results only in RAM."""
        unique = []
        seen = set()
        for motion_path, camera in motion_camera_pairs:
            video_path = self._video_path(motion_path, camera)
            key = (str(video_path), float(target_fps), self.max_target_frames)
            if key in self._cache or key in seen:
                continue
            seen.add(key)
            unique.append((motion_path, camera))
        if not unique:
            return
        workers = max(1, min(int(workers), len(unique)))
        if workers == 1:
            for motion_path, camera in unique:
                self(
                    motion_path,
                    camera=camera,
                    target_fps=target_fps,
                    return_conf=True,
                )
            return

        chunks = [unique[index::workers] for index in range(workers)]
        counter = {"done": 0}
        lock = threading.Lock()
        started = time.perf_counter()

        def extract_chunk(chunk):
            child = RuntimeMediaPipeAISTProvider(
                python_executable=self.python_executable,
                model_path=self.model_path,
                video_dir=self.video_dir,
                max_target_frames=self.max_target_frames,
                confidence_threshold=self.confidence_threshold,
                log_progress=False,
            )
            try:
                for motion_path, camera in chunk:
                    child(
                        motion_path,
                        camera=camera,
                        target_fps=target_fps,
                        return_conf=True,
                    )
                    with lock:
                        counter["done"] += 1
                        done = counter["done"]
                        if done == 1 or done % 25 == 0 or done == len(unique):
                            elapsed = max(time.perf_counter() - started, 1e-6)
                            rate = done / elapsed
                            remaining = (len(unique) - done) / max(rate, 1e-6)
                            print(
                                "MediaPipe RAM preload: "
                                f"{done}/{len(unique)} videos, {rate:.2f} videos/s, "
                                f"ETA {remaining / 60.0:.1f} min",
                                flush=True,
                            )
                return child._cache
            finally:
                child.close()
                atexit.unregister(child.close)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for cache in executor.map(extract_chunk, chunks):
                self._cache.update(cache)
        self.extracted_videos += len(unique)
        self.extraction_seconds += time.perf_counter() - started

    def close(self):
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write(json.dumps({"stop": True}) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)


def _prepare_flowmimic_body25(body25, confidence_threshold):
    coords = np.asarray(body25[..., :2], dtype=np.float32).copy()
    conf = np.asarray(body25[..., 2], dtype=np.float32).copy()
    coords[~np.isfinite(coords)] = 0.0
    conf[~np.isfinite(conf)] = 0.0
    conf = np.clip(conf, 0.0, 1.0)
    coords[..., 1] *= -1.0
    visible = conf >= float(confidence_threshold)

    if len(coords) and visible[0, 8]:
        pelvis = coords[0, 8].copy()
    elif len(coords) and visible[0].any():
        pelvis = coords[0, visible[0]].mean(axis=0)
    else:
        pelvis = np.zeros(2, dtype=np.float32)
    coords -= pelvis[None, None, :]
    return coords, visible.astype(np.float32), conf


def _extract_video(landmarker, video_path, target_fps, max_frames):
    import cv2
    import mediapipe as mp

    from compare_pose_extractors import mediapipe_to_body25

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_fps <= 0.0:
        capture.release()
        raise RuntimeError(f"Video reports invalid FPS {source_fps}: {video_path}")

    frames = []
    source_index = 0
    target_index = 0
    previous_timestamp = -1
    next_source_index = 0
    try:
        while target_index < max_frames:
            ok, bgr = capture.read()
            if not ok:
                break
            if source_index < next_source_index:
                source_index += 1
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = max(
                previous_timestamp + 1,
                int(round(source_index * 1000.0 / source_fps)),
            )
            result = landmarker.detect_for_video(image, timestamp_ms)
            if result.pose_landmarks:
                frame = mediapipe_to_body25(
                    result.pose_landmarks[0],
                    width,
                    height,
                )
            else:
                frame = np.zeros((25, 3), dtype=np.float32)
            frames.append(frame)
            previous_timestamp = timestamp_ms
            target_index += 1
            next_source_index = int(round(target_index * source_fps / target_fps))
            source_index += 1
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return np.stack(frames).astype(np.float32, copy=False)


def _worker(model_path):
    import mediapipe as mp

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("stop"):
                break
            # VIDEO mode carries temporal tracking state and enforces globally
            # increasing timestamps. Each AIST file is an independent stream,
            # so reset the landmarker at video boundaries.
            with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
                body25 = _extract_video(
                    landmarker,
                    request["video"],
                    float(request["target_fps"]),
                    int(request["max_frames"]),
                )
                packed = zlib.compress(body25.tobytes(order="C"), level=1)
                response = {
                    "ok": True,
                    "shape": list(body25.shape),
                    "data": base64.b64encode(packed).decode("ascii"),
                }
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    if not args.worker:
        parser.error("This module is launched as a persistent --worker by eval_flow.py")
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    _worker(args.model)


if __name__ == "__main__":
    main()
